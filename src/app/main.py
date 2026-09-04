"""
ChemGIME Streamlit Application
===============================
Main Streamlit UI for the GEM-Centric Enzyme Identifier.

Input modes
-----------
XML only   — SBML/XML GEM as the organism reaction source (original behaviour)
.tbl only  — Gapseq all-Reactions.tbl as the primary organism source
             (requires a reaction-SMILES lookup file)
Hybrid     — Both files: XML provides SMILES, .tbl enriches GPR/BLAST evidence

Launch with:
    streamlit run src/app/main.py
"""

import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np
import streamlit as st

# --- Ensure project root is on sys.path for imports ---
_APP_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _APP_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.core import (
    BIGG_MET_TO_SMILES_FILE,
    COFACTOR_SMILES_FILE,
    MODELSEED_RXN_SMILES_FILE,
    REPORTS_DIR,
    FingerprintType,
    build_organism_database,
    build_organism_database_from_tbl,
    calculate_fingerprints_parallel,
    generate_html_report,
    get_closest_rxns,
    load_bigg_to_smiles_map,
    load_cofactor_set,
    load_gapseq_evidence,
    load_rxn_smiles_map,
    load_data,
    predict_ec_weighted,
    _calculate_drfp_fingerprint,
    _calculate_qrfp_fingerprint,
    calibrate_confidence,
    tanimoto_distance_plot,
    distance_histogram,
    ec_confidence_radar,
)
from src.core.gapseq_parser import parse_gapseq_tbl

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# Suppress noisy RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.warning")

# ── Input mode constants ─────────────────────────────────────────────────────
_MODE_XML = "xml_only"    # SBML/XML GEM only (original path)
_MODE_TBL = "tbl_only"   # Gapseq .tbl as primary reaction source
_MODE_HYBRID = "hybrid"  # XML for SMILES + .tbl for GPR/BLAST enrichment


# ── Streamlit-Cached Wrappers ────────────────────────────────────────────────

@st.cache_data
def cached_load_cofactor_set(cofactor_file: Path) -> Set[str]:
    return load_cofactor_set(cofactor_file)


@st.cache_data
def cached_load_bigg_to_smiles_map(map_file: Path) -> Dict[str, str]:
    return load_bigg_to_smiles_map(map_file)


@st.cache_data
def cached_build_organism_database(
    _gem_file_obj: Any,
    met_to_smiles: Dict[str, str],
    cofactor_set: Set[str],
):
    """XML → standard DataFrame (original path)."""
    return build_organism_database(_gem_file_obj, met_to_smiles, cofactor_set)


def _save_uploaded_to_tmp(uploaded_file, suffix: str) -> Path:
    """Write an uploaded file to a named temp file so Path-based parsers can read it."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()
    uploaded_file.seek(0)
    return Path(tmp.name)


# ── Organism Database Building ───────────────────────────────────────────────

def _build_gem_db(
    input_mode: str,
    gem_file,
    tbl_df,
    met_to_smiles_map: Dict[str, str],
    cofactor_set: Set[str],
    rxn_smiles_map: Dict[str, Tuple[str, str]],
):
    """Route to the correct organism database builder based on input_mode.

    Returns the standard pd.DataFrame (or None on failure) and a human-readable
    string describing what was built (for the summary panel).
    """
    if input_mode == _MODE_XML:
        # --- Original SBML path ---
        df = cached_build_organism_database(gem_file, met_to_smiles_map, cofactor_set)
        source = f"SBML/XML: {gem_file.name}"
        return df, source

    elif input_mode == _MODE_TBL:
        # --- .tbl-only path ---
        df = build_organism_database_from_tbl(tbl_df, rxn_smiles_map, cofactor_set)
        source = "Gapseq .tbl (standalone)"
        return df, source

    elif input_mode == _MODE_HYBRID:
        # --- Hybrid: XML for SMILES, .tbl for GPR enrichment ---
        # 1. Build from XML (gets SMILES)
        df = cached_build_organism_database(gem_file, met_to_smiles_map, cofactor_set)
        if df is None:
            return None, "SBML/XML (failed)"
        # 2. Overlay GPR strings from .tbl (where available)
        if tbl_df is not None:
            from src.core.gapseq_parser import build_gene_reaction_map, build_reaction_gpr_map
            gene_map = build_gene_reaction_map(tbl_df)
            tbl_gpr = build_reaction_gpr_map(gene_map)
            # Overwrite GPR only where .tbl has evidence (keeps XML GPR as fallback)
            df["gpr"] = df["reaction_id"].map(tbl_gpr).fillna(df["gpr"])
        source = f"Hybrid (SBML: {gem_file.name} + Gapseq .tbl GPR enrichment)"
        return df, source

    raise ValueError(f"Unknown input_mode: {input_mode!r}")


# ── Per-Query Analysis ───────────────────────────────────────────────────────

def run_single_analysis(
    gem_db_df,
    query_file: Any,
    cofactor_set: Set[str],
    num_results: int,
    ec_threshold: float,
    *,
    blast_features=None,
    organism_label: str = "organism",
    fp_type: FingerprintType = FingerprintType.DRFP,
) -> Optional[dict]:
    """Run fingerprint search + EC prediction for one query against the organism DB.

    Returns a rich dict of artifacts, or None on failure.
    """
    query_filename = query_file.name
    fp_label = fp_type.value.upper()
    t0 = time.time()

    try:
        # 1. Fingerprint the organism DB
        db_fps, valid_fp_indices = calculate_fingerprints_parallel(gem_db_df, fp_type=fp_type)
        if db_fps is None or not valid_fp_indices:
            st.error(f"Could not generate any valid {fp_label} fingerprints from **{organism_label}**.")
            return None

        fp_yield = len(valid_fp_indices) / len(gem_db_df) if len(gem_db_df) > 0 else 0.0

        # 2. Load and fingerprint the query
        query_df = load_data(query_file).rename(columns={"reaction": "id"}).iloc[0:1]
        if query_df is None or query_df.empty:
            return None

        query_row = query_df.iloc[0]
        _fp_fn = (
            _calculate_drfp_fingerprint
            if fp_type == FingerprintType.DRFP
            else _calculate_qrfp_fingerprint
        )
        query_fp = _fp_fn((
            query_row.get("left_smiles"),
            query_row.get("right_smiles"),
        ))
        if query_fp is None:
            st.error(f"Could not generate a {fp_label} fingerprint for **{query_filename}**.")
            return None

        query_fp_array = query_fp.reshape(1, -1)

        # 3. Rank reactions
        df_rank = get_closest_rxns(
            query_fp_array, db_fps, gem_db_df, valid_fp_indices,
            fp_type=fp_type.value,
        )

        # 4. Confidence calibration (F6)
        df_rank = calibrate_confidence(df_rank, blast_features)

        # 5. EC Prediction
        ec_predictions = predict_ec_weighted(df_rank, ec_threshold)

        # 6. HTML Report
        html_content = generate_html_report(
            df_rank, query_df, organism_label, query_filename, num_results,
            ec_predictions, cofactor_set,
        )

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        org_stem = os.path.splitext(organism_label)[0]
        q_stem = os.path.splitext(query_filename)[0]
        report_path = REPORTS_DIR / f"report_{org_stem}_{q_stem}.html"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        elapsed = time.time() - t0
        best_dist = float(df_rank["tan_distance"].iloc[0]) if not df_rank.empty else None
        best_conf = (
            float(df_rank["confidence"].iloc[0])
            if "confidence" in df_rank.columns and not df_rank.empty
            else None
        )

        return {
            "report_path": report_path,
            "html_content": html_content,
            "df_rank": df_rank,
            "ec_predictions": ec_predictions,
            "query_df": query_df,
            "organism_label": organism_label,
            "query_filename": query_filename,
            "fp_yield": fp_yield,
            "n_reactions": len(gem_db_df),
            "best_tanimoto": best_dist,
            "best_confidence": best_conf,
            "elapsed_s": elapsed,
        }

    except Exception as e:
        logger.error(f"Unexpected error for query {query_filename}: {e}", exc_info=True)
        st.error(f"Unexpected error while processing **{query_filename}**. See logs.")
        return None


# ── Result Renderers ─────────────────────────────────────────────────────────

def _render_interactive_section(df_rank, ec_predictions, num_results, idx):
    """Plotly visualizations (F7)."""
    st.markdown("**Interactive Distance Landscape**")
    tab_scatter, tab_hist, tab_ec = st.tabs([
        "Distance Scatter", "Distribution", "EC Confidence"
    ])
    with tab_scatter:
        plot_html = tanimoto_distance_plot(
            df_rank, top_n=min(num_results, 200), highlight_n=10,
        )
        st.components.v1.html(plot_html, height=520, scrolling=False)
    with tab_hist:
        hist_html = distance_histogram(df_rank)
        st.components.v1.html(hist_html, height=420, scrolling=False)
    with tab_ec:
        radar_html = ec_confidence_radar(ec_predictions)
        st.components.v1.html(radar_html, height=420, scrolling=False)


def _render_confidence_section(df_rank, num_results):
    """Calibrated confidence table (F6)."""
    top = df_rank.head(min(num_results, 20))
    if "confidence" not in top.columns:
        return
    st.markdown("**Calibrated Confidence (Top 20)**")
    st.caption(
        "Confidence combines Tanimoto chemical similarity with BLAST genomic "
        "evidence (pident, evalue) via a logistic calibration model."
    )
    display_cols = ["reaction_id", "name", "gpr", "ec", "tan_distance", "confidence"]
    display_cols = [c for c in display_cols if c in top.columns]
    styled = top[display_cols].copy()
    styled["confidence"] = styled["confidence"].apply(lambda x: f"{x:.1%}")
    styled["tan_distance"] = styled["tan_distance"].apply(lambda x: f"{x:.4f}")
    st.dataframe(styled, use_container_width=True, hide_index=True)


def _render_export_buttons(df_rank, ec_predictions, num_results, gem_label, query_fn, i):
    """JSON + CSV download buttons (F5)."""
    from src.core.export import _ec_predictions_to_dict, _sanitise_df_for_export

    gem_stem = os.path.splitext(gem_label)[0]
    q_stem = os.path.splitext(query_fn)[0]
    top_df = df_rank.head(num_results)
    clean_df = _sanitise_df_for_export(top_df)
    if "confidence" in top_df.columns:
        clean_df["confidence"] = top_df["confidence"].values[:len(clean_df)]

    payload = {
        "metadata": {
            "tool": "ChemGIME",
            "version": "2.0",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "gem_file": gem_label,
            "query_file": query_fn,
            "n_results": len(clean_df),
        },
        "ec_predictions": _ec_predictions_to_dict(ec_predictions),
        "top_reactions": clean_df.to_dict(orient="records"),
    }
    json_data = json.dumps(payload, indent=2, default=str)
    csv_data = clean_df.to_csv(index=False)

    col1, col2 = st.columns(2)
    col1.download_button(
        "Download JSON", json_data,
        file_name=f"{gem_stem}_{q_stem}_results.json",
        mime="application/json", key=f"dl_json_{i}",
    )
    col2.download_button(
        "Download CSV", csv_data,
        file_name=f"{gem_stem}_{q_stem}_results.csv",
        mime="text/csv", key=f"dl_csv_{i}",
    )


@st.cache_data(show_spinner=False)
def _cached_attribution(q_left: str, q_right: str, t_left: str, t_right: str) -> Optional[str]:
    """Cache DRFP atom-map encoding (slow) keyed on the four SMILES strings."""
    try:
        from src.core.interpretability import explain_reaction_similarity
        return explain_reaction_similarity((q_left, q_right), (t_left, t_right))
    except Exception:
        return None


def _render_attribution_section(result: dict, section_key: str) -> None:
    """Atom-level heat-map for the top hit.

    Uses st.checkbox instead of st.expander to avoid the
    'expanders may not be nested' Streamlit restriction.
    """
    st.markdown("**Structural Attribution** — why is the top hit similar?")
    show = st.checkbox(
        "Show atom-level heat-map (query vs top hit)",
        value=False,
        key=f"attr_toggle_{section_key}",
        help=(
            "Atoms coloured yellow→red contributed more matching DRFP circular "
            "substructure shingles. Computed once and cached per SMILES pair."
        ),
    )
    if not show:
        return

    try:
        query_df = result.get("query_df")
        df_rank = result.get("df_rank")
        if query_df is None or df_rank is None or df_rank.empty:
            st.caption("No ranking data available.")
            return

        q_row = query_df.iloc[0]
        t_row = df_rank.iloc[0]
        q_left = str(q_row.get("left_smiles") or q_row.get("raw_left_smiles") or "")
        q_right = str(q_row.get("right_smiles") or q_row.get("raw_right_smiles") or "")
        t_left = str(t_row.get("raw_left_smiles") or "")
        t_right = str(t_row.get("raw_right_smiles") or "")

        if not all([q_left, q_right, t_left, t_right]):
            st.caption("Attribution unavailable: SMILES missing from result.")
            return

        with st.spinner("Computing structural attribution…"):
            attr_html = _cached_attribution(q_left, q_right, t_left, t_right)

        if attr_html:
            n_mols = max(
                len(q_left.split(".")), len(q_right.split(".")),
                len(t_left.split(".")), len(t_right.split(".")),
            )
            height = max(400, 260 + n_mols * 220)
            st.components.v1.html(attr_html, height=height, scrolling=True)
        else:
            st.caption(
                "Attribution unavailable — check that SMILES are valid and "
                "`drfp` is installed (`pip install drfp`)."
            )
    except Exception as exc:
        st.caption(f"Attribution error: {exc}")


def _render_execution_summary(results: list, input_mode: str, organism_label: str):
    """Post-execution summary panel — one scannable table for the whole run."""
    if not results:
        return

    with st.expander("Run Summary", expanded=True):
        # ── Mode banner ──
        mode_labels = {
            _MODE_XML: "SBML/XML GEM",
            _MODE_TBL: "Gapseq .tbl (standalone)",
            _MODE_HYBRID: "Hybrid (XML + .tbl GPR)",
        }
        st.info(
            f"**Input mode:** {mode_labels.get(input_mode, input_mode)}  |  "
            f"**Organism source:** {organism_label}"
        )

        # ── Organism DB stats (from first result) ──
        r0 = results[0]
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Reactions in DB", r0["n_reactions"])
        fp_pct = f"{r0['fp_yield']:.1%}"
        col_b.metric("Fingerprint Yield", fp_pct)
        col_c.metric("Queries Run", len(results))

        # ── Per-query summary table ──
        st.markdown("**Per-Query Results**")
        rows = []
        for r in results:
            ec_l3 = "—"
            ec_conf = None
            preds = r.get("ec_predictions", {})
            if "L3" in preds:
                ec_l3 = preds["L3"][0] or "—"
                ec_conf = preds["L3"][1]

            rows.append({
                "Query": r["query_filename"],
                "Best Tanimoto Dist.": (
                    f"{r['best_tanimoto']:.4f}" if r["best_tanimoto"] is not None else "—"
                ),
                "Best Confidence": (
                    f"{r['best_confidence']:.1%}" if r["best_confidence"] is not None else "—"
                ),
                "EC L3 Prediction": ec_l3,
                "EC L3 Score": f"{ec_conf:.3f}" if ec_conf is not None else "—",
                "Time (s)": f"{r['elapsed_s']:.1f}",
            })

        st.dataframe(rows, use_container_width=True, hide_index=True)

        # ── Summary CSV download ──
        import pandas as pd
        summary_csv = pd.DataFrame(rows).to_csv(index=False)
        st.download_button(
            "Download Summary CSV",
            summary_csv,
            file_name="chemgime_run_summary.csv",
            mime="text/csv",
            key="dl_summary",
        )


# ── Main App ─────────────────────────────────────────────────────────────────

def main_app():
    st.set_page_config(layout="wide", page_title="GEM-Centric Enzyme Identifier")
    st.title("GEM-Centric Enzyme Identifier")
    st.markdown(
        "Find similar chemical reactions *within a specific organism's metabolic model* "
        "and identify the associated genes (GPRs)."
    )

    # --- Load static data ---
    try:
        met_to_smiles_map = cached_load_bigg_to_smiles_map(BIGG_MET_TO_SMILES_FILE)
        cofactor_set = cached_load_cofactor_set(COFACTOR_SMILES_FILE)
    except Exception as e:
        logger.error(f"Failed to load critical data on startup: {e}")
        st.error("Application failed to load critical data files. See logs for details.")
        st.stop()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Analysis Configuration")

        # ── Section 1: Organism source ────────────────────────────────────
        st.subheader("1. Organism Model")
        st.caption(
            "Upload **either** format, or **both** for hybrid mode.  \n"
            "• XML only → original SBML path  \n"
            "• .tbl only → Gapseq-primary (requires Reaction SMILES below)  \n"
            "• Both → XML provides SMILES; .tbl enriches GPR/BLAST evidence"
        )

        organism_gem_file = st.file_uploader(
            "Organism GEM (.xml or .sbml) — optional",
            type=["xml", "sbml"],
            accept_multiple_files=False,
            help="SBML/XML GEM from BiGG or gapseq gapfilling output.",
        )

        gapseq_tbl_file = st.file_uploader(
            "Gapseq all-Reactions.tbl — optional",
            type=["tbl", "tsv", "txt"],
            accept_multiple_files=False,
            help=(
                "Gapseq protein-all-Reactions.tbl file.  "
                "In standalone mode this IS the organism reaction database; "
                "in hybrid mode it enriches GPR strings and BLAST confidence."
            ),
        )

        rxn_smiles_file = st.file_uploader(
            "Reaction-SMILES lookup (.tsv) — required for .tbl-only mode",
            type=["tsv", "txt", "csv"],
            accept_multiple_files=False,
            help=(
                "TSV with columns: rxn | left_smiles | right_smiles.  "
                "Maps ModelSEED reaction IDs to SMILES (same namespace as Gapseq).  "
                "Not needed if an XML GEM is also uploaded."
            ),
        )

        # ── Section 2: Query reactions ────────────────────────────────────
        st.subheader("2. Query Reaction(s)")
        query_files = st.file_uploader(
            "Query file(s) (.csv or .tsv)",
            type=["csv", "tsv"],
            accept_multiple_files=True,
            help="Must have 'left_smiles' and 'right_smiles' columns.",
        )

        # ── Section 3: Output options ─────────────────────────────────────
        st.subheader("3. Output Options")
        num_results = st.number_input(
            "Top reactions to display",
            min_value=10, max_value=5000, value=1000, step=50,
        )
        fp_choice = st.selectbox(
            "Reaction fingerprint type",
            options=["DRFP", "QRFP"],
            index=0,
            help=(
                "**DRFP** — Differential Reaction Fingerprint (binary, fast). "
                "Standard method for reaction similarity.\n\n"
                "**QRFP** — Quaternary Reaction Fingerprint (4-valued, ~3x slower). "
                "Encodes substrate-only, product-only, and shared features separately. "
                "May improve discrimination for reactions with asymmetric transformations."
            ),
        )
        fp_type = FingerprintType(fp_choice.lower())
        enable_plots = st.checkbox("Show interactive plots", value=True)
        enable_export = st.checkbox("Enable JSON/CSV export", value=False)

        # ── Section 4: EC threshold ───────────────────────────────────────
        st.subheader("4. EC Prediction Options")
        ec_threshold = st.slider(
            "Distance threshold",
            min_value=0.0, max_value=1.0, value=0.5, step=0.05,
            help="Include reactions below this Tanimoto distance for EC voting.",
        )

        st.markdown("---")
        run_button = st.button("Run Analysis", type="primary")

    # ── Resolve input mode ────────────────────────────────────────────────────
    has_xml = organism_gem_file is not None
    has_tbl = gapseq_tbl_file is not None

    if has_xml and has_tbl:
        input_mode = _MODE_HYBRID
    elif has_xml:
        input_mode = _MODE_XML
    elif has_tbl:
        input_mode = _MODE_TBL
    else:
        input_mode = None  # nothing uploaded yet

    # Show live mode badge in sidebar
    if input_mode:
        badges = {
            _MODE_XML: ("blue", "Mode: SBML/XML only"),
            _MODE_TBL: ("orange", "Mode: Gapseq .tbl standalone"),
            _MODE_HYBRID: ("green", "Mode: Hybrid (XML + .tbl)"),
        }
        color, label = badges[input_mode]
        st.sidebar.markdown(f":{color}[**{label}**]")

    # ── Parse Gapseq .tbl (once, before the run loop) ────────────────────────
    tbl_df = None
    blast_features = None
    gpr_from_tbl = None

    if has_tbl:
        try:
            tbl_path = _save_uploaded_to_tmp(gapseq_tbl_file, ".tbl")
            gpr_from_tbl, blast_features = load_gapseq_evidence(tbl_path)
            # Also keep the full parsed DF for the .tbl-only builder
            tbl_df = parse_gapseq_tbl(tbl_path)
            tbl_path.unlink(missing_ok=True)
            st.sidebar.success(
                f"Gapseq loaded: **{len(gpr_from_tbl)}** reaction GPRs, "
                f"**{len(blast_features)}** BLAST features."
            )
        except Exception as e:
            logger.warning(f"Failed to parse .tbl: {e}")
            st.sidebar.warning(f"Could not parse .tbl: {e}")

    # ── Load reaction-SMILES map (for .tbl-only mode) ─────────────────────────
    rxn_smiles_map: Dict[str, tuple] = {}
    if input_mode == _MODE_TBL:
        if rxn_smiles_file is not None:
            rxn_smiles_map = load_rxn_smiles_map(rxn_smiles_file)
            st.sidebar.success(f"Reaction-SMILES map loaded: **{len(rxn_smiles_map)}** entries.")
        elif MODELSEED_RXN_SMILES_FILE.exists():
            rxn_smiles_map = load_rxn_smiles_map(MODELSEED_RXN_SMILES_FILE)
            st.sidebar.info(
                f"Using pre-installed reaction-SMILES DB: **{len(rxn_smiles_map)}** entries."
            )
        else:
            st.sidebar.warning(
                "Standalone .tbl mode requires a Reaction-SMILES lookup file.  \n"
                "Upload one above, or install `data/database/modelseed_rxn_to_smiles.tsv`."
            )

    # ── Main analysis loop ────────────────────────────────────────────────────
    if run_button:
        # Validate inputs
        if not query_files:
            st.error("Please upload at least one query file.")
            st.stop()
        if input_mode is None:
            st.error("Please upload at least one organism source (XML or .tbl).")
            st.stop()
        if input_mode == _MODE_TBL and not rxn_smiles_map:
            st.error(
                "Standalone .tbl mode requires a Reaction-SMILES lookup file.  \n"
                "Upload a TSV with `rxn | left_smiles | right_smiles` columns, "
                "or install `data/database/modelseed_rxn_to_smiles.tsv`."
            )
            st.stop()

        # ── Build organism database (once per run) ────────────────────────
        with st.spinner("Building organism reaction database..."):
            organism_label = (
                gapseq_tbl_file.name if has_tbl and not has_xml
                else (organism_gem_file.name if has_xml else "unknown")
            )
            gem_db_df, db_source = _build_gem_db(
                input_mode,
                organism_gem_file,
                tbl_df,
                met_to_smiles_map,
                cofactor_set,
                rxn_smiles_map,
            )

        if gem_db_df is None or gem_db_df.empty:
            st.error(
                f"No processable reactions could be built from **{organism_label}**.  \n"
                + ("Check that the Reaction-SMILES lookup covers the reaction IDs in your .tbl."
                   if input_mode == _MODE_TBL else
                   "Check that the GEM contains reactions with BiGG-mapped metabolites.")
            )
            st.stop()

        st.info(
            f"Organism database ready: **{len(gem_db_df)} reactions** "
            f"from {db_source}."
        )

        # ── Per-query loop ────────────────────────────────────────────────
        all_results = []
        status_ph = st.empty()
        start_time = time.time()

        for i, query_file in enumerate(query_files):
            status_ph.info(
                f"Query {i + 1}/{len(query_files)}: **{query_file.name}**"
            )

            result = run_single_analysis(
                gem_db_df,
                query_file,
                cofactor_set,
                num_results,
                ec_threshold,
                blast_features=blast_features,
                organism_label=organism_label,
                fp_type=fp_type,
            )

            if result is None:
                continue

            all_results.append(result)
            html_content = result["html_content"]
            df_rank = result["df_rank"]
            ec_predictions = result["ec_predictions"]
            q_fn = result["query_filename"]
            org_stem = os.path.splitext(organism_label)[0]
            q_stem = os.path.splitext(q_fn)[0]

            with st.expander(
                f"Report: {organism_label} vs {q_fn}",
                expanded=(len(all_results) == 1),
            ):
                # ── HTML download ──
                dl_cols = st.columns(3 if enable_export else 1)
                dl_cols[0].download_button(
                    "Download HTML Report",
                    html_content,
                    file_name=f"report_{org_stem}_{q_stem}.html",
                    mime="text/html",
                    key=f"dl_html_{i}",
                )
                if enable_export:
                    _render_export_buttons(
                        df_rank, ec_predictions, num_results,
                        organism_label, q_fn, i,
                    )

                # ── Plots ──
                if enable_plots:
                    _render_interactive_section(df_rank, ec_predictions, num_results, i)

                # ── Confidence table ──
                if "confidence" in df_rank.columns:
                    _render_confidence_section(df_rank, num_results)

                # ── Structural Attribution ──
                _render_attribution_section(result, section_key=f"{org_stem}_{q_stem}_{i}")

                st.markdown("---")
                st.markdown("**Full HTML Report**")
                st.components.v1.html(html_content, height=800, scrolling=True)

        status_ph.empty()
        total = time.time() - start_time
        st.success(
            f"Analysis complete — {len(all_results)} report(s) in {total:.2f}s."
        )

        # ── Post-execution summary ────────────────────────────────────────
        _render_execution_summary(all_results, input_mode, organism_label)


if __name__ == "__main__":
    main_app()
