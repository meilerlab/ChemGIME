"""
ChemGIME Gapseq Parser
=======================
Flexible ingestion layer for gapseq ``protein-all-Reactions.tbl`` files.

Gapseq outputs a TSV where each row represents one BLAST hit of a reference
protein against the organism's proteome for a given reaction.  The same gene
(organism protein) can catalyse multiple reactions, and the same reaction can
be supported by multiple genes — a classic many-to-many mapping.

This module:
  1. Parses the .tbl file, retaining the columns requested in the project spec
     (F2–F7, F9, F10: name, ec, bihit, qseqid, pident, evalue, qcovs, stitle).
  2. Extracts the organism protein ID from the ``stitle`` field.
  3. Builds a consolidated Gene → Reaction(s) mapping that can be expressed as
     a GPR string (suitable for SBML injection).
  4. Provides quality-tier filtering (good_blast / bad_blast / no_blast).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import pandas as pd

# Public symbol added here so callers can do:
#   from src.core.gapseq_parser import build_organism_database_from_tbl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column specification — positions are 0-indexed
# ---------------------------------------------------------------------------
_GAPSEQ_COLUMNS = [
    "rxn", "name", "ec", "bihit", "qseqid", "pident", "evalue",
    "bitscore", "qcovs", "stitle", "sstart", "send", "pathway",
    "status", "pathway.status", "dbhit", "complex", "exception",
    "complex.status",
]

# The columns the spec asks for (F2–F7, F9, F10 — 1-indexed → 0-indexed)
_SPEC_COLUMNS = ["name", "ec", "bihit", "qseqid", "pident", "evalue", "qcovs", "stitle"]

# Regex to pull the first WP_ / NP_ / YP_ / XP_ accession from ``stitle``
_PROTEIN_ID_RE = re.compile(r"([A-Z]{2}_\d+\.\d+)")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class GapseqHit:
    """One row from the .tbl file after cleaning."""

    rxn: str
    name: str
    ec: str
    bihit: str
    qseqid: str
    pident: float
    evalue: float
    qcovs: float
    stitle: str
    protein_id: Optional[str]
    status: str
    pathway: str
    bitscore: float
    complex_info: str


@dataclass
class GeneReactionMapping:
    """Consolidated mapping of one organism protein to its reaction set."""

    protein_id: str
    reaction_ids: Set[str] = field(default_factory=set)
    best_pident: float = 0.0
    best_evalue: float = 1.0
    ec_numbers: Set[str] = field(default_factory=set)

    @property
    def gpr_string(self) -> str:
        """Return a COBRA-compatible GPR string for this single gene."""
        return self.protein_id


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def _extract_protein_id(stitle: str) -> Optional[str]:
    """Extract the organism's protein accession from the BLAST subject title.

    Examples
    --------
    >>> _extract_protein_id("WP_020104528.1 dihydrolipoyl dehydrogenase [Mesomycoplasma hyorhinis]")
    'WP_020104528.1'
    """
    if not stitle or pd.isna(stitle):
        return None
    m = _PROTEIN_ID_RE.search(str(stitle))
    return m.group(1) if m else None


def parse_gapseq_tbl(
    filepath: Union[str, Path],
    *,
    status_filter: Optional[List[str]] = None,
    min_pident: float = 0.0,
    max_evalue: float = 1e10,
    min_qcovs: float = 0.0,
) -> pd.DataFrame:
    """Parse a gapseq ``protein-all-Reactions.tbl`` file into a DataFrame.

    Parameters
    ----------
    filepath : str or Path
        Path to the .tbl file.
    status_filter : list of str, optional
        Keep only rows whose ``status`` is in this list.
        Default ``None`` keeps all rows.  Typical: ``["good_blast"]``.
    min_pident : float
        Minimum percent identity to retain a hit.
    max_evalue : float
        Maximum e-value to retain a hit.
    min_qcovs : float
        Minimum query coverage (%) to retain a hit.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with columns:
        ``rxn, name, ec, bihit, qseqid, pident, evalue, bitscore, qcovs,
        stitle, protein_id, status, pathway, complex``.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Gapseq .tbl file not found: {filepath}")

    logger.info(f"Parsing gapseq file: {filepath.name}")

    # Read, skip comment lines (start with #)
    df = pd.read_csv(
        filepath,
        sep="\t",
        comment="#",
        header=0,
        names=_GAPSEQ_COLUMNS,
        dtype=str,
        on_bad_lines="warn",
    )

    initial_rows = len(df)
    logger.info(f"  Raw rows parsed: {initial_rows}")

    # --- Coerce numeric columns ---
    for col in ("pident", "evalue", "bitscore", "qcovs"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Extract protein_id from stitle ---
    df["protein_id"] = df["stitle"].apply(_extract_protein_id)

    # --- Apply quality filters ---
    if status_filter is not None:
        df = df[df["status"].isin(status_filter)]
        logger.info(f"  After status filter {status_filter}: {len(df)} rows")

    mask = (
        (df["pident"].fillna(0) >= min_pident)
        & (df["evalue"].fillna(1e30) <= max_evalue)
        & (df["qcovs"].fillna(0) >= min_qcovs)
    )
    df = df[mask].copy()
    logger.info(f"  After numeric filters (pident>={min_pident}, evalue<={max_evalue}, qcovs>={min_qcovs}): {len(df)} rows")

    # --- Retain useful columns ---
    keep_cols = [
        "rxn", "name", "ec", "bihit", "qseqid", "pident", "evalue",
        "bitscore", "qcovs", "stitle", "protein_id", "status", "pathway", "complex",
    ]
    df = df[[c for c in keep_cols if c in df.columns]].reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Gene–Reaction Mapping
# ---------------------------------------------------------------------------
def build_gene_reaction_map(
    df: pd.DataFrame,
    *,
    require_protein_id: bool = True,
) -> Dict[str, GeneReactionMapping]:
    """Build a consolidated gene → reaction(s) map from parsed gapseq data.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`parse_gapseq_tbl`.
    require_protein_id : bool
        If True (default), skip rows without a resolved protein_id.

    Returns
    -------
    dict
        Mapping of ``protein_id`` → :class:`GeneReactionMapping`.
    """
    gene_map: Dict[str, GeneReactionMapping] = {}

    for _, row in df.iterrows():
        pid = row.get("protein_id")
        if require_protein_id and (not pid or pd.isna(pid)):
            continue

        if pid not in gene_map:
            gene_map[pid] = GeneReactionMapping(protein_id=pid)

        mapping = gene_map[pid]
        mapping.reaction_ids.add(row["rxn"])

        pident = row.get("pident", 0.0)
        evalue = row.get("evalue", 1.0)
        if pd.notna(pident) and pident > mapping.best_pident:
            mapping.best_pident = pident
        if pd.notna(evalue) and evalue < mapping.best_evalue:
            mapping.best_evalue = evalue

        ec_raw = row.get("ec", "")
        if ec_raw and pd.notna(ec_raw):
            for ec in str(ec_raw).split("/"):
                ec_clean = ec.strip()
                if ec_clean:
                    mapping.ec_numbers.add(ec_clean)

    logger.info(
        f"Built gene-reaction map: {len(gene_map)} unique genes "
        f"covering {sum(len(m.reaction_ids) for m in gene_map.values())} gene-reaction links."
    )
    return gene_map


def build_reaction_gpr_map(
    gene_map: Dict[str, GeneReactionMapping],
) -> Dict[str, str]:
    """Invert the gene map to produce reaction_id → GPR string.

    When multiple genes catalyse the same reaction the GPR is an OR:
    ``( gene_A or gene_B )``.

    Returns
    -------
    dict
        ``{reaction_id: gpr_string}``
    """
    rxn_to_genes: Dict[str, List[str]] = {}
    for mapping in gene_map.values():
        for rxn_id in mapping.reaction_ids:
            rxn_to_genes.setdefault(rxn_id, []).append(mapping.protein_id)

    gpr_map: Dict[str, str] = {}
    for rxn_id, genes in rxn_to_genes.items():
        unique_genes = sorted(set(genes))
        if len(unique_genes) == 1:
            gpr_map[rxn_id] = unique_genes[0]
        else:
            gpr_map[rxn_id] = "( " + " or ".join(unique_genes) + " )"

    logger.info(f"Built GPR map for {len(gpr_map)} reactions.")
    return gpr_map


# ---------------------------------------------------------------------------
# Feature matrix for scoring integration (F2–F7, F9, F10)
# ---------------------------------------------------------------------------
def build_scoring_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build a feature matrix from the spec columns for downstream scoring.

    Retains and cleans features F2–F7, F9, F10 (name, ec, bihit, qseqid,
    pident, evalue, qcovs, stitle) plus the derived protein_id.

    Aggregates per (rxn, protein_id): best pident, best evalue, best qcovs,
    hit count.

    Returns
    -------
    pd.DataFrame
        One row per (rxn, protein_id) pair with aggregated BLAST quality features.
    """
    required = ["rxn", "protein_id", "pident", "evalue", "qcovs"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df_valid = df.dropna(subset=["protein_id"]).copy()
    if df_valid.empty:
        logger.warning("No rows with valid protein_id for scoring features.")
        return pd.DataFrame()

    agg = (
        df_valid.groupby(["rxn", "protein_id"])
        .agg(
            name=("name", "first"),
            ec=("ec", "first"),
            best_pident=("pident", "max"),
            best_evalue=("evalue", "min"),
            best_qcovs=("qcovs", "max"),
            best_bitscore=("bitscore", "max"),
            hit_count=("qseqid", "count"),
            stitle=("stitle", "first"),
        )
        .reset_index()
    )

    logger.info(f"Scoring feature matrix: {len(agg)} (rxn, protein_id) pairs.")
    return agg


# ---------------------------------------------------------------------------
# Convenience: end-to-end pipeline
# ---------------------------------------------------------------------------
def load_gapseq_evidence(
    filepath: Union[str, Path],
    *,
    quality: str = "good_blast",
    min_pident: float = 30.0,
    max_evalue: float = 1e-5,
    min_qcovs: float = 70.0,
) -> Tuple[Dict[str, str], pd.DataFrame]:
    """One-call loader: parse → filter → return (gpr_map, scoring_features).

    Parameters
    ----------
    filepath : str or Path
        Path to the gapseq .tbl file.
    quality : str
        Status filter value (e.g. ``"good_blast"``).
    min_pident, max_evalue, min_qcovs : float
        BLAST quality thresholds.

    Returns
    -------
    tuple
        ``(gpr_map, scoring_features_df)``
    """
    df = parse_gapseq_tbl(
        filepath,
        status_filter=[quality],
        min_pident=min_pident,
        max_evalue=max_evalue,
        min_qcovs=min_qcovs,
    )
    gene_map = build_gene_reaction_map(df)
    gpr_map = build_reaction_gpr_map(gene_map)
    features = build_scoring_features(df)
    return gpr_map, features


# ---------------------------------------------------------------------------
# .tbl → standard organism DataFrame  (Strategy Pattern — same contract as
# gem_parser.build_organism_database)
# ---------------------------------------------------------------------------
def build_organism_database_from_tbl(
    tbl_df: pd.DataFrame,
    rxn_smiles_map: Dict[str, Tuple[str, str]],
    cofactor_set: Optional[Set[str]] = None,
) -> Optional[pd.DataFrame]:
    """Build the standard organism reaction DataFrame from a parsed gapseq .tbl.

    This is the Strategy Pattern twin of ``gem_parser.build_organism_database``.
    Both return an identical ``pd.DataFrame`` schema so the downstream engine
    (fingerprinting, ranking, reporting) is format-agnostic.

    Output schema
    -------------
    reaction_id | name | gpr | ec | subsystem | raw_left_smiles | raw_right_smiles

    Parameters
    ----------
    tbl_df : pd.DataFrame
        Output of :func:`parse_gapseq_tbl` (already quality-filtered).
    rxn_smiles_map : dict
        Mapping of ``rxn_id -> (left_smiles, right_smiles)``.
        Keys must match the ``rxn`` column of *tbl_df*.
        Build this from a pre-computed TSV (e.g. ``modelseed_rxn_to_smiles.tsv``)
        or from the reaction-SMILES uploader in the Streamlit UI.
    cofactor_set : set of str, optional
        SMILES strings to strip from each side of the reaction.

    Returns
    -------
    pd.DataFrame or None
        Standard organism DataFrame, or ``None`` if no reactions could be built.
    """
    if cofactor_set is None:
        cofactor_set = set()

    # Inline import to avoid a circular dependency — gem_parser is a sibling.
    from src.core.gem_parser import filter_smiles_string

    # Build the GPR map first so each unique reaction gets a consolidated GPR.
    gene_map = build_gene_reaction_map(tbl_df)
    gpr_map = build_reaction_gpr_map(gene_map)

    # One row per unique reaction (the .tbl has many rows per reaction because
    # each BLAST hit is a separate row).
    unique_rxns = tbl_df.drop_duplicates(subset=["rxn"])

    records = []
    skipped_no_smiles = 0

    for _, row in unique_rxns.iterrows():
        rxn_id = row["rxn"]
        smiles_pair = rxn_smiles_map.get(rxn_id)
        if smiles_pair is None:
            skipped_no_smiles += 1
            continue

        raw_left, raw_right = smiles_pair

        # Apply cofactor stripping (same step as gem_parser)
        clean_left = filter_smiles_string(raw_left, cofactor_set)
        clean_right = filter_smiles_string(raw_right, cofactor_set)

        # Skip reactions that become empty after cofactor removal
        if not clean_left or not clean_right:
            continue

        records.append({
            "reaction_id": rxn_id,
            "name": str(row.get("name", rxn_id)),
            "gpr": gpr_map.get(rxn_id, ""),
            "ec": str(row.get("ec", "")),
            "subsystem": str(row.get("pathway", "")),
            "raw_left_smiles": clean_left,
            "raw_right_smiles": clean_right,
        })

    if skipped_no_smiles:
        logger.info(
            f"  {skipped_no_smiles} reactions had no SMILES in lookup map and were skipped."
        )

    if not records:
        logger.warning(
            "build_organism_database_from_tbl: no reactions with SMILES could be built. "
            "Ensure rxn_smiles_map contains keys matching the .tbl 'rxn' column."
        )
        return None

    logger.info(
        f"build_organism_database_from_tbl: built {len(records)} reactions "
        f"from .tbl ({skipped_no_smiles} skipped — no SMILES)."
    )
    return pd.DataFrame(records)
