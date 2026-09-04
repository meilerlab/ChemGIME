#!/usr/bin/env python3
"""
ChemGIME Comprehensive Benchmarking Suite
==========================================

Three validation regimes in a single CLI-ready script:

  1. **Split-GEM Retrieval** — 80/20 synthetic split on iML1515 and iJN746.
     Measures GPR retrieval accuracy (Top-k, MRR).

  2. **M-CSA Novelty Benchmark** — experimentally verified M-CSA reactions
     whose enzymes (by UniProt ID) are absent from the target GEM.
     Measures zero-shot EC class prediction.

  3. **Stratified 5-Fold Cross-Validation** — EC-class-stratified folds on
     the full GEM to probe whether 2048-bit fingerprints overfit.

Usage
-----
::

    # Split-GEM on iML1515 with DRFP, cofactors filtered
    python tests/benchmarking_suite.py split-gem \\
        --gem data/iML1515.xml --fingerprint drfp --filter_cofactors true

    # M-CSA novelty for both organisms
    python tests/benchmarking_suite.py mcsa-novelty \\
        --gem data/iML1515.xml --fingerprint qrfp --filter_cofactors false

    # 5-fold CV on iJN746
    python tests/benchmarking_suite.py cross-validate \\
        --gem data/iJN746.xml --fingerprint drfp --folds 5

    # Run ALL benchmarks for a complete evaluation matrix
    python tests/benchmarking_suite.py all \\
        --gem data/iML1515.xml --fingerprint drfp --filter_cofactors true
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Reasoning: Project imports
# ---------------------------------------------------------------------------
# We add the project root to sys.path so that ``src.core`` is importable
# without requiring an editable install.  This mirrors the pattern used by
# every existing script in benchmarks/.
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import (                          # noqa: E402
    BIGG_MET_TO_SMILES_FILE,
    COFACTOR_SMILES_FILE,
    FingerprintType,
    _calculate_drfp_fingerprint,
    _calculate_drfp_substrate_fingerprint,
    _calculate_qrfp_fingerprint,
    build_organism_database,
    calculate_fingerprints_parallel,
    filter_smiles_string,
    get_closest_rxns,
    load_bigg_to_smiles_map,
    load_cofactor_set,
    predict_ec_weighted,
)
from src.core.metrics import (                  # noqa: E402
    mean_reciprocal_rank,
    retrieval_summary,
    top_k_accuracy,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("benchmarking_suite")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MCSA_CSV  = PROJECT_ROOT / "data" / "database" / "mcsa_curated_data.csv"
MCSA_JSON = PROJECT_ROOT / "data" / "database" / "mcsa_entries.json"

OUTPUT_DIR = PROJECT_ROOT / "tests" / "benchmark_output"

# ---------------------------------------------------------------------------
# Reasoning: Hardcoded cofactor SMILES
# ---------------------------------------------------------------------------
# Common high-energy cofactors, proton carriers, and ubiquitous small
# metabolites that dominate fingerprint bits without contributing
# reaction-specific information.  When --filter_cofactors is *true* these
# SMILES (plus any loaded from cofactors_smiles.txt) are removed from
# both substrates and products before featurisation.  When *false* they
# remain, which inflates Tanimoto similarity between unrelated reactions
# that happen to share cofactors—this is exactly what the benchmark
# measures, so the toggle is scientifically meaningful.
# ---------------------------------------------------------------------------
_HARDCODED_COFACTORS: Set[str] = {
    # --- proton ---
    "[H+]",
    # --- water ---
    "O", "[H]O[H]",
    # --- phosphate / pyrophosphate ---
    "OP(=O)([O-])[O-]",
    "OP(=O)([O-])OP(=O)([O-])[O-]",
    # --- CO2 ---
    "O=C=O",
    # --- NAD+ / NADH (canonical) ---
    (
        "NC(=O)C1=CC=C[N+](=C1)[C@@H]1O[C@H](COP(=O)([O-])OP(=O)([O-])"
        "OC[C@H]2O[C@@H](N3C=NC4=C3N=CN=C4N)[C@H](O)[C@@H]2O)"
        "[C@@H](O)[C@H]1O"
    ),
    (
        "[H]NC(=O)C1=CN([C@@H]2O[C@H](COP(=O)([O-])OP(=O)([O-])"
        "OC[C@H]3O[C@@H](N4C=NC5=C4N=CN=C5N)[C@H](O)[C@@H]3O)"
        "[C@@H](O)[C@H]2O)C=CC1"
    ),
    # --- ATP / ADP / AMP (truncated canonical) ---
    (
        "NC1=NC=NC2=C1N=CN2[C@@H]1O[C@H](COP(=O)([O-])OP(=O)([O-])"
        "OP(=O)([O-])[O-])[C@@H](O)[C@H]1O"
    ),
    (
        "NC1=NC=NC2=C1N=CN2[C@@H]1O[C@H](COP(=O)([O-])OP(=O)([O-])"
        "[O-])[C@@H](O)[C@H]1O"
    ),
    # --- CoA (coenzyme A) ---
    (
        "[H]OP(=O)([O-])OCC(C)(C)C(O)C(=O)NCCC(=O)NCCSC(=O)"
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

class FileWrapper:
    """File-like wrapper so ``build_organism_database`` can read a Path."""

    def __init__(self, path: Path):
        self.name = path.name
        with open(path, "rb") as fh:
            self._buf = BytesIO(fh.read())

    def seek(self, pos: int) -> None:
        self._buf.seek(pos)

    def read(self) -> bytes:
        return self._buf.read()


def _resolve_cofactor_set(filter_cofactors: bool) -> Set[str]:
    """Return the appropriate cofactor set based on the toggle.

    Reasoning
    ---------
    When ``filter_cofactors=True`` we load the project cofactor list **and**
    supplement it with a hardcoded set of universally accepted cofactors.
    This ensures robustness even when the on-disk list is incomplete or
    missing.  When ``False`` we return the empty set so that cofactor
    SMILES remain in the reaction strings.
    """
    if not filter_cofactors:
        return set()
    base: Set[str] = set()
    try:
        base = load_cofactor_set(COFACTOR_SMILES_FILE)
    except Exception:
        log.warning("cofactors_smiles.txt not found; using hardcoded set only.")
    return base | _HARDCODED_COFACTORS


def _resolve_fp_type(name: str) -> FingerprintType:
    return FingerprintType(name.lower())


def _fp_fn(fp_type: FingerprintType):
    """Return the single-pair fingerprint function for the chosen type."""
    if fp_type == FingerprintType.DRFP:
        return _calculate_drfp_fingerprint
    if fp_type == FingerprintType.DRFP_SUB:
        return _calculate_drfp_substrate_fingerprint
    return _calculate_qrfp_fingerprint


def _load_bigg_map() -> Dict[str, str]:
    return load_bigg_to_smiles_map(BIGG_MET_TO_SMILES_FILE)


def _normalize_gpr(gpr: str) -> Set[str]:
    clean = gpr.replace("(", " ").replace(")", " ")
    clean = clean.replace(" and ", " ").replace(" or ", " ")
    return set(clean.split())


def _gpr_match(query_gpr: str, db_gpr: str) -> bool:
    return bool(_normalize_gpr(query_gpr) & _normalize_gpr(db_gpr))


def _gpr_match_strict(query_gpr: str, db_gpr: str) -> bool:
    # Exact gene-set equality: candidate must carry the identical GPR gene set
    # (same enzyme/complex), not merely share one subunit gene with the query.
    q = _normalize_gpr(query_gpr)
    return bool(q) and q == _normalize_gpr(db_gpr)


def _ec_match(ec1: str, ec2: str, level: int = 3) -> bool:
    """Check whether two EC strings agree at the given hierarchy level."""
    for e1 in str(ec1).split(";"):
        for e2 in str(ec2).split(";"):
            p1 = e1.strip().split(".")[:level]
            p2 = e2.strip().split(".")[:level]
            if (
                len(p1) >= level
                and len(p2) >= level
                and all(x.isdigit() for x in p1)
                and all(x.isdigit() for x in p2)
                and p1 == p2
            ):
                return True
    return False


def _save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    log.info("Results saved → %s", path)


def _render_reaction_png(left_smiles: str, right_smiles: str, out_path: Path) -> bool:
    """Render a reaction as a .png image using RDKit.

    Reasoning
    ---------
    Embedding 2-D depictions of sample M-CSA reactions in the report lets
    reviewers visually inspect the chemical transformations that the
    pipeline was evaluated on.  RDKit's ``ReactionToImage`` produces
    publication-quality PNGs without any external service dependency.
    """
    try:
        from rdkit.Chem import AllChem, Draw
        rxn_smarts = f"{left_smiles}>>{right_smiles}"
        rxn = AllChem.ReactionFromSmarts(rxn_smarts, useSmiles=True)
        if rxn is None:
            return False
        img = Draw.ReactionToImage(rxn, subImgSize=(350, 200))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(out_path))
        return True
    except Exception as exc:
        log.debug("PNG render failed: %s", exc)
        return False


# ═══════════════════════════════════════════════════════════════════════════
#  COBRA-level helpers (import cobra lazily to keep startup fast)
# ═══════════════════════════════════════════════════════════════════════════

def _load_cobra_model(gem_path: Path):
    import cobra
    return cobra.io.read_sbml_model(str(gem_path))


def _filter_valid_reactions(model, bigg_map: dict) -> List[dict]:
    """Return reactions with GPR + resolvable SMILES on both sides."""
    valid = []
    for rxn in model.reactions:
        if rxn.boundary or not rxn.gene_reaction_rule.strip():
            continue
        left, right = [], []
        for met, stoich in rxn.metabolites.items():
            base = met.id.rsplit("_", 1)[0] if "_" in met.id else met.id
            smi = bigg_map.get(base)
            if smi:
                (left if stoich < 0 else right).append(smi)
        if not left or not right:
            continue
        ec = rxn.annotation.get("ec-code", "")
        if isinstance(ec, list):
            ec = ";".join(ec)
        valid.append({
            "reaction_id": rxn.id,
            "left_smiles": ".".join(left),
            "right_smiles": ".".join(right),
            "gpr": rxn.gene_reaction_rule,
            "ec": ec,
        })
    return valid


def _create_split_model(model, rxn_ids: set):
    import cobra
    new = cobra.Model(model.id + "_fold")
    new.name = (model.name or model.id) + " (CV fold)"
    new.add_reactions([r.copy() for r in model.reactions if r.id in rxn_ids])
    return new


EXTERNAL_UNIPROT_MAPS = {
    "iJN746": PROJECT_ROOT / "data" / "database" / "iJN746_gene_to_uniprot.json",
}


def _extract_uniprot_ids_from_gem(model) -> Set[str]:
    """Extract all UniProt IDs annotated on genes in the GEM.

    Reasoning
    ---------
    COBRA models store gene annotations in ``gene.annotation``.  The key
    ``uniprot`` (sometimes a list) holds the protein identifiers we need
    to cross-reference against M-CSA entries.  Some GEMs (e.g. iJN746)
    lack UniProt annotations but have NCBI Gene IDs; for these we load
    a pre-computed external mapping file.
    """
    ids: Set[str] = set()
    for gene in model.genes:
        ann = getattr(gene, "annotation", {}) or {}
        for key in ("uniprot", "UniProt", "UNIPROT"):
            val = ann.get(key, [])
            if isinstance(val, str):
                ids.add(val)
            elif isinstance(val, list):
                ids.update(val)
    return ids


def _get_gene_to_uniprot_map(model) -> Dict[str, str]:
    """Build gene_id → UniProt mapping, using external files if needed.

    Reasoning
    ---------
    iML1515 has native gene.annotation['uniprot'] fields.  iJN746 does
    not, so we fall back to a pre-computed mapping file generated by
    matching ordered locus names (PP_XXXX) against the UniProt reviewed
    proteome for P. putida KT2440 (organism_id:160488).
    """
    gene_to_uniprot: Dict[str, str] = {}

    # First try native COBRA annotations
    for gene in model.genes:
        ann = getattr(gene, "annotation", {}) or {}
        val = ann.get("uniprot", [])
        if isinstance(val, str):
            gene_to_uniprot[gene.id] = val
        elif isinstance(val, list) and val:
            gene_to_uniprot[gene.id] = val[0]

    if gene_to_uniprot:
        return gene_to_uniprot

    # Fall back to external mapping files
    model_stem = model.id.split("_")[0] if "_" in model.id else model.id
    for name, path in EXTERNAL_UNIPROT_MAPS.items():
        if name.lower() in model_stem.lower() and path.exists():
            log.info("Loading external UniProt mapping from %s", path)
            with open(path, "r", encoding="utf-8") as fh:
                gene_to_uniprot = json.load(fh)
            return gene_to_uniprot

    return gene_to_uniprot


# ═══════════════════════════════════════════════════════════════════════════
#  Benchmark 1 — Split-GEM Retrieval
# ═══════════════════════════════════════════════════════════════════════════

def run_split_gem(
    gem_path: Path,
    fp_type: FingerprintType,
    cofactor_set: Set[str],
    seed: int = 42,
    train_ratio: float = 0.80,
) -> dict:
    """80/20 split: train = search space, test = GPR retrieval queries.

    Reasoning
    ---------
    The Split-GEM methodology is the gold standard for evaluating
    reaction-retrieval pipelines.  By withholding 20 % of annotated
    reactions and measuring whether the pipeline can recover the correct
    Gene-Protein-Reaction (GPR) rule, we obtain Top-k accuracy and MRR
    metrics that directly quantify discovery power.

    We intentionally split at the *reaction* level (not gene level)
    because ChemGIME searches by chemical similarity, not sequence
    similarity.  A gene present in the training set does NOT guarantee
    the held-out reaction is trivially retrievable—the reaction SMILES
    are what get compared.
    """
    log.info("═" * 60)
    log.info("BENCHMARK 1: Split-GEM Retrieval  (%s, cofactors=%s)",
             fp_type.value.upper(), "filtered" if cofactor_set else "kept")
    log.info("═" * 60)

    t0 = time.time()
    bigg_map = _load_bigg_map()
    model = _load_cobra_model(gem_path)
    log.info("Loaded %s: %d reactions", model.id, len(model.reactions))

    valid = _filter_valid_reactions(model, bigg_map)
    random.seed(seed)
    random.shuffle(valid)
    split = int(len(valid) * train_ratio)
    train_records, test_records = valid[:split], valid[split:]
    log.info("Split: %d train / %d test reactions", len(train_records), len(test_records))

    # Build training GEM and organism DB
    import cobra
    train_ids = {r["reaction_id"] for r in train_records}
    train_model = _create_split_model(model, train_ids)
    tmp_path = OUTPUT_DIR / f"_tmp_{model.id}_train.xml"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    cobra.io.write_sbml_model(train_model, str(tmp_path))

    gem_wrapper = FileWrapper(tmp_path)
    met_map = _load_bigg_map()
    db_df = build_organism_database(gem_wrapper, met_map, cofactor_set)
    tmp_path.unlink(missing_ok=True)

    if db_df is None or db_df.empty:
        return {"error": "empty_database"}

    db_fps, valid_idx = calculate_fingerprints_parallel(db_df, fp_type=fp_type)
    if db_fps is None:
        return {"error": "no_fingerprints"}
    log.info("DB fingerprints: %d / %d valid", len(valid_idx), len(db_df))

    # Query each test reaction. We score every query under two ground-truth
    # criteria in a single pass: permissive (any shared GPR gene) and strict
    # (identical GPR gene set). The permissive-to-strict gap measures the
    # Top-k inflation introduced by the shared-gene rule (reviewer §4.4).
    calc_fp = _fp_fn(fp_type)
    ranks: List[Optional[int]] = []
    strict_ranks: List[Optional[int]] = []
    shared_gene_counts: List[int] = []
    exact_gpr_counts: List[int] = []
    details: List[dict] = []

    # Pre-normalise the searchable database GPRs once.
    db_gpr_sets = [_normalize_gpr(g) for g in db_df.get("gpr", "").fillna("").astype(str)]

    for rec in test_records:
        q_genes = _normalize_gpr(rec.get("gpr", ""))
        # Inflation factor: how many search-space reactions the criteria admit.
        n_shared = sum(1 for s in db_gpr_sets if q_genes & s)
        n_exact = sum(1 for s in db_gpr_sets if q_genes and q_genes == s)
        shared_gene_counts.append(n_shared)
        exact_gpr_counts.append(n_exact)

        cl = filter_smiles_string(rec["left_smiles"], cofactor_set)
        cr = filter_smiles_string(rec["right_smiles"], cofactor_set)
        if not cl or not cr:
            ranks.append(None)
            strict_ranks.append(None)
            details.append({"id": rec["reaction_id"], "rank": None,
                            "strict_rank": None, "reason": "empty_after_filter"})
            continue
        qfp = calc_fp((cl, cr))
        if qfp is None:
            ranks.append(None)
            strict_ranks.append(None)
            details.append({"id": rec["reaction_id"], "rank": None,
                            "strict_rank": None, "reason": "fp_failed"})
            continue

        ranked_df = get_closest_rxns(
            qfp.reshape(1, -1), db_fps, db_df, valid_idx,
            fp_type=fp_type.value,
        )
        found = strict_found = None
        for rank_i, (_, row) in enumerate(ranked_df.head(100).iterrows()):
            cand = _normalize_gpr(row.get("gpr", ""))
            if found is None and q_genes & cand:
                found = rank_i + 1
            if strict_found is None and q_genes and q_genes == cand:
                strict_found = rank_i + 1
            if found is not None and strict_found is not None:
                break
        ranks.append(found)
        strict_ranks.append(strict_found)
        details.append({"id": rec["reaction_id"], "rank": found,
                        "strict_rank": strict_found,
                        "n_shared_gene_rxns": n_shared,
                        "n_exact_gpr_rxns": n_exact, "reason": "ok"})

    summary = retrieval_summary(ranks, k_values=(1, 5, 10, 50, 100))
    strict_summary = retrieval_summary(strict_ranks, k_values=(1, 5, 10, 50, 100))
    elapsed = time.time() - t0
    summary["elapsed_s"] = round(elapsed, 1)
    summary["gem"] = gem_path.name
    summary["fp_type"] = fp_type.value
    summary["cofactors_filtered"] = bool(cofactor_set)
    summary["n_train"] = len(train_records)
    summary["n_test"] = len(test_records)
    summary["strict"] = strict_summary
    summary["median_shared_gene_reactions"] = (
        float(np.median(shared_gene_counts)) if shared_gene_counts else 0.0)
    summary["mean_shared_gene_reactions"] = (
        round(float(np.mean(shared_gene_counts)), 2) if shared_gene_counts else 0.0)
    summary["median_exact_gpr_reactions"] = (
        float(np.median(exact_gpr_counts)) if exact_gpr_counts else 0.0)

    log.info("PERMISSIVE  MRR=%.4f  Top-1=%.1f%%  Top-5=%.1f%%  Top-10=%.1f%%  (%.1fs)",
             summary["mrr"], summary["top1"], summary["top5"],
             summary["top10"], elapsed)
    log.info("STRICT      MRR=%.4f  Top-1=%.1f%%  Top-5=%.1f%%  Top-10=%.1f%%",
             strict_summary["mrr"], strict_summary["top1"],
             strict_summary["top5"], strict_summary["top10"])
    log.info("Inflation: median %.1f (mean %.1f) shared-gene reactions/query; "
             "median %.1f exact-GPR reactions/query",
             summary["median_shared_gene_reactions"],
             summary["mean_shared_gene_reactions"],
             summary["median_exact_gpr_reactions"])

    cof_tag = "filtered" if cofactor_set else "kept"
    out = OUTPUT_DIR / f"split_gem_{gem_path.stem}_{fp_type.value}_{cof_tag}.json"
    _save_json({"summary": summary, "details": details}, out)
    return summary


# ═══════════════════════════════════════════════════════════════════════════
#  Benchmark 2 — M-CSA Cross-Database Enzyme Retrieval
# ═══════════════════════════════════════════════════════════════════════════

MCSA_FULL_JSON = PROJECT_ROOT / "data" / "database" / "mcsa_entries_full.json"
MOL_CACHE_DIR  = PROJECT_ROOT / "data" / "database" / "mcsa_mol_cache"


def _fetch_all_mcsa_entries() -> List[dict]:
    """Fetch all M-CSA entries via paginated REST API, with local caching.

    Reasoning
    ---------
    The local mcsa_entries.json contains only page 1 (100 of 1003 entries).
    We paginate the full API to get all entries with their reaction compound
    data (ChEBI IDs + mol_file URLs).  Results are cached to
    data/database/mcsa_entries_full.json for offline re-runs.
    """
    if MCSA_FULL_JSON.exists():
        log.info("Loading cached M-CSA entries from %s", MCSA_FULL_JSON)
        with open(MCSA_FULL_JSON, "r", encoding="utf-8") as fh:
            return json.load(fh)

    import urllib.request
    all_entries: List[dict] = []
    url: Optional[str] = "https://www.ebi.ac.uk/thornton-srv/m-csa/api/entries/?format=json"
    page = 0
    while url:
        page += 1
        log.info("Fetching M-CSA page %d: %s", page, url)
        try:
            req = urllib.request.urlopen(url, timeout=30)
            data = json.loads(req.read().decode("utf-8"))
        except Exception as exc:
            log.warning("M-CSA API fetch failed at page %d: %s", page, exc)
            break
        all_entries.extend(data.get("results", []))
        url = data.get("next")

    if not all_entries:
        # Fallback: use local JSON (page 1 only)
        log.warning("API fetch returned 0 entries; falling back to local %s", MCSA_JSON)
        with open(MCSA_JSON, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            all_entries = data.get("results", [])
    else:
        # Cache for future runs
        MCSA_FULL_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(MCSA_FULL_JSON, "w", encoding="utf-8") as fh:
            json.dump(all_entries, fh, indent=1)
        log.info("Cached %d M-CSA entries to %s", len(all_entries), MCSA_FULL_JSON)

    return all_entries


def _chebi_mol_to_smiles(mol_url: str, chebi_id: str) -> Optional[str]:
    """Download .mol from M-CSA, convert to canonical SMILES, cache locally.

    Reasoning
    ---------
    M-CSA stores compound structures as .mol files hosted on their server.
    We download each file, parse it with RDKit MolFromMolBlock, and
    convert to canonical SMILES.  Files are cached to avoid re-downloads
    and to enable offline re-runs.
    """
    from rdkit import Chem

    MOL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = MOL_CACHE_DIR / f"{chebi_id}.mol"

    if cache_path.exists():
        mol_block = cache_path.read_text(encoding="utf-8")
    else:
        import urllib.request
        full_url = f"https://{mol_url}" if not mol_url.startswith("http") else mol_url
        try:
            req = urllib.request.urlopen(full_url, timeout=15)
            mol_block = req.read().decode("utf-8")
            cache_path.write_text(mol_block, encoding="utf-8")
        except Exception as exc:
            log.debug("Mol download failed for ChEBI %s: %s", chebi_id, exc)
            return None

    mol = Chem.MolFromMolBlock(mol_block)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def _build_mcsa_query_smiles(entry: dict) -> Tuple[Optional[str], Optional[str]]:
    """Build reaction SMILES (left, right) from M-CSA ChEBI compounds.

    Reasoning
    ---------
    Each M-CSA entry has a reaction dict with compounds typed as
    "reactant" or "product", each with a ChEBI ID and mol_file URL.
    We download each mol file, convert to SMILES, then join with "."
    to form the dot-separated multi-component SMILES that ChemGIME
    expects on each side.
    """
    rxn = entry.get("reaction", {})
    compounds = rxn.get("compounds", [])
    if not compounds:
        return None, None

    reactant_smiles: List[str] = []
    product_smiles: List[str] = []

    for comp in compounds:
        chebi_id = str(comp.get("chebi_id", ""))
        mol_url = comp.get("mol_file", "")
        comp_type = comp.get("type", "")
        if not chebi_id or not mol_url:
            continue

        smi = _chebi_mol_to_smiles(mol_url, chebi_id)
        if smi is None:
            continue

        if comp_type == "reactant":
            reactant_smiles.append(smi)
        elif comp_type == "product":
            product_smiles.append(smi)

    left = ".".join(reactant_smiles) if reactant_smiles else None
    right = ".".join(product_smiles) if product_smiles else None
    return left, right


def _build_rxn_to_uniprot_map(model) -> Dict[str, Set[str]]:
    """Map GEM reaction IDs to UniProt IDs via GPR → gene annotations.

    Reasoning
    ---------
    For each GEM reaction, we traverse its GPR genes, look up the
    UniProt mapping (native annotations or external file), and collect
    all UniProt IDs.  This map is used to check whether a retrieved
    GEM hit corresponds to the target M-CSA enzyme.
    """
    gene_to_uniprot = _get_gene_to_uniprot_map(model)

    rxn_to_uniprot: Dict[str, Set[str]] = {}
    for rxn in model.reactions:
        uniprots: Set[str] = set()
        for gene in rxn.genes:
            u = gene_to_uniprot.get(gene.id)
            if u:
                uniprots.add(u)
        if uniprots:
            rxn_to_uniprot[rxn.id] = uniprots
    return rxn_to_uniprot


def run_mcsa_novelty(
    gem_path: Path,
    fp_type: FingerprintType,
    cofactor_set: Set[str],
    ec_level: int = 3,
) -> dict:
    """Cross-database enzyme retrieval using M-CSA reactions as queries.

    Reasoning
    ---------
    The corrected M-CSA benchmark tests whether ChemGIME can retrieve
    the correct enzyme (UniProt ID) from a GEM when queried with a
    reaction built from an independent data source (M-CSA / ChEBI).

    **Overlap filter**: We keep M-CSA entries whose reference_uniprot_id
    IS present in the GEM.  These enzymes catalyse reactions in both
    databases, but the SMILES come from different sources (ChEBI mol
    files vs BiGG metabolite map), making the comparison non-trivial.

    **Query source**: Instead of using a GEM reaction as a proxy, we
    build the query reaction from M-CSA's own ChEBI compound structures
    (.mol → SMILES via RDKit).

    **Ground truth**: UniProt ID retrieval — we scan the ranked GEM hits
    for the first reaction whose GPR genes map to the target UniProt.
    """
    log.info("=" * 60)
    log.info("BENCHMARK 2: M-CSA Cross-Database Retrieval  (%s, cofactors=%s)",
             fp_type.value.upper(), "filtered" if cofactor_set else "kept")
    log.info("=" * 60)

    t0 = time.time()

    # --- 1. Load M-CSA entries (all pages, cached) ---
    mcsa_entries = _fetch_all_mcsa_entries()
    log.info("M-CSA entries loaded: %d", len(mcsa_entries))

    # --- 2. Load GEM and extract UniProt IDs ---
    model = _load_cobra_model(gem_path)
    gene_to_uniprot = _get_gene_to_uniprot_map(model)
    gem_uniprots = set(gene_to_uniprot.values())
    log.info("UniProt IDs in GEM: %d (via %s)",
             len(gem_uniprots),
             "native annotations" if any(
                 (getattr(g, "annotation", {}) or {}).get("uniprot")
                 for g in model.genes
             ) else "external mapping")

    # --- 3. Overlap filter: keep entries whose UniProt IS in GEM ---
    overlap_entries = []
    for entry in mcsa_entries:
        up = entry.get("reference_uniprot_id", "")
        ecs = entry.get("all_ecs", [])
        rxn = entry.get("reaction", {})
        if up and up in gem_uniprots and rxn.get("compounds"):
            # Need at least one complete 4-level EC
            valid_ec = None
            for ec in ecs:
                if ec and ec.count(".") >= 3:
                    valid_ec = ec
                    break
            if valid_ec:
                overlap_entries.append({
                    "mcsa_id": entry.get("mcsa_id"),
                    "enzyme_name": entry.get("enzyme_name", ""),
                    "uniprot_id": up,
                    "ec": valid_ec,
                    "reaction": rxn,
                })
    log.info("Overlap M-CSA entries (UniProt in %s): %d",
             gem_path.stem, len(overlap_entries))

    if not overlap_entries:
        return {"error": "no_overlap_entries",
                "gem": gem_path.name, "fp_type": fp_type.value}

    # --- 4. Build GEM database ---
    met_map = _load_bigg_map()
    gem_wrapper = FileWrapper(gem_path)
    db_df = build_organism_database(gem_wrapper, met_map, cofactor_set)
    if db_df is None or db_df.empty:
        return {"error": "empty_database"}

    db_fps, valid_idx = calculate_fingerprints_parallel(db_df, fp_type=fp_type)
    if db_fps is None:
        return {"error": "no_fingerprints"}
    log.info("GEM DB: %d reactions, %d fingerprinted", len(db_df), len(valid_idx))

    # --- 5. Build reaction → UniProt map ---
    rxn_to_uniprot = _build_rxn_to_uniprot_map(model)

    # --- 6. Query & Evaluate ---
    calc_fp = _fp_fn(fp_type)
    results: List[dict] = []
    img_dir = OUTPUT_DIR / "mcsa_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    for entry in overlap_entries:
        mcsa_id = entry["mcsa_id"]
        target_uniprot = entry["uniprot_id"]
        target_ec = entry["ec"]

        # Build query SMILES from M-CSA ChEBI compounds
        left_smi, right_smi = _build_mcsa_query_smiles(entry)
        if not left_smi or not right_smi:
            results.append({
                "mcsa_id": mcsa_id, "target_uniprot": target_uniprot,
                "target_ec": target_ec, "status": "mol_download_failed",
                "rank": None, "reciprocal_rank": 0.0,
            })
            continue

        # Apply cofactor filtering
        left_filtered = filter_smiles_string(left_smi, cofactor_set)
        right_filtered = filter_smiles_string(right_smi, cofactor_set)
        if not left_filtered or not right_filtered:
            results.append({
                "mcsa_id": mcsa_id, "target_uniprot": target_uniprot,
                "target_ec": target_ec, "status": "all_cofactors",
                "rank": None, "reciprocal_rank": 0.0,
            })
            continue

        # Compute fingerprint
        qfp = calc_fp((left_filtered, right_filtered))
        if qfp is None:
            results.append({
                "mcsa_id": mcsa_id, "target_uniprot": target_uniprot,
                "target_ec": target_ec, "status": "fp_failed",
                "rank": None, "reciprocal_rank": 0.0,
            })
            continue

        # Rank GEM reactions by distance
        ranked_df = get_closest_rxns(
            qfp.reshape(1, -1), db_fps, db_df, valid_idx,
            fp_type=fp_type.value,
        )

        # Scan ranked list for first hit with correct UniProt
        found_rank = None
        hit_rxn_id = None
        hit_distance = None
        hit_ec = None
        for rank, (_, row) in enumerate(ranked_df.iterrows(), start=1):
            rxn_id = row.get("reaction_id", "")
            hit_uniprots = rxn_to_uniprot.get(rxn_id, set())
            if target_uniprot in hit_uniprots:
                found_rank = rank
                hit_rxn_id = rxn_id
                hit_distance = float(row.get("tan_distance", 1.0))
                hit_ec = str(row.get("ec", ""))
                break

        # Top-hit EC for secondary metric
        top_ec = str(ranked_df.iloc[0].get("ec", "")) if len(ranked_df) > 0 else ""
        top_dist = float(ranked_df.iloc[0].get("tan_distance", 1.0)) if len(ranked_df) > 0 else None

        results.append({
            "mcsa_id": mcsa_id,
            "target_uniprot": target_uniprot,
            "target_ec": target_ec,
            "enzyme_name": entry.get("enzyme_name", ""),
            "status": "success",
            "found": found_rank is not None,
            "rank": found_rank,
            "reciprocal_rank": 1.0 / found_rank if found_rank else 0.0,
            "hit_rxn_id": hit_rxn_id,
            "hit_distance": hit_distance,
            "hit_ec": hit_ec,
            "top_hit_ec": top_ec,
            "top_hit_distance": top_dist,
            "ec_match_l1": _ec_match(target_ec, top_ec, level=1),
            "ec_match_l2": _ec_match(target_ec, top_ec, level=2),
            "ec_match_l3": _ec_match(target_ec, top_ec, level=3),
        })

        # Render first 10 successful reactions as PNGs
        n_imgs = sum(1 for r in results if r["status"] == "success" and
                     (img_dir / f"mcsa_{r['mcsa_id']}.png").exists())
        if n_imgs < 10:
            _render_reaction_png(left_filtered, right_filtered,
                                 img_dir / f"mcsa_{mcsa_id}.png")

    # --- 7. Aggregate metrics ---
    valid_results = [r for r in results if r["status"] == "success"]
    found_results = [r for r in valid_results if r["found"]]
    n_total = len(results)
    n_valid = len(valid_results)
    n_found = len(found_results)

    rr_values = [r["reciprocal_rank"] for r in valid_results]
    mrr = float(np.mean(rr_values)) if rr_values else 0.0
    top1 = sum(1 for r in valid_results if r["rank"] == 1) / n_valid * 100 if n_valid else 0
    top5 = sum(1 for r in valid_results if r["rank"] is not None and r["rank"] <= 5) / n_valid * 100 if n_valid else 0
    top10 = sum(1 for r in valid_results if r["rank"] is not None and r["rank"] <= 10) / n_valid * 100 if n_valid else 0

    ec_l1_acc = sum(r["ec_match_l1"] for r in valid_results) / n_valid * 100 if n_valid else 0
    ec_l2_acc = sum(r["ec_match_l2"] for r in valid_results) / n_valid * 100 if n_valid else 0
    ec_l3_acc = sum(r["ec_match_l3"] for r in valid_results) / n_valid * 100 if n_valid else 0

    hit_dists = [r["hit_distance"] for r in found_results if r["hit_distance"] is not None]
    elapsed = time.time() - t0

    summary = {
        "gem": gem_path.name,
        "fp_type": fp_type.value,
        "cofactors_filtered": bool(cofactor_set),
        "n_mcsa_total": len(mcsa_entries),
        "n_overlap": len(overlap_entries),
        "n_evaluated": n_total,
        "n_valid": n_valid,
        "n_found": n_found,
        "mrr": round(mrr, 4),
        "top1_accuracy": round(top1, 2),
        "top5_accuracy": round(top5, 2),
        "top10_accuracy": round(top10, 2),
        "ec_l1_accuracy": round(ec_l1_acc, 2),
        "ec_l2_accuracy": round(ec_l2_acc, 2),
        "ec_l3_accuracy": round(ec_l3_acc, 2),
        "mean_hit_distance": round(float(np.mean(hit_dists)), 4) if hit_dists else None,
        "median_hit_distance": round(float(np.median(hit_dists)), 4) if hit_dists else None,
        "elapsed_s": round(elapsed, 1),
    }

    log.info("MRR=%.4f  Top-1=%.1f%%  Top-5=%.1f%%  Top-10=%.1f%%  (n=%d, %.1fs)",
             mrr, top1, top5, top10, n_valid, elapsed)
    log.info("EC L1=%.1f%%  L2=%.1f%%  L3=%.1f%%",
             ec_l1_acc, ec_l2_acc, ec_l3_acc)

    cof_tag = "filtered" if cofactor_set else "kept"
    out = OUTPUT_DIR / f"mcsa_novelty_{gem_path.stem}_{fp_type.value}_{cof_tag}.json"
    _save_json({"summary": summary, "details": results}, out)
    return summary


# ═══════════════════════════════════════════════════════════════════════════
#  Benchmark 2b — M-CSA Leave-One-Enzyme-Out (strict generalisation)
# ═══════════════════════════════════════════════════════════════════════════

def run_mcsa_loo(
    gem_path: Path,
    fp_type: FingerprintType,
    cofactor_set: Set[str],
    ec_level: int = 3,
) -> dict:
    """Leave-one-enzyme-out M-CSA retrieval (strict held-out variant).

    Reasoning
    ---------
    The standard M-CSA benchmark (``run_mcsa_novelty``) selects entries whose
    UniProt ID is present in the GEM, so the enzyme's own reaction is itself a
    candidate in the search space.  High recall there partly reflects
    *re-finding* a reaction whose chemistry is encoded twice — once via BiGG
    (the GEM) and once via ChEBI (the M-CSA query) — rather than generalisation
    to unseen chemistry.

    This variant removes that confound.  For each query we hold out **all** GEM
    reactions whose GPR maps to the target UniProt (the entire enzyme, not just
    one reaction), then re-rank the remaining candidates.  Success is redefined
    on a chemistry axis: the rank of the first retained reaction that matches
    the target EC at sub-subclass level (L3) — i.e. a *chemically related but
    distinct* reaction catalysed by a different enzyme.

    To isolate the inflation attributable to the held-out twin, we report the
    same EC-L3 rank metric on (a) the full database (twin present) and (b) the
    leave-one-out database (enzyme removed).  The gap between the two MRRs is a
    direct, single-axis estimate of the "encoded-twice" advantage.
    """
    log.info("=" * 60)
    log.info("BENCHMARK 2b: M-CSA Leave-One-Enzyme-Out  (%s, cofactors=%s)",
             fp_type.value.upper(), "filtered" if cofactor_set else "kept")
    log.info("=" * 60)

    t0 = time.time()

    mcsa_entries = _fetch_all_mcsa_entries()
    model = _load_cobra_model(gem_path)
    gene_to_uniprot = _get_gene_to_uniprot_map(model)
    gem_uniprots = set(gene_to_uniprot.values())

    overlap_entries = []
    for entry in mcsa_entries:
        up = entry.get("reference_uniprot_id", "")
        ecs = entry.get("all_ecs", [])
        rxn = entry.get("reaction", {})
        if up and up in gem_uniprots and rxn.get("compounds"):
            valid_ec = next((ec for ec in ecs if ec and ec.count(".") >= 3), None)
            if valid_ec:
                overlap_entries.append({
                    "mcsa_id": entry.get("mcsa_id"),
                    "enzyme_name": entry.get("enzyme_name", ""),
                    "uniprot_id": up, "ec": valid_ec, "reaction": rxn,
                })
    if not overlap_entries:
        return {"error": "no_overlap_entries",
                "gem": gem_path.name, "fp_type": fp_type.value}

    met_map = _load_bigg_map()
    gem_wrapper = FileWrapper(gem_path)
    db_df = build_organism_database(gem_wrapper, met_map, cofactor_set)
    if db_df is None or db_df.empty:
        return {"error": "empty_database"}
    db_fps, valid_idx = calculate_fingerprints_parallel(db_df, fp_type=fp_type)
    if db_fps is None:
        return {"error": "no_fingerprints"}

    rxn_to_uniprot = _build_rxn_to_uniprot_map(model)
    uniprot_to_rxns: Dict[str, Set[str]] = defaultdict(set)
    for rid, ups in rxn_to_uniprot.items():
        for u in ups:
            uniprot_to_rxns[u].add(rid)

    calc_fp = _fp_fn(fp_type)
    results: List[dict] = []

    for entry in overlap_entries:
        mcsa_id = entry["mcsa_id"]
        target_uniprot = entry["uniprot_id"]
        target_ec = entry["ec"]

        left_smi, right_smi = _build_mcsa_query_smiles(entry)
        if not left_smi or not right_smi:
            results.append({"mcsa_id": mcsa_id, "target_uniprot": target_uniprot,
                            "target_ec": target_ec, "status": "mol_download_failed"})
            continue
        left_f = filter_smiles_string(left_smi, cofactor_set)
        right_f = filter_smiles_string(right_smi, cofactor_set)
        if not left_f or not right_f:
            results.append({"mcsa_id": mcsa_id, "target_uniprot": target_uniprot,
                            "target_ec": target_ec, "status": "all_cofactors"})
            continue
        qfp = calc_fp((left_f, right_f))
        if qfp is None:
            results.append({"mcsa_id": mcsa_id, "target_uniprot": target_uniprot,
                            "target_ec": target_ec, "status": "fp_failed"})
            continue

        ranked_df = get_closest_rxns(
            qfp.reshape(1, -1), db_fps, db_df, valid_idx, fp_type=fp_type.value,
        )

        own_rxns = uniprot_to_rxns.get(target_uniprot, set())

        # Single pass: first EC-L3 hit on full DB and on LOO DB (enzyme removed).
        uniprot_rank_full = None
        ecl3_rank_full = None
        ecl3_rank_loo = None
        loo_pos = 0
        for full_pos, (_, row) in enumerate(ranked_df.iterrows(), start=1):
            rid = row.get("reaction_id", "")
            held_out = rid in own_rxns
            if not held_out:
                loo_pos += 1
            if uniprot_rank_full is None and target_uniprot in rxn_to_uniprot.get(rid, set()):
                uniprot_rank_full = full_pos
            if _ec_match(target_ec, str(row.get("ec", "")), level=ec_level):
                if ecl3_rank_full is None:
                    ecl3_rank_full = full_pos
                if not held_out and ecl3_rank_loo is None:
                    ecl3_rank_loo = loo_pos
            if (uniprot_rank_full is not None and ecl3_rank_full is not None
                    and ecl3_rank_loo is not None):
                break

        results.append({
            "mcsa_id": mcsa_id,
            "target_uniprot": target_uniprot,
            "target_ec": target_ec,
            "enzyme_name": entry.get("enzyme_name", ""),
            "status": "success",
            "n_held_out": len(own_rxns),
            "has_ecl3_sibling": ecl3_rank_loo is not None,
            "uniprot_rank_full": uniprot_rank_full,
            "ecl3_rank_full": ecl3_rank_full,
            "ecl3_rank_loo": ecl3_rank_loo,
            "rr_full": 1.0 / ecl3_rank_full if ecl3_rank_full else 0.0,
            "rr_loo": 1.0 / ecl3_rank_loo if ecl3_rank_loo else 0.0,
        })

    valid = [r for r in results if r["status"] == "success"]
    n_valid = len(valid)
    with_sib = [r for r in valid if r["has_ecl3_sibling"]]

    def _mrr(rs, key):
        return float(np.mean([r[key] for r in rs])) if rs else 0.0

    def _topk(rs, key, k):
        hit = sum(1 for r in rs if r[key] is not None and r[key] <= k)
        return hit / len(rs) * 100 if rs else 0.0

    n_held = [r["n_held_out"] for r in valid]
    elapsed = time.time() - t0

    summary = {
        "gem": gem_path.name,
        "fp_type": fp_type.value,
        "cofactors_filtered": bool(cofactor_set),
        "ec_level": ec_level,
        "n_evaluated": len(results),
        "n_valid": n_valid,
        "n_with_ecl3_sibling": len(with_sib),
        "median_enzyme_reactions_held_out": float(np.median(n_held)) if n_held else 0.0,
        # EC-L3 ground truth, full database (held-out twin present)
        "mrr_ecl3_full": round(_mrr(valid, "rr_full"), 4),
        "top1_ecl3_full": round(_topk(valid, "ecl3_rank_full", 1), 2),
        "top10_ecl3_full": round(_topk(valid, "ecl3_rank_full", 10), 2),
        # EC-L3 ground truth, leave-one-enzyme-out
        "mrr_ecl3_loo": round(_mrr(valid, "rr_loo"), 4),
        "top1_ecl3_loo": round(_topk(valid, "ecl3_rank_loo", 1), 2),
        "top10_ecl3_loo": round(_topk(valid, "ecl3_rank_loo", 10), 2),
        # Conditional on a related reaction existing (n_with_ecl3_sibling)
        "mrr_ecl3_loo_conditional": round(_mrr(with_sib, "rr_loo"), 4),
        "top10_ecl3_loo_conditional": round(_topk(with_sib, "ecl3_rank_loo", 10), 2),
        "elapsed_s": round(elapsed, 1),
    }

    log.info("EC-L3 MRR: full=%.4f  LOO=%.4f  (drop=%.4f)  | LOO Top-10=%.1f%%  "
             "n_sibling=%d/%d  (%.1fs)",
             summary["mrr_ecl3_full"], summary["mrr_ecl3_loo"],
             summary["mrr_ecl3_full"] - summary["mrr_ecl3_loo"],
             summary["top10_ecl3_loo"], summary["n_with_ecl3_sibling"],
             n_valid, elapsed)

    cof_tag = "filtered" if cofactor_set else "kept"
    out = OUTPUT_DIR / f"mcsa_loo_{gem_path.stem}_{fp_type.value}_{cof_tag}.json"
    _save_json({"summary": summary, "details": results}, out)
    return summary


# ═══════════════════════════════════════════════════════════════════════════
#  Benchmark 3 — Stratified 5-Fold Cross-Validation
# ═══════════════════════════════════════════════════════════════════════════

def run_cross_validate(
    gem_path: Path,
    fp_type: FingerprintType,
    cofactor_set: Set[str],
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    """EC-stratified K-fold CV to detect 2048-bit fingerprint overfitting.

    Reasoning
    ---------
    The manuscript peer-review audit (Felipe feedback) flags that
    "2048-bit DRFP vectors on relatively small and highly curated
    datasets substantially increase the risk of overfitting."  A
    single 80/20 split can be "lucky".

    **Stratified** splitting is critical: we group reactions by their
    EC L1 class (oxidoreductase, transferase, …) and distribute each
    class proportionally across all folds.  This prevents a fold from
    lacking an entire enzyme class, which would artificially inflate
    error and mask genuine overfitting.

    If MRR variance across folds is high (std > 0.05), the model is
    likely memorising fold-specific patterns rather than learning
    generalisable chemical similarity—a direct overfitting signal.
    """
    log.info("═" * 60)
    log.info("BENCHMARK 3: %d-Fold Stratified CV  (%s, cofactors=%s)",
             n_folds, fp_type.value.upper(), "filtered" if cofactor_set else "kept")
    log.info("═" * 60)

    t0 = time.time()
    bigg_map = _load_bigg_map()
    model = _load_cobra_model(gem_path)
    valid = _filter_valid_reactions(model, bigg_map)
    log.info("Valid reactions: %d", len(valid))

    # --- EC-stratified fold assignment ---
    random.seed(seed)
    np.random.seed(seed)

    # Group by EC L1 class
    ec_buckets: Dict[str, List[dict]] = defaultdict(list)
    for rec in valid:
        ec_str = str(rec.get("ec", ""))
        parts = ec_str.split(";")[0].strip().split(".")
        ec_l1 = parts[0] if parts[0].isdigit() else "none"
        ec_buckets[ec_l1].append(rec)

    # Shuffle within each bucket and deal round-robin into folds
    folds: List[List[dict]] = [[] for _ in range(n_folds)]
    for _ec_class, bucket in ec_buckets.items():
        random.shuffle(bucket)
        for i, rec in enumerate(bucket):
            folds[i % n_folds].append(rec)

    log.info("Fold sizes: %s", [len(f) for f in folds])

    # --- Evaluate each fold ---
    import cobra

    fold_results: List[dict] = []
    calc_fp = _fp_fn(fp_type)

    for fold_i in range(n_folds):
        log.info("── Fold %d/%d ──", fold_i + 1, n_folds)
        test_recs = folds[fold_i]
        train_recs = [r for j, f in enumerate(folds) if j != fold_i for r in f]

        # Build training model
        train_ids = {r["reaction_id"] for r in train_recs}
        train_model = _create_split_model(model, train_ids)
        fold_path = OUTPUT_DIR / f"_tmp_cv_fold{fold_i}.xml"
        fold_path.parent.mkdir(parents=True, exist_ok=True)
        cobra.io.write_sbml_model(train_model, str(fold_path))

        gem_w = FileWrapper(fold_path)
        db_df = build_organism_database(gem_w, bigg_map, cofactor_set)
        fold_path.unlink(missing_ok=True)

        if db_df is None or db_df.empty:
            fold_results.append({"fold": fold_i + 1, "error": "empty_db"})
            continue

        db_fps, v_idx = calculate_fingerprints_parallel(db_df, fp_type=fp_type)
        if db_fps is None:
            fold_results.append({"fold": fold_i + 1, "error": "no_fps"})
            continue

        ranks: List[Optional[int]] = []
        for rec in test_recs:
            cl = filter_smiles_string(rec["left_smiles"], cofactor_set)
            cr = filter_smiles_string(rec["right_smiles"], cofactor_set)
            if not cl or not cr:
                ranks.append(None)
                continue
            qfp = calc_fp((cl, cr))
            if qfp is None:
                ranks.append(None)
                continue
            ranked = get_closest_rxns(
                qfp.reshape(1, -1), db_fps, db_df, v_idx,
                fp_type=fp_type.value,
            )
            found = None
            for ri, (_, row) in enumerate(ranked.head(100).iterrows()):
                db_gpr = row.get("gpr", "")
                if db_gpr and _gpr_match(rec["gpr"], db_gpr):
                    found = ri + 1
                    break
            ranks.append(found)

        s = retrieval_summary(ranks, k_values=(1, 5, 10))
        s["fold"] = fold_i + 1
        s["n_train"] = len(train_recs)
        s["n_test"] = len(test_recs)
        fold_results.append(s)
        log.info("  MRR=%.4f  Top-1=%.1f%%  Top-5=%.1f%%  Top-10=%.1f%%",
                 s["mrr"], s["top1"], s["top5"], s["top10"])

    # Aggregate
    mrrs = [r["mrr"] for r in fold_results if "mrr" in r]
    elapsed = time.time() - t0

    aggregate = {
        "gem": gem_path.name,
        "fp_type": fp_type.value,
        "cofactors_filtered": bool(cofactor_set),
        "n_folds": n_folds,
        "seed": seed,
        "mrr_mean": round(float(np.mean(mrrs)), 4) if mrrs else 0,
        "mrr_std": round(float(np.std(mrrs)), 4) if mrrs else 0,
        "elapsed_s": round(elapsed, 1),
    }
    for k in (1, 5, 10):
        vals = [r.get(f"top{k}", 0) for r in fold_results if f"top{k}" in r]
        aggregate[f"top{k}_mean"] = round(float(np.mean(vals)), 2) if vals else 0
        aggregate[f"top{k}_std"] = round(float(np.std(vals)), 2) if vals else 0

    # Overfitting diagnostic
    if aggregate["mrr_std"] > 0.05:
        log.warning(
            "HIGH MRR VARIANCE (std=%.4f) — possible overfitting of %s "
            "fingerprints on %s.  Consider reducing dimensionality or "
            "adding regularisation.",
            aggregate["mrr_std"], fp_type.value.upper(), gem_path.stem,
        )
    else:
        log.info("MRR variance within tolerance (std=%.4f)", aggregate["mrr_std"])

    log.info("AGGREGATE  MRR=%.4f±%.4f  Top-1=%.1f±%.1f%%  (%.1fs)",
             aggregate["mrr_mean"], aggregate["mrr_std"],
             aggregate["top1_mean"], aggregate["top1_std"], elapsed)

    cof_tag = "filtered" if cofactor_set else "kept"
    out = OUTPUT_DIR / f"cv_{gem_path.stem}_{fp_type.value}_{n_folds}fold_{cof_tag}.json"
    _save_json({"aggregate": aggregate, "per_fold": fold_results}, out)
    return aggregate


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="benchmarking_suite",
        description="ChemGIME comprehensive benchmarking suite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # --- shared arguments ---
    def add_common(sp):
        sp.add_argument("--gem", required=True, type=Path,
                        help="Path to GEM .xml file (e.g. data/iML1515.xml)")
        sp.add_argument("--fingerprint", choices=["drfp", "qrfp", "drfp_sub"],
                        default="drfp",
                        help="Fingerprint type: drfp (binary), qrfp (quaternary), "
                             "or drfp_sub (substrate-only DRFP, MicrobeRX ablation)")
        sp.add_argument("--filter_cofactors", choices=["true", "false"], default="true",
                        help="Filter common cofactors from SMILES before featurisation")
        sp.add_argument("--seed", type=int, default=42, help="Random seed")

    # split-gem
    sp1 = sub.add_parser("split-gem", help="80/20 Split-GEM GPR retrieval benchmark")
    add_common(sp1)
    sp1.add_argument("--train-ratio", type=float, default=0.80)

    # mcsa-novelty
    sp2 = sub.add_parser("mcsa-novelty", help="M-CSA zero-shot EC prediction")
    add_common(sp2)
    sp2.add_argument("--ec-level", type=int, default=3, choices=[1, 2, 3, 4])

    # mcsa-loo
    sp2b = sub.add_parser("mcsa-loo",
                          help="M-CSA leave-one-enzyme-out (strict generalisation)")
    add_common(sp2b)
    sp2b.add_argument("--ec-level", type=int, default=3, choices=[1, 2, 3, 4])

    # cross-validate
    sp3 = sub.add_parser("cross-validate", help="EC-stratified K-fold cross-validation")
    add_common(sp3)
    sp3.add_argument("--folds", type=int, default=5)

    # all
    sp4 = sub.add_parser("all", help="Run all three benchmarks sequentially")
    add_common(sp4)
    sp4.add_argument("--folds", type=int, default=5)

    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    fp_type = _resolve_fp_type(args.fingerprint)
    cofactor_set = _resolve_cofactor_set(args.filter_cofactors == "true")
    gem = args.gem

    if not gem.exists():
        log.error("GEM file not found: %s", gem)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results: Dict[str, dict] = {}

    if args.command in ("split-gem", "all"):
        ratio = getattr(args, "train_ratio", 0.80)
        all_results["split_gem"] = run_split_gem(
            gem, fp_type, cofactor_set, seed=args.seed, train_ratio=ratio,
        )

    if args.command in ("mcsa-novelty", "all"):
        ec_lev = getattr(args, "ec_level", 3)
        all_results["mcsa_novelty"] = run_mcsa_novelty(
            gem, fp_type, cofactor_set, ec_level=ec_lev,
        )

    if args.command == "mcsa-loo":
        ec_lev = getattr(args, "ec_level", 3)
        all_results["mcsa_loo"] = run_mcsa_loo(
            gem, fp_type, cofactor_set, ec_level=ec_lev,
        )

    if args.command in ("cross-validate", "all"):
        n_folds = getattr(args, "folds", 5)
        all_results["cross_validate"] = run_cross_validate(
            gem, fp_type, cofactor_set, n_folds=n_folds, seed=args.seed,
        )

    # Print final summary
    print("\n" + "=" * 70)
    print("  BENCHMARKING SUITE — FINAL SUMMARY")
    print(f"  GEM: {gem.name}  |  FP: {fp_type.value.upper()}  |  "
          f"Cofactors: {'filtered' if cofactor_set else 'kept'}")
    print("=" * 70)
    for name, result in all_results.items():
        if "error" in result:
            print(f"  {name}: ERROR — {result['error']}")
            continue
        print(f"\n  [{name}]")
        for k, v in result.items():
            if k not in ("gem", "fp_type", "cofactors_filtered"):
                print(f"    {k}: {v}")
    print("=" * 70)


if __name__ == "__main__":
    main()
