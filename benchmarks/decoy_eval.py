"""
ChemGIME Specificity / Decoy Benchmark
======================================

Recall metrics (MRR, Top-k) show ChemGIME *finds* the true enzyme; they do
not show it *rejects* chemically similar but enzymatically wrong reactions.
This script supplies that missing specificity evidence and answers the
reviewer question: "for every true hit you retrieve, how many false positives
do you also retrieve?"

Design (Strategy 2 — cross-EC GEM decoys, see ``decoy_plan.md``)
---------------------------------------------------------------
For each M-CSA query (the *same* 130-query set as the headline benchmark):

  1. Build the query reaction SMILES from M-CSA ChEBI mol files.
  2. Rank every GEM reaction by Tanimoto/Jaccard distance (the normal pipeline).
  3. Label each GEM reaction:
       active  — same EC class as the query (default L3)
       decoy   — structurally similar (similarity >= threshold) but a
                 *different* EC class  (the hard negatives)
       easy/no-EC — everything else (excluded from the hard ROC test)
  4. Score discrimination: ROC-AUC, BEDROC, enrichment factor, precision@k,
     and the FPR/TPR at ChemGIME's operational threshold.

Each query is its own retrieval task, so the headline number is the
**mean per-query ROC-AUC** (with a bootstrap CI). A pooled ROC curve
(within-query standardised similarities) is drawn for visualisation.

Usage
-----
    python benchmarks/decoy_eval.py                       # iML1515, DRFP, cofactors kept
    python benchmarks/decoy_eval.py --fp-type qrfp
    python benchmarks/decoy_eval.py --active-def ec --active-level 4
    python benchmarks/decoy_eval.py --scramble            # scrambled-reaction decoys
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import (
    BIGG_MET_TO_SMILES_FILE,
    COFACTOR_SMILES_FILE,
    FingerprintType,
    _calculate_drfp_fingerprint,
    _calculate_qrfp_fingerprint,
    build_organism_database,
    calculate_fingerprints_parallel,
    filter_smiles_string,
    get_closest_rxns,
    load_bigg_to_smiles_map,
    load_cofactor_set,
)
from src.core.specificity import (bootstrap_ci, discrimination_summary,
                                  recall_at_k, roc_points)

from benchmarks.decoy_generation import (
    _any_ec_matches,
    label_candidates,
    make_scrambled_reactions,
)

log = logging.getLogger("decoy_eval")

DATA_DIR = SCRIPT_DIR / "data"
DB_DIR = PROJECT_ROOT / "data" / "database"
MCSA_FULL_JSON = DB_DIR / "mcsa_entries_full.json"
MOL_CACHE_DIR = DB_DIR / "mcsa_mol_cache"
DEFAULT_GEM = PROJECT_ROOT / "iML1515.xml"
DEFAULT_QUERY_SET = (PROJECT_ROOT / "tests" / "benchmark_output" /
                     "mcsa_novelty_iML1515_drfp_kept.json")
GENE_TO_UNIPROT = DB_DIR / "iML1515_gene_to_uniprot.json"


# ════════════════════════════════════════════════════════════════════════
#  GEM loading
# ════════════════════════════════════════════════════════════════════════

class FileWrapper:
    """Minimal file-like wrapper accepted by build_organism_database."""
    def __init__(self, path: Path):
        self.name = path.name
        with open(path, "rb") as f:
            self._buf = BytesIO(f.read())

    def seek(self, pos):
        self._buf.seek(pos)

    def read(self):
        return self._buf.read()


def build_gem(gem_path: Path, cofactor_set: set, fp_type: FingerprintType):
    """Build the GEM reaction DataFrame and its fingerprint matrix."""
    met_to_smiles = load_bigg_to_smiles_map(BIGG_MET_TO_SMILES_FILE)
    db_df = build_organism_database(FileWrapper(gem_path), met_to_smiles, cofactor_set)
    if db_df is None or db_df.empty:
        raise RuntimeError(f"Failed to build GEM database from {gem_path}")
    db_fps, valid_indices = calculate_fingerprints_parallel(db_df, fp_type=fp_type)
    if db_fps is None or len(db_fps) == 0:
        raise RuntimeError("No fingerprints generated for GEM")
    return db_df, db_fps, valid_indices


# ════════════════════════════════════════════════════════════════════════
#  Active-target identification (precise per-query ground truth)
# ════════════════════════════════════════════════════════════════════════

def _normalize_gpr(gpr: str) -> set:
    """Genes mentioned in a gene-reaction rule, stripped of boolean syntax."""
    clean = (str(gpr).replace("(", " ").replace(")", " ")
             .replace(" and ", " ").replace(" or ", " "))
    return {g for g in clean.split() if g}


def build_reaction_gene_map(db_df: pd.DataFrame) -> dict:
    """``reaction_id -> set(genes)`` for GPR-based active identification."""
    return {
        row["reaction_id"]: _normalize_gpr(row.get("gpr", ""))
        for _, row in db_df.iterrows()
    }


def active_reaction_ids(
    query: dict,
    db_df: pd.DataFrame,
    rxn_gene_map: dict,
    uniprot_to_genes: dict,
    *,
    active_def: str,
    active_level: int,
) -> set:
    """Precise set of GEM reaction IDs that are the query enzyme's true target.

    * ``gpr`` — reactions whose GPR shares a gene with the query enzyme
      (target UniProt -> genes). This is "the reaction this enzyme catalyses",
      matching the manuscript's GPR-based ground truth.
    * ``ec``  — reactions whose EC equals the query EC at ``active_level``
      (default 4 = the exact enzymatic reaction).
    """
    if active_def == "gpr":
        genes = set(uniprot_to_genes.get(query.get("target_uniprot", ""), []))
        if not genes:
            return set()
        return {rid for rid, gset in rxn_gene_map.items() if gset & genes}

    # EC-based
    tec = query["target_ec"]
    return {
        row["reaction_id"]
        for _, row in db_df.iterrows()
        if _any_ec_matches(tec, str(row.get("ec", "")), active_level)
    }


# ════════════════════════════════════════════════════════════════════════
#  M-CSA query construction (ChEBI mol -> reaction SMILES)
# ════════════════════════════════════════════════════════════════════════

def _chebi_mol_to_smiles(mol_url: str, chebi_id: str) -> Optional[str]:
    """Convert an M-CSA ChEBI .mol file to canonical SMILES (cached locally)."""
    from rdkit import Chem

    MOL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = MOL_CACHE_DIR / f"{chebi_id}.mol"
    if cache_path.exists():
        mol_block = cache_path.read_text(encoding="utf-8")
    else:
        full_url = mol_url if mol_url.startswith("http") else f"https://{mol_url}"
        try:
            req = urllib.request.urlopen(full_url, timeout=15)
            mol_block = req.read().decode("utf-8")
            cache_path.write_text(mol_block, encoding="utf-8")
        except Exception as exc:
            log.debug("mol download failed for ChEBI %s: %s", chebi_id, exc)
            return None
    mol = Chem.MolFromMolBlock(mol_block)
    return Chem.MolToSmiles(mol) if mol is not None else None


def build_query_smiles(entry: dict) -> tuple[Optional[str], Optional[str]]:
    """Build (left_smiles, right_smiles) from an M-CSA entry's compounds."""
    compounds = entry.get("reaction", {}).get("compounds", [])
    reactants, products = [], []
    for comp in compounds:
        chebi_id = str(comp.get("chebi_id", ""))
        mol_url = comp.get("mol_file", "")
        if not chebi_id or not mol_url:
            continue
        smi = _chebi_mol_to_smiles(mol_url, chebi_id)
        if smi is None:
            continue
        if comp.get("type") == "reactant":
            reactants.append(smi)
        elif comp.get("type") == "product":
            products.append(smi)
    left = ".".join(reactants) if reactants else None
    right = ".".join(products) if products else None
    return left, right


def load_query_set(query_set_path: Path, mcsa_full: list[dict]) -> list[dict]:
    """Resolve the canonical query set: the exact M-CSA entries used by the
    headline benchmark, joined to their full reaction (compound) records.

    Falls back to deriving the overlap directly from the GEM's UniProt set if
    the headline-benchmark JSON is unavailable.
    """
    by_id = {e.get("mcsa_id"): e for e in mcsa_full}

    if query_set_path.exists():
        details = json.load(open(query_set_path))["details"]
        queries = []
        for d in details:
            full = by_id.get(d["mcsa_id"])
            if full is None:
                continue
            queries.append({
                "mcsa_id": d["mcsa_id"],
                "enzyme_name": d.get("enzyme_name", ""),
                "target_ec": d["target_ec"],
                "target_uniprot": d.get("target_uniprot", ""),
                "reaction": full.get("reaction", {}),
            })
        log.info("Loaded %d queries from headline benchmark %s",
                 len(queries), query_set_path.name)
        return queries

    # Fallback: derive overlap from the GEM's UniProt set.
    log.warning("Query-set file %s missing; deriving overlap from GEM UniProts",
                query_set_path)
    g2u = json.load(open(GENE_TO_UNIPROT))
    gem_uniprots = set(g2u.values())
    queries = []
    for e in mcsa_full:
        up = e.get("reference_uniprot_id", "")
        ecs = [ec for ec in e.get("all_ecs", []) if ec and ec.count(".") >= 3]
        if up in gem_uniprots and ecs and e.get("reaction", {}).get("compounds"):
            queries.append({
                "mcsa_id": e.get("mcsa_id"),
                "enzyme_name": e.get("enzyme_name", ""),
                "target_ec": ecs[0],
                "target_uniprot": up,
                "reaction": e.get("reaction", {}),
            })
    return queries


# ════════════════════════════════════════════════════════════════════════
#  Cross-EC decoy evaluation
# ════════════════════════════════════════════════════════════════════════

def run_cross_ec_eval(args) -> dict:
    fp_type = FingerprintType(args.fp_type)
    filter_cofactors = (args.cofactors == "filter")
    cofactor_set = load_cofactor_set(COFACTOR_SMILES_FILE) if filter_cofactors else set()
    query_filter_set = cofactor_set  # same set used to clean query SMILES

    qfp_fn = (_calculate_qrfp_fingerprint if fp_type == FingerprintType.QRFP
              else _calculate_drfp_fingerprint)

    print(f"\nBuilding GEM database: {args.gem}  "
          f"(fp={fp_type.value}, cofactors={'filtered' if filter_cofactors else 'kept'})")
    db_df, db_fps, valid_indices = build_gem(Path(args.gem), cofactor_set, fp_type)
    print(f"  Search space: {len(db_df)} reactions, "
          f"{len(db_fps)} fingerprinted")

    mcsa_full = json.load(open(MCSA_FULL_JSON))
    queries = load_query_set(Path(args.query_set), mcsa_full)
    print(f"  Queries: {len(queries)} M-CSA entries")
    print(f"  Active target definition: {args.active_def}"
          + (f" (EC level {args.active_level})" if args.active_def == "ec" else "")
          + "\n")

    uniprot_to_genes = json.load(open(args.uniprot_map)) if Path(args.uniprot_map).exists() else {}
    rxn_gene_map = build_reaction_gene_map(db_df)

    per_query: list[dict] = []
    hard_roc_data: list[tuple] = []  # (labels, scores) per hard-decoy query
    global_roc_data: list[tuple] = []  # (labels, scores) per query, full decoy pool

    for q in tqdm(queries, desc="  decoy eval"):
        left, right = build_query_smiles(q)
        if filter_cofactors:
            left = filter_smiles_string(left or "", query_filter_set)
            right = filter_smiles_string(right or "", query_filter_set)
        if not left or not right:
            per_query.append({**_meta(q), "status": "smiles_failed"})
            continue

        qfp = qfp_fn((left, right))
        if qfp is None:
            per_query.append({**_meta(q), "status": "fingerprint_failed"})
            continue

        actives = active_reaction_ids(
            q, db_df, rxn_gene_map, uniprot_to_genes,
            active_def=args.active_def, active_level=args.active_level,
        )
        if not actives:
            per_query.append({**_meta(q), "status": "no_active_in_gem"})
            continue

        ranked = get_closest_rxns(
            qfp.reshape(1, -1), db_fps, db_df, valid_indices, fp_type=fp_type.value
        )

        labelled = label_candidates(
            ranked, actives,
            target_ec=q["target_ec"],
            decoy_level=args.decoy_level,
            sim_threshold=args.sim_threshold,
        )

        if labelled.n_active == 0:
            # The true reaction exists in the GEM but was not fingerprinted
            # (no parseable SMILES), so it is absent from the search space.
            per_query.append({**_meta(q), "status": "active_not_fingerprinted"})
            continue
        if labelled.n_decoy_all == 0:
            # No different-class reaction anywhere: nothing to discriminate.
            per_query.append({**_meta(q), "status": "no_decoys_at_all"})
            continue

        sim_thr = 1.0 - args.tau
        g_labels, g_scores = labelled.global_subset()
        record = {
            **_meta(q), "status": "ok",
            "n_actives": labelled.n_active,
            "n_decoys_hard": labelled.n_decoy,
            "n_decoys_all": labelled.n_decoy_all,
            "n_ambiguous": labelled.n_ambiguous,
            "has_hard_decoys": labelled.n_decoy > 0,
            "global": discrimination_summary(g_labels, g_scores, sim_threshold=sim_thr),
            "hard": None,
        }

        if labelled.n_decoy > 0:
            h_labels, h_scores = labelled.hard_subset()
            record["hard"] = discrimination_summary(h_labels, h_scores, sim_threshold=sim_thr)
            hard_roc_data.append((h_labels, h_scores))

        global_roc_data.append((g_labels, g_scores))
        per_query.append(record)

    aggregate = _aggregate(per_query, hard_roc_data, args)
    result = {
        "benchmark": "M-CSA cross-EC decoy / specificity",
        "config": {
            "gem": str(args.gem),
            "fp_type": fp_type.value,
            "cofactors": "filtered" if filter_cofactors else "kept",
            "active_def": args.active_def,
            "active_ec_level": args.active_level if args.active_def == "ec" else None,
            "decoy_ec_level": args.decoy_level,
            "decoy_sim_threshold": args.sim_threshold,
            "operational_tau": args.tau,
            "query_set": Path(args.query_set).name,
        },
        "aggregate": aggregate,
        "per_query": per_query,
    }
    return result, hard_roc_data, global_roc_data


def _meta(q: dict) -> dict:
    return {
        "mcsa_id": q["mcsa_id"],
        "enzyme_name": q["enzyme_name"],
        "target_ec": q["target_ec"],
    }


def _mean_std_ci(values: list[float], seed: int) -> dict:
    arr = np.array([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean": None, "std": None, "ci95": [None, None]}
    mean, lo, hi = bootstrap_ci(arr, seed=seed)
    return {
        "n": int(arr.size),
        "mean": float(mean),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "ci95": [float(lo), float(hi)],
    }


_METRIC_KEYS = ["roc_auc", "bedroc", "average_precision", "ef_0.01", "ef_0.05",
                "precision_at_1", "precision_at_5", "precision_at_10"]


def _summarise_namespace(records, namespace, args) -> dict:
    """Aggregate the discrimination metrics held under ``record[namespace]``."""
    subs = [r[namespace] for r in records if r.get(namespace) is not None]
    out = {"n_queries": len(subs)}
    out["metrics"] = {
        key: _mean_std_ci([s.get(key) for s in subs], args.seed)
        for key in _METRIC_KEYS
    }
    tpr = [s["at_threshold"]["tpr"] for s in subs if s["at_threshold"]["tpr"] is not None]
    fpr = [s["at_threshold"]["fpr"] for s in subs if s["at_threshold"]["fpr"] is not None]
    out["operational_threshold"] = {
        "tau_distance": args.tau,
        "mean_tpr": float(np.mean(tpr)) if tpr else None,
        "mean_fpr": float(np.mean(fpr)) if fpr else None,
    }
    return out


def _aggregate(per_query, hard_roc_data, args) -> dict:
    evaluable = [r for r in per_query if r.get("status") == "ok"]
    with_hard = [r for r in evaluable if r.get("has_hard_decoys")]

    def _count(status):
        return sum(1 for r in per_query if r["status"] == status)

    agg = {
        "n_queries_total": len(per_query),
        "n_queries_evaluable": len(evaluable),
        "n_clean_window": len(evaluable) - len(with_hard),
        "n_with_hard_decoys": len(with_hard),
        "n_active_not_fingerprinted": _count("active_not_fingerprinted"),
        "n_no_active_in_gem": _count("no_active_in_gem"),
        "n_no_decoys_at_all": _count("no_decoys_at_all"),
        "n_smiles_failed": _count("smiles_failed"),
        "median_actives_per_query": (
            float(np.median([r["n_actives"] for r in evaluable])) if evaluable else None),
        "median_hard_decoys_per_query": (
            float(np.median([r["n_decoys_hard"] for r in with_hard])) if with_hard else None),
        "median_global_decoys_per_query": (
            float(np.median([r["n_decoys_all"] for r in evaluable])) if evaluable else None),
    }

    # Two discrimination views: hard (within operational window) and global
    # (against every different-class reaction in the GEM).
    agg["hard"] = _summarise_namespace(with_hard, "hard", args)
    agg["global"] = _summarise_namespace(evaluable, "global", args)
    return agg


def mean_roc_curve(roc_data, n_points: int = 200):
    """Vertical-averaged ROC across per-query retrieval tasks.

    Each query is one retrieval task; pooling raw (label, score) pairs lets
    queries with many decoys dominate, so we interpolate every query's ROC
    onto a shared FPR grid and average the TPR.  The area under the mean
    curve corresponds to the mean per-query AUC.

    The per-query AUC distribution is strongly bimodal, so a mean +/- SD band
    is not a meaningful summary (the band spans almost the whole unit square).
    We therefore also return the 25th/75th percentile envelope, which is
    distribution-free and valid under bimodality.  Returns
    ``(grid, mean_tpr, lo_tpr, hi_tpr, tprs, area)`` or ``None``.
    """
    grid = np.linspace(0, 1, n_points)
    tprs = []
    for labels, scores in roc_data:
        rp = roc_points(labels, scores)
        if rp is None:
            continue
        fpr, tpr, _ = rp
        tprs.append(np.interp(grid, fpr, tpr))
    if not tprs:
        return None
    tprs = np.vstack(tprs)
    mean_tpr = tprs.mean(axis=0)
    mean_tpr[0], mean_tpr[-1] = 0.0, 1.0
    lo_tpr = np.percentile(tprs, 25, axis=0)
    hi_tpr = np.percentile(tprs, 75, axis=0)
    return grid, mean_tpr, lo_tpr, hi_tpr, tprs, float(np.trapz(mean_tpr, grid))


# ════════════════════════════════════════════════════════════════════════
#  Scrambled-reaction decoy evaluation (reviewer.md §4.7 i)
# ════════════════════════════════════════════════════════════════════════

def _best_hit_distance(rxn_smiles: str, qfp_fn, db_fps, db_df, valid_indices,
                       fp_type) -> Optional[float]:
    """Tanimoto distance of a reaction's nearest GEM neighbour, or None."""
    left, _, right = rxn_smiles.partition(">>")
    fp = qfp_fn((left, right))
    if fp is None:
        return None
    ranked = get_closest_rxns(fp.reshape(1, -1), db_fps, db_df,
                              valid_indices, fp_type=fp_type.value)
    return float(ranked["tan_distance"].iloc[0])


def run_scramble_eval(args) -> dict:
    """Discriminate true reactions from scrambled (wrong-product) reactions.

    For each query the true reaction (L>>R) is compared against ``n_decoys``
    scrambled reactions (L>>R', R' a product set from other GEM reactions).
    The score is the similarity of each reaction to its nearest GEM neighbour;
    a reaction-aware fingerprint should match the true reaction more closely
    than its scrambled counterparts.  Reported as per-query ROC-AUC etc.
    """
    fp_type = FingerprintType(args.fp_type)
    filter_cofactors = (args.cofactors == "filter")
    cofactor_set = load_cofactor_set(COFACTOR_SMILES_FILE) if filter_cofactors else set()
    qfp_fn = (_calculate_qrfp_fingerprint if fp_type == FingerprintType.QRFP
              else _calculate_drfp_fingerprint)

    print(f"\nBuilding GEM database: {args.gem}")
    db_df, db_fps, valid_indices = build_gem(Path(args.gem), cofactor_set, fp_type)

    # Donor product pool: real product-side SMILES from the GEM's reactions.
    donor_pool = [s for s in db_df["raw_right_smiles"].astype(str).tolist() if s]

    mcsa_full = json.load(open(MCSA_FULL_JSON))
    queries = load_query_set(Path(args.query_set), mcsa_full)
    print(f"  Queries: {len(queries)};  scrambled decoys/query: {args.n_decoys}\n")

    per_query: list[dict] = []
    roc_data: list[tuple] = []
    for q in tqdm(queries, desc="  scramble eval"):
        left, right = build_query_smiles(q)
        if filter_cofactors:
            left = filter_smiles_string(left or "", cofactor_set)
            right = filter_smiles_string(right or "", cofactor_set)
        if not left or not right:
            per_query.append({**_meta(q), "status": "smiles_failed"})
            continue

        d_true = _best_hit_distance(f"{left}>>{right}", qfp_fn, db_fps, db_df,
                                    valid_indices, fp_type)
        if d_true is None:
            per_query.append({**_meta(q), "status": "fingerprint_failed"})
            continue

        decoys = make_scrambled_reactions(left, right, donor_pool,
                                          n_decoys=args.n_decoys, seed=args.seed)
        d_decoys = [d for r in decoys
                    if (d := _best_hit_distance(r, qfp_fn, db_fps, db_df,
                                                valid_indices, fp_type)) is not None]
        if not d_decoys:
            per_query.append({**_meta(q), "status": "no_decoys"})
            continue

        labels = np.array([1] + [0] * len(d_decoys))
        scores = 1.0 - np.array([d_true] + d_decoys)  # similarity to nearest GEM hit
        summ = discrimination_summary(labels, scores, sim_threshold=(1.0 - args.tau))
        per_query.append({
            **_meta(q), "status": "ok",
            "true_best_distance": d_true,
            "median_decoy_best_distance": float(np.median(d_decoys)),
            **summ,
        })
        roc_data.append((labels, scores))

    evaluable = [r for r in per_query if r["status"] == "ok"]
    metrics = {
        key: _mean_std_ci([r.get(key) for r in evaluable], args.seed)
        for key in ["roc_auc", "bedroc", "average_precision",
                    "precision_at_1", "precision_at_5"]
    }
    true_d = [r["true_best_distance"] for r in evaluable]
    dec_d = [r["median_decoy_best_distance"] for r in evaluable]
    return {
        "benchmark": "Scrambled-reaction decoy specificity",
        "config": {"gem": str(args.gem), "fp_type": fp_type.value,
                   "n_decoys_per_query": args.n_decoys, "seed": args.seed},
        "n_queries_total": len(per_query),
        "n_queries_evaluable": len(evaluable),
        "mean_true_best_distance": float(np.mean(true_d)) if true_d else None,
        "mean_decoy_best_distance": float(np.mean(dec_d)) if dec_d else None,
        "metrics": metrics,
        "per_query": per_query,
    }


# ════════════════════════════════════════════════════════════════════════
#  Figures
# ════════════════════════════════════════════════════════════════════════

def make_figures(result: dict, per_query, hard_roc_data, global_roc_data, out_prefix: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        log.warning("matplotlib unavailable (%s); skipping figures", exc)
        return []

    evaluable = [r for r in per_query if r.get("status") == "ok"]
    aucs = [r["hard"]["roc_auc"] for r in evaluable
            if r.get("hard") and r["hard"].get("roc_auc") is not None]
    hard_metrics = result["aggregate"]["hard"]["metrics"]
    written = []

    # LaTeX \scriptsize in an 11pt document = 8pt
    FSIZE = 15
    plt.rcParams.update({
        "font.size": FSIZE,
        "axes.titlesize": FSIZE,
        "axes.labelsize": FSIZE,
        "xtick.labelsize": FSIZE,
        "ytick.labelsize": FSIZE,
        "legend.fontsize": FSIZE,
    })

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (a) Vertical-averaged ROC across per-query hard-decoy tasks.
    # Panel (a) shows the FULL decoy pool, not the hard tier.  The hard-tier
    # per-query AUCs are strongly bimodal (panel b), so any mean-plus-spread
    # ROC for that subset spans nearly the whole unit square and summarises
    # nothing.  The full-pool distribution is unimodal and tight, so a mean
    # curve with an IQR envelope is meaningful there, and it is also the
    # realistic retrieval setting.
    ax = axes[0]
    mr = mean_roc_curve(global_roc_data)
    if mr is not None:
        grid, mean_tpr, lo_tpr, hi_tpr, tprs, area = mr
        ax.fill_between(grid, lo_tpr, hi_tpr, color="#4477aa", alpha=0.25,
                        label="IQR across queries")
        ax.plot(grid, mean_tpr, lw=2, color="#4477aa",
                label=f"mean ROC (AUC = {area:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("(a) Mean ROC — full decoy pool")
    ax.legend(loc="lower right", framealpha=0.9)

    # (b) Per-query hard ROC-AUC distribution.
    ax = axes[1]
    if aucs:
        # Bin edges land on 1.0 so the terminal bar counts only AUC == 1.0
        # exactly; a [0.95, 1.0] bin would silently absorb near-perfect queries
        # and disagree with the count quoted in the caption.
        ax.hist(aucs, bins=20, range=(0, 1), color="#4477aa", edgecolor="white")
        m = hard_metrics["roc_auc"]["mean"]
        med = float(np.median(aucs))
        n_perfect = int(np.sum(np.asarray(aucs) == 1.0))
        ax.axvline(m, color="#cc3311", lw=2, label=f"mean = {m:.3f}")
        ax.axvline(med, color="#117733", lw=2, ls="-",
                   label=f"median = {med:.2f} ({n_perfect}/{len(aucs)} perfect)")
        ax.axvline(0.5, color="k", ls="--", lw=1, alpha=0.6, label="random")
        ax.legend(fontsize=FSIZE - 3, framealpha=0.9)
    ax.set_xlabel("Per-query ROC-AUC (hard decoys)")
    ax.set_ylabel("Number of queries")
    ax.set_title("(b) Per-query discrimination")

    # (c) Mean recall@k on the hard decoy challenge.
    #
    # precision@k is a poor summary here: with a median of one active per
    # query it is capped near 1/k, so a declining precision@k curve is
    # arithmetically forced by the active-set size rather than being evidence
    # about discrimination.  recall@k ("is the true reaction inside the top
    # k?") carries no such ceiling and matches the Top-k language used for the
    # retrieval benchmarks elsewhere in the paper.
    ax = axes[2]
    ks = [1, 5, 10]
    rec_per_query = {k: [] for k in ks}
    for labels, scores in hard_roc_data:
        y = np.asarray(labels, dtype=int)
        if y.sum() == 0:
            continue
        for k in ks:
            rec_per_query[k].append(recall_at_k(labels, scores, k))
    vals, errs = [], []
    for k in ks:
        vv = rec_per_query[k]
        if not vv:
            vals.append(0.0); errs.append([0, 0]); continue
        m, lo, hi = bootstrap_ci(vv, seed=42)
        vals.append(float(m))
        errs.append([max(0.0, m - lo), max(0.0, hi - m)])
    errs = np.array(errs).T
    ax.bar([str(k) for k in ks], vals, yerr=errs, color="#228833",
           capsize=4, edgecolor="white")
    ax.set_ylim(0, 1)
    ax.set_xlabel("k")
    ax.set_ylabel("Recall@k (true reaction in top k)")
    ax.set_title("(c) Top-k recall")

    fig.tight_layout()
    fig_path = out_prefix.with_name(out_prefix.name + "_figure.png")
    fig.savefig(fig_path, dpi=600)
    pdf_path = fig_path.with_suffix(".pdf")
    fig.savefig(pdf_path)
    plt.rcParams.update(plt.rcParamsDefault)
    plt.close(fig)
    written.append(fig_path)
    written.append(pdf_path)
    return written


# ════════════════════════════════════════════════════════════════════════
#  Reporting
# ════════════════════════════════════════════════════════════════════════

def _fmt(m):
    if m["mean"] is None:
        return "n/a"
    return f"{m['mean']:.3f}  95% CI [{m['ci95'][0]:.3f}, {m['ci95'][1]:.3f}]  (n={m['n']})"


def print_summary(result: dict):
    agg = result["aggregate"]
    print("\n" + "=" * 68)
    print("SPECIFICITY / DECOY BENCHMARK — SUMMARY")
    print("=" * 68)
    print(f"  Queries evaluable: {agg['n_queries_evaluable']}/{agg['n_queries_total']}")
    print(f"    with confusable in-window decoys: {agg['n_with_hard_decoys']}")
    print(f"    clean window (only true target near): {agg['n_clean_window']}")
    print(f"    true reaction not fingerprinted:  {agg['n_active_not_fingerprinted']}")
    print(f"    no active in GEM:                 {agg['n_no_active_in_gem']}")
    print(f"  Median actives/query: {agg['median_actives_per_query']}; "
          f"median hard/global decoys: {agg['median_hard_decoys_per_query']}"
          f" / {agg['median_global_decoys_per_query']}")

    print("\n  [HARD] active vs in-window different-class decoys "
          f"(n={agg['hard']['n_queries']}):")
    hm = agg["hard"]["metrics"]
    print(f"    ROC-AUC:        {_fmt(hm['roc_auc'])}")
    print(f"    BEDROC (a=20):  {_fmt(hm['bedroc'])}")
    print(f"    Avg precision:  {_fmt(hm['average_precision'])}")
    print(f"    EF @1% / @5%:   {_fmt(hm['ef_0.01'])} / {_fmt(hm['ef_0.05'])}")
    print(f"    Precision@1/5/10: {_fmt(hm['precision_at_1'])} | "
          f"{_fmt(hm['precision_at_5'])} | {_fmt(hm['precision_at_10'])}")

    print("\n  [GLOBAL] active vs all different-class GEM reactions "
          f"(n={agg['global']['n_queries']}):")
    gm = agg["global"]["metrics"]
    print(f"    ROC-AUC:        {_fmt(gm['roc_auc'])}")
    print(f"    BEDROC (a=20):  {_fmt(gm['bedroc'])}")
    print(f"    EF @1% / @5%:   {_fmt(gm['ef_0.01'])} / {_fmt(gm['ef_0.05'])}")
    ot = agg["global"]["operational_threshold"]
    if ot["mean_tpr"] is not None:
        print(f"    At operational tau={ot['tau_distance']}: mean TPR "
              f"{ot['mean_tpr']:.3f} (true target retained), mean FPR "
              f"{ot['mean_fpr']:.3f} (other-class reactions admitted)")


# ════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="ChemGIME specificity / decoy benchmark")
    p.add_argument("--gem", default=str(DEFAULT_GEM), help="Path to GEM .xml")
    p.add_argument("--fp-type", choices=["drfp", "qrfp"], default="drfp")
    p.add_argument("--cofactors", choices=["keep", "filter"], default="keep")
    p.add_argument("--query-set", default=str(DEFAULT_QUERY_SET),
                   help="Headline-benchmark JSON defining the canonical query set")
    p.add_argument("--uniprot-map", default=str(DB_DIR / "iML1515_uniprot_to_genes.json"),
                   help="UniProt->genes JSON for GPR-based active identification")
    p.add_argument("--active-def", choices=["gpr", "ec"], default="gpr",
                   help="How the true target reaction(s) are identified: by "
                        "shared GPR gene with the query enzyme (default) or by "
                        "exact EC match")
    p.add_argument("--active-level", type=int, default=4, choices=[1, 2, 3, 4],
                   help="EC level for active matching when --active-def ec (default 4)")
    p.add_argument("--decoy-level", type=int, default=3, choices=[1, 2, 3, 4],
                   help="EC level at which a similar candidate must differ to be a decoy")
    p.add_argument("--sim-threshold", type=float, default=0.5,
                   help="Min similarity (1-distance) for a hard decoy")
    p.add_argument("--tau", type=float, default=0.5,
                   help="ChemGIME operational distance threshold for FPR/TPR")
    p.add_argument("--scramble", action="store_true",
                   help="Run the scrambled-reaction decoy benchmark instead "
                        "of the cross-EC GEM decoy benchmark")
    p.add_argument("--n-decoys", type=int, default=10,
                   help="Scrambled decoys per query (--scramble mode)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None, help="Output JSON path")
    args = p.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    # DRFP fingerprints are float; sklearn casts to bool for Jaccard. Harmless.
    import warnings
    from sklearn.exceptions import DataConversionWarning
    warnings.filterwarnings("ignore", category=DataConversionWarning)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("ChemGIME Specificity / Decoy Benchmark")
    print("=" * 68)

    if args.scramble:
        result = run_scramble_eval(args)
        tag = f"{Path(args.gem).stem}_{args.fp_type}_{args.cofactors}"
        out_path = Path(args.output) if args.output else DATA_DIR / f"decoy_scramble_{tag}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        m = result["metrics"]
        print(f"\n  SCRAMBLED-REACTION DECOY ({result['n_queries_evaluable']}"
              f"/{result['n_queries_total']} queries)")
        print(f"    True vs scrambled best-hit distance: "
              f"{result['mean_true_best_distance']:.3f} vs "
              f"{result['mean_decoy_best_distance']:.3f}")
        print(f"    ROC-AUC:     {_fmt(m['roc_auc'])}")
        print(f"    BEDROC:      {_fmt(m['bedroc'])}")
        print(f"    Precision@1: {_fmt(m['precision_at_1'])}")
        print(f"\nResults saved to: {out_path}")
        return

    result, hard_roc_data, global_roc_data = run_cross_ec_eval(args)

    tag = f"{Path(args.gem).stem}_{args.fp_type}_{args.cofactors}"
    out_path = Path(args.output) if args.output else DATA_DIR / f"decoy_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    figs = make_figures(result, result["per_query"], hard_roc_data, global_roc_data,
                        out_path.with_suffix(""))
    print_summary(result)
    print(f"\nResults saved to: {out_path}")
    for fp in figs:
        print(f"Figure saved to:  {fp}")


if __name__ == "__main__":
    main()
