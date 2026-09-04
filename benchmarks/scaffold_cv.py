"""
ChemGIME Bemis-Murcko Scaffold-Split Cross-Validation
======================================================

Reviewer robustness check for Major Weakness 4 (peer-review audit):
"The 5-fold cross-validation uses EC level-1 stratification only. With
2,048-bit DRFPs on a small GEM, structurally near-duplicate reactions
can co-occur in train and test folds, allowing nearest-neighbour
retrieval to succeed via structural memorisation rather than genuine
generalisation."

This script repeats the cross-validation with a **scaffold-aware split**:
all reactions whose dominant substrate shares the same Bemis-Murcko
scaffold are assigned to the same fold.  If ChemGIME's MRR under this
harder split remains within ~10 percentage points of the EC-stratified
numbers in Table 2, the retrieval is not exploiting scaffold memorisation.

Scaffold definition
-------------------
For each reaction the "dominant substrate" is the largest (by heavy-atom
count) non-cofactor molecule on the left-hand side.  Its Bemis-Murcko
scaffold (RDKit ``MurckoDecompose``) is the fold-assignment key.  For
acyclic substrates (no ring) the canonical SMILES itself is used as the
key, preserving molecule identity while avoiding empty-scaffold collisions.

Fold assignment
---------------
Scaffold clusters are sorted by size (largest first) and greedily assigned
to the smallest fold (bin-packing), keeping all same-scaffold reactions in
the same fold.  The resulting folds are as balanced as possible given the
cluster structure.

MRR evaluation
--------------
Identical to the EC-stratified CV in ``tests/benchmarking_suite.py``:
* A test reaction is encoded as a fingerprint.
* It is ranked against the training GEM (4 other folds).
* The correct answers are training reactions whose GPR rule shares at
  least one gene with the test reaction (GPR-overlap ground truth).
* Rank is found within the top 100; if absent, reciprocal rank = 0.

Usage
-----
    python benchmarks/scaffold_cv.py
    python benchmarks/scaffold_cv.py --fp-type qrfp --cofactors keep
    python benchmarks/scaffold_cv.py --gem iJN746.xml --fp-type drfp
    python benchmarks/scaffold_cv.py --all    # runs 2x2 matrix, both FPs x cofactor settings
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

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
from src.core.metrics import mean_reciprocal_rank, retrieval_summary, top_k_accuracy

log = logging.getLogger("scaffold_cv")
DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_GEM = PROJECT_ROOT / "iML1515.xml"


# ════════════════════════════════════════════════════════════════════════
#  File wrapper (mirrors benchmarking_suite.py)
# ════════════════════════════════════════════════════════════════════════

class FileWrapper:
    def __init__(self, path: Path):
        self.name = path.name
        with open(path, "rb") as fh:
            self._buf = BytesIO(fh.read())

    def seek(self, pos: int) -> None:
        self._buf.seek(pos)

    def read(self) -> bytes:
        return self._buf.read()


# ════════════════════════════════════════════════════════════════════════
#  GPR helpers (mirrors benchmarking_suite.py)
# ════════════════════════════════════════════════════════════════════════

def _normalize_gpr(gpr: str) -> Set[str]:
    clean = gpr.replace("(", " ").replace(")", " ")
    clean = clean.replace(" and ", " ").replace(" or ", " ")
    return {g for g in clean.split() if g}


def _gpr_overlap(q: str, db: str) -> bool:
    return bool(_normalize_gpr(q) & _normalize_gpr(db))


# ════════════════════════════════════════════════════════════════════════
#  GEM loading
# ════════════════════════════════════════════════════════════════════════

def _load_valid_reactions(gem_path: Path, bigg_map: Dict[str, str]) -> List[dict]:
    """Load GEM reactions that have GPR annotations and resolvable SMILES."""
    import cobra
    model = cobra.io.read_sbml_model(str(gem_path))
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
    return model, valid


def _create_fold_model(model, rxn_ids: set):
    import cobra
    new = cobra.Model(model.id + "_fold")
    new.name = (model.name or model.id) + " (scaffold-CV fold)"
    new.add_reactions([r.copy() for r in model.reactions if r.id in rxn_ids])
    return new


# ════════════════════════════════════════════════════════════════════════
#  Bemis-Murcko scaffold computation
# ════════════════════════════════════════════════════════════════════════

def _dominant_substrate(left_smiles: str, cofactor_set: Set[str]) -> Optional[str]:
    """Return the SMILES of the largest non-cofactor substrate."""
    from rdkit import Chem
    candidates = [s for s in left_smiles.split(".") if s and s not in cofactor_set]
    if not candidates:
        candidates = [s for s in left_smiles.split(".") if s]
    if not candidates:
        return None
    best, best_n = None, -1
    for smi in candidates:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        n = mol.GetNumHeavyAtoms()
        if n > best_n:
            best_n, best = n, smi
    return best


def _murcko_key(smiles: str) -> str:
    """Bemis-Murcko scaffold SMILES; falls back to canonical SMILES for acyclics."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    if not scaffold:
        return Chem.MolToSmiles(mol, canonical=True)
    return scaffold


def assign_scaffold_keys(
    valid: List[dict],
    cofactor_set: Set[str],
) -> List[str]:
    """Compute a scaffold key for every valid reaction.

    Returns a parallel list of scaffold key strings.
    """
    keys = []
    for rec in valid:
        dom = _dominant_substrate(rec["left_smiles"], cofactor_set)
        if dom is None:
            keys.append(f"__no_substrate_{rec['reaction_id']}")
        else:
            keys.append(_murcko_key(dom))
    return keys


# ════════════════════════════════════════════════════════════════════════
#  Scaffold-cluster → fold assignment (greedy bin-packing)
# ════════════════════════════════════════════════════════════════════════

def scaffold_folds(
    valid: List[dict],
    scaffold_keys: List[str],
    n_folds: int = 5,
) -> Tuple[List[List[dict]], Dict[str, int]]:
    """Assign reactions to folds so that all reactions with the same scaffold
    key land in the same fold.

    Uses greedy bin-packing (largest-cluster first) to maximise fold balance.
    Returns (folds, scaffold_to_fold_map).
    """
    # Build scaffold → [reaction indices] map
    scaffold_to_indices: Dict[str, List[int]] = defaultdict(list)
    for i, key in enumerate(scaffold_keys):
        scaffold_to_indices[key].append(i)

    # Sort clusters by size descending
    clusters = sorted(scaffold_to_indices.items(), key=lambda x: len(x[1]), reverse=True)

    # Greedy bin-packing: assign each cluster to the currently smallest fold
    fold_sizes = [0] * n_folds
    folds: List[List[dict]] = [[] for _ in range(n_folds)]
    scaffold_to_fold: Dict[str, int] = {}

    for scaffold_key, indices in clusters:
        smallest_fold = int(np.argmin(fold_sizes))
        for i in indices:
            folds[smallest_fold].append(valid[i])
        fold_sizes[smallest_fold] += len(indices)
        scaffold_to_fold[scaffold_key] = smallest_fold

    return folds, scaffold_to_fold


# ════════════════════════════════════════════════════════════════════════
#  Single-condition scaffold CV
# ════════════════════════════════════════════════════════════════════════

def run_scaffold_cv(
    gem_path: Path,
    fp_type: FingerprintType,
    cofactor_set: Set[str],
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    """Run one scaffold-split CV condition.

    Returns a result dict matching the structure produced by the
    EC-stratified CV in tests/benchmarking_suite.py.
    """
    t0 = time.time()
    bigg_map = load_bigg_to_smiles_map(BIGG_MET_TO_SMILES_FILE)

    log.info("Loading GEM: %s", gem_path.name)
    model, valid = _load_valid_reactions(gem_path, bigg_map)
    log.info("Valid reactions with GPR + SMILES: %d", len(valid))

    # Compute scaffold keys
    log.info("Computing Bemis-Murcko scaffold keys ...")
    np.random.seed(seed)
    scaffold_keys = assign_scaffold_keys(valid, cofactor_set)
    n_unique = len(set(scaffold_keys))
    log.info("Unique scaffold keys: %d  (reactions: %d)", n_unique, len(valid))

    # Assign folds
    folds, scaffold_to_fold = scaffold_folds(valid, scaffold_keys, n_folds)
    fold_sizes = [len(f) for f in folds]
    log.info("Fold sizes (scaffold-split): %s", fold_sizes)

    # Count scaffold clusters per fold
    from collections import Counter
    fold_scaffold_counts = Counter(scaffold_to_fold.values())
    log.info("Scaffold clusters per fold: %s",
             [fold_scaffold_counts.get(i, 0) for i in range(n_folds)])

    calc_fp = _calculate_qrfp_fingerprint if fp_type == FingerprintType.QRFP \
              else _calculate_drfp_fingerprint

    import cobra
    fold_results = []

    for fold_i in range(n_folds):
        log.info("── Scaffold-CV fold %d/%d ──", fold_i + 1, n_folds)
        test_recs = folds[fold_i]
        train_recs = [r for j, f in enumerate(folds) if j != fold_i for r in f]

        train_ids = {r["reaction_id"] for r in train_recs}
        train_model = _create_fold_model(model, train_ids)
        fold_path = DATA_DIR / f"_tmp_scaffold_cv_fold{fold_i}.xml"
        fold_path.parent.mkdir(parents=True, exist_ok=True)
        cobra.io.write_sbml_model(train_model, str(fold_path))

        gem_w = FileWrapper(fold_path)
        db_df = build_organism_database(gem_w, bigg_map, cofactor_set)
        fold_path.unlink(missing_ok=True)

        if db_df is None or db_df.empty:
            fold_results.append({"fold": fold_i + 1, "error": "empty_db"})
            continue

        db_fps, v_idx = calculate_fingerprints_parallel(db_df, fp_type=fp_type)
        if db_fps is None or len(db_fps) == 0:
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
                if _gpr_overlap(rec["gpr"], str(row.get("gpr", ""))):
                    found = ri + 1
                    break
            ranks.append(found)

        s = retrieval_summary(ranks, k_values=(1, 5, 10))
        s["fold"] = fold_i + 1
        s["n_train"] = len(train_recs)
        s["n_test"] = len(test_recs)
        s["n_scaffold_clusters_in_fold"] = fold_scaffold_counts.get(fold_i, 0)
        fold_results.append(s)

    # Aggregate
    valid_folds = [f for f in fold_results if "error" not in f]
    mrrs = [f["mrr"] for f in valid_folds]
    top1s = [f["top1"] for f in valid_folds]
    top10s = [f["top10"] for f in valid_folds]

    elapsed = time.time() - t0
    result = {
        "benchmark": "Scaffold-split cross-validation (Bemis-Murcko)",
        "config": {
            "gem": gem_path.name,
            "fp_type": fp_type.value,
            "cofactors": "filtered" if cofactor_set else "kept",
            "n_folds": n_folds,
            "seed": seed,
        },
        "split_stats": {
            "n_reactions": len(valid),
            "n_unique_scaffolds": n_unique,
            "fold_sizes": fold_sizes,
            "fold_scaffold_counts": [fold_scaffold_counts.get(i, 0) for i in range(n_folds)],
        },
        "aggregate": {
            "mrr_mean": float(np.mean(mrrs)) if mrrs else None,
            "mrr_std": float(np.std(mrrs)) if mrrs else None,
            "top1_mean": float(np.mean(top1s)) if top1s else None,
            "top1_std": float(np.std(top1s)) if top1s else None,
            "top10_mean": float(np.mean(top10s)) if top10s else None,
            "top10_std": float(np.std(top10s)) if top10s else None,
        },
        "fold_results": fold_results,
        "elapsed_s": round(elapsed, 1),
    }
    return result


# ════════════════════════════════════════════════════════════════════════
#  Reporting
# ════════════════════════════════════════════════════════════════════════

def print_summary(result: dict) -> None:
    cfg = result["config"]
    agg = result["aggregate"]
    ss = result["split_stats"]
    print("\n" + "=" * 68)
    print("SCAFFOLD-SPLIT CV — SUMMARY")
    print("=" * 68)
    print(f"  GEM: {cfg['gem']}   FP: {cfg['fp_type'].upper()}   "
          f"cofactors: {cfg['cofactors']}")
    print(f"  Reactions: {ss['n_reactions']}   "
          f"Unique scaffolds: {ss['n_unique_scaffolds']}   "
          f"Folds: {cfg['n_folds']}")
    print(f"  Fold sizes: {ss['fold_sizes']}")
    print()
    print(f"  MRR:   {agg['mrr_mean']:.3f} ± {agg['mrr_std']:.3f}")
    print(f"  Top-1: {agg['top1_mean']:.1f} ± {agg['top1_std']:.1f} %")
    print(f"  Top-10:{agg['top10_mean']:.1f} ± {agg['top10_std']:.1f} %")
    print()
    print("  Per-fold results:")
    print(f"  {'Fold':>4} {'n_test':>6} {'n_scaffolds':>11} "
          f"{'MRR':>6} {'Top-1':>6} {'Top-10':>7}")
    for f in result["fold_results"]:
        if "error" in f:
            print(f"  {f['fold']:>4}  ERROR: {f['error']}")
            continue
        print(f"  {f['fold']:>4} {f['n_test']:>6} "
              f"{f.get('n_scaffold_clusters_in_fold', '?'):>11} "
              f"{f['mrr']:>6.3f} {f['top1']:>6.1f} {f['top10']:>7.1f}")
    print("=" * 68)


# ════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(
        description="Scaffold-split (Bemis-Murcko) CV robustness check for ChemGIME")
    p.add_argument("--gem", default=str(DEFAULT_GEM), help="Path to GEM .xml")
    p.add_argument("--fp-type", choices=["drfp", "qrfp"], default="drfp")
    p.add_argument("--cofactors", choices=["keep", "filter"], default="keep")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None, help="Output JSON path (auto-named if omitted)")
    p.add_argument(
        "--all", action="store_true",
        help="Run the full 2×2 matrix (DRFP/QRFP × keep/filter) on the given GEM"
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    gem_path = Path(args.gem)
    if not gem_path.exists():
        print(f"ERROR: GEM not found: {gem_path}", file=sys.stderr)
        sys.exit(1)

    if args.all:
        conditions = [
            ("drfp", "keep"),
            ("drfp", "filter"),
            ("qrfp", "keep"),
            ("qrfp", "filter"),
        ]
    else:
        conditions = [(args.fp_type, args.cofactors)]

    all_results = []
    for fp_str, cof_str in conditions:
        fp_type = FingerprintType(fp_str)
        if cof_str == "filter":
            try:
                cofactor_set = load_cofactor_set(COFACTOR_SMILES_FILE)
            except Exception:
                cofactor_set = set()
        else:
            cofactor_set = set()

        print(f"\nRunning scaffold CV: {gem_path.name}  FP={fp_str.upper()}  "
              f"cofactors={cof_str}")
        result = run_scaffold_cv(gem_path, fp_type, cofactor_set, args.folds, args.seed)
        print_summary(result)

        tag = f"{gem_path.stem}_{fp_str}_{cof_str}"
        out_path = Path(args.output) if args.output else DATA_DIR / f"scaffold_cv_{tag}.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, default=str)
        print(f"\nResults saved to: {out_path}")
        all_results.append(result)

    if len(all_results) > 1:
        print("\n" + "=" * 68)
        print("SCAFFOLD-SPLIT CV — MATRIX SUMMARY")
        print("=" * 68)
        print(f"{'GEM':<12} {'FP':<6} {'Cofactors':<9} "
              f"{'MRR mean±std':>16} {'Top-1 mean±std':>16} {'Top-10 mean±std':>17}")
        for r in all_results:
            c = r["config"]
            a = r["aggregate"]
            mrr_s = f"{a['mrr_mean']:.3f}±{a['mrr_std']:.3f}"
            t1_s = f"{a['top1_mean']:.1f}±{a['top1_std']:.1f}"
            t10_s = f"{a['top10_mean']:.1f}±{a['top10_std']:.1f}"
            print(f"{c['gem']:<12} {c['fp_type'].upper():<6} {c['cofactors']:<9} "
                  f"{mrr_s:>16} {t1_s:>16} {t10_s:>17}")
        print("=" * 68)


if __name__ == "__main__":
    main()
