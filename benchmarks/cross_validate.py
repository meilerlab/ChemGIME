"""
Split-GEM K-Fold Cross-Validation
==================================

Partitions a GEM's reactions into K folds, training on K-1 and testing
on the held-out fold. Reports per-fold and aggregate metrics:

  - Top-k accuracy (k = 1, 5, 10)
  - Mean Reciprocal Rank (MRR)

This resolves the reviewer concern that a single 80/20 split may be
"lucky" — K independent evaluations prove consistent retrieval power.

Usage::

    python benchmarks/cross_validate.py --folds 5 --seed 42
    python benchmarks/cross_validate.py --folds 10 --gem data/iML1515.xml
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from io import BytesIO
from pathlib import Path
from typing import List

import cobra
import numpy as np
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import (
    BIGG_MET_TO_SMILES_FILE,
    COFACTOR_SMILES_FILE,
    _calculate_drfp_fingerprint,
    build_organism_database,
    calculate_fingerprints_parallel,
    filter_smiles_string,
    get_closest_rxns,
    load_bigg_to_smiles_map,
    load_cofactor_set,
)
from src.core.metrics import mean_reciprocal_rank

DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_GEM = PROJECT_ROOT / "data" / "iML1515.xml"
BIGG_MAP_PATH = PROJECT_ROOT / "data" / "database" / "bigg_to_smiles.tsv"


# ── helpers ──────────────────────────────────────────────────────────────

class FileWrapper:
    """File-like wrapper for build_organism_database."""
    def __init__(self, path: Path):
        self.name = path.name
        with open(path, "rb") as f:
            self._buf = BytesIO(f.read())

    def seek(self, pos):
        self._buf.seek(pos)

    def read(self):
        return self._buf.read()


def _normalize_gpr(gpr: str) -> set:
    clean = gpr.replace("(", " ").replace(")", " ")
    clean = clean.replace(" and ", " ").replace(" or ", " ")
    return set(clean.split())


def _gpr_match(query_gpr: str, db_gpr: str) -> bool:
    return bool(_normalize_gpr(query_gpr) & _normalize_gpr(db_gpr))


def _get_smiles_for_met(met_id: str, bigg_map: dict):
    base = met_id.rsplit("_", 1)[0] if "_" in met_id else met_id
    return bigg_map.get(base)


def _build_rxn_smiles(rxn, bigg_map):
    left = [s for m in rxn.reactants if (s := _get_smiles_for_met(m.id, bigg_map))]
    right = [s for m in rxn.products if (s := _get_smiles_for_met(m.id, bigg_map))]
    if not left or not right:
        return None, None
    return ".".join(left), ".".join(right)


def _filter_valid_reactions(model, bigg_map):
    valid = []
    for rxn in model.reactions:
        if rxn.boundary or not rxn.gene_reaction_rule.strip():
            continue
        ls, rs = _build_rxn_smiles(rxn, bigg_map)
        if ls is None:
            continue
        ec = rxn.annotation.get("ec-code", "")
        if isinstance(ec, list):
            ec = ";".join(ec)
        valid.append({
            "reaction": rxn,
            "reaction_id": rxn.id,
            "left_smiles": ls,
            "right_smiles": rs,
            "gpr": rxn.gene_reaction_rule,
            "ec": ec,
        })
    return valid


def _create_split_model(model, rxn_ids: set):
    new_model = cobra.Model(model.id + "_fold")
    new_model.name = (model.name or model.id) + " (CV fold)"
    new_model.add_reactions([r.copy() for r in model.reactions if r.id in rxn_ids])
    return new_model


# ── single-fold evaluation ───────────────────────────────────────────────

def evaluate_fold(
    train_model_path: Path,
    test_records: List[dict],
    met_to_smiles: dict,
    cofactor_set: set,
    top_k_values: tuple = (1, 5, 10),
    max_rank: int = 100,
) -> dict:
    """Run ChemGIME retrieval on one fold and return metrics."""
    gem_wrapper = FileWrapper(train_model_path)
    db_df = build_organism_database(gem_wrapper, met_to_smiles, cofactor_set)
    if db_df is None or db_df.empty:
        return {"error": "empty_database"}

    db_fps, valid_indices = calculate_fingerprints_parallel(db_df)
    if db_fps is None or len(db_fps) == 0:
        return {"error": "no_fingerprints"}

    ranks = []
    for rec in test_records:
        clean_l = filter_smiles_string(rec["left_smiles"], cofactor_set)
        clean_r = filter_smiles_string(rec["right_smiles"], cofactor_set)
        if not clean_l or not clean_r:
            ranks.append(None)
            continue

        qfp = _calculate_drfp_fingerprint((clean_l, clean_r))
        if qfp is None:
            ranks.append(None)
            continue

        ranked = get_closest_rxns(qfp.reshape(1, -1), db_fps, db_df, valid_indices)
        found = None
        for rank_idx, (_, row) in enumerate(ranked.head(max_rank).iterrows()):
            db_gpr = row.get("gpr", "")
            if db_gpr and _gpr_match(rec["gpr"], db_gpr):
                found = rank_idx + 1
                break
        ranks.append(found)

    valid_ranks = [r for r in ranks if r is not None]
    mrr = mean_reciprocal_rank(valid_ranks)

    topk_acc = {}
    for k in top_k_values:
        hits = sum(1 for r in valid_ranks if r <= k)
        topk_acc[f"top{k}"] = hits / len(valid_ranks) * 100 if valid_ranks else 0.0

    return {
        "n_test": len(test_records),
        "n_valid": len(valid_ranks),
        "n_skipped": len(ranks) - len(valid_ranks),
        "mrr": mrr,
        **topk_acc,
    }


# ── main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="K-fold cross-validation for Split-GEM")
    parser.add_argument("--gem", default=str(DEFAULT_GEM), help="Path to full GEM .xml")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default=None, help="JSON file for fold-level results")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    k = args.folds

    print("=" * 60)
    print(f"ChemGIME {k}-Fold Cross-Validation")
    print("=" * 60)

    # Load data
    bigg_map_raw = load_bigg_to_smiles_map(BIGG_MET_TO_SMILES_FILE)
    cofactor_set = load_cofactor_set(COFACTOR_SMILES_FILE)

    # Also need the bigg_map for creating split models
    bigg_map_df = pd.read_csv(BIGG_MAP_PATH, sep="\t", dtype=str, na_values=["null", "nan", ""])
    bigg_map_df = bigg_map_df.dropna(subset=["smiles"])
    bigg_map = {}
    for _, row in bigg_map_df.iterrows():
        bid = row["bigg_id"]
        if bid.startswith("biggM:M_"):
            bid = bid[8:]
        elif bid.startswith("biggM:"):
            bid = bid[6:]
        bigg_map[bid] = row["smiles"]

    print(f"\nLoading GEM: {args.gem}")
    model = cobra.io.read_sbml_model(args.gem)
    print(f"  {model.id}: {len(model.reactions)} reactions")

    valid = _filter_valid_reactions(model, bigg_map)
    random.shuffle(valid)
    print(f"  {len(valid)} valid reactions (GPR + SMILES)")

    # Create folds
    fold_size = len(valid) // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else len(valid)
        folds.append(valid[start:end])

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fold_results = []

    for fold_idx in range(k):
        print(f"\n{'─' * 40}")
        print(f"  Fold {fold_idx + 1}/{k}")
        print(f"{'─' * 40}")

        test_records = folds[fold_idx]
        train_records = [rec for i, fold in enumerate(folds) if i != fold_idx for rec in fold]

        # Build training model
        train_ids = {r["reaction_id"] for r in train_records}
        train_model = _create_split_model(model, train_ids)
        fold_model_path = DATA_DIR / f"cv_fold{fold_idx}_train.xml"
        cobra.io.write_sbml_model(train_model, str(fold_model_path))
        print(f"  Train: {len(train_records)} rxns, Test: {len(test_records)} rxns")

        metrics = evaluate_fold(
            fold_model_path, test_records, bigg_map_raw, cofactor_set
        )
        metrics["fold"] = fold_idx + 1
        fold_results.append(metrics)

        print(f"  MRR: {metrics.get('mrr', 0):.4f}")
        for kk in (1, 5, 10):
            print(f"  Top-{kk}: {metrics.get(f'top{kk}', 0):.2f}%")

        # Clean up temp model
        fold_model_path.unlink(missing_ok=True)

    # Aggregate
    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)

    mrrs = [r["mrr"] for r in fold_results if "mrr" in r]
    print(f"  MRR:   {np.mean(mrrs):.4f} +/- {np.std(mrrs):.4f}")
    for kk in (1, 5, 10):
        vals = [r[f"top{kk}"] for r in fold_results if f"top{kk}" in r]
        print(f"  Top-{kk}: {np.mean(vals):.2f}% +/- {np.std(vals):.2f}%")

    # Save results
    out_path = Path(args.output) if args.output else DATA_DIR / "cv_results.json"
    summary = {
        "folds": k,
        "seed": args.seed,
        "per_fold": fold_results,
        "aggregate": {
            "mrr_mean": float(np.mean(mrrs)),
            "mrr_std": float(np.std(mrrs)),
            **{
                f"top{kk}_mean": float(np.mean([r[f"top{kk}"] for r in fold_results if f"top{kk}" in r]))
                for kk in (1, 5, 10)
            },
            **{
                f"top{kk}_std": float(np.std([r[f"top{kk}"] for r in fold_results if f"top{kk}" in r]))
                for kk in (1, 5, 10)
            },
        },
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
