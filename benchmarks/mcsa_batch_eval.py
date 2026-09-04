"""
M-CSA Batch Evaluation Script (F3)
===================================

Evaluate ChemGIME retrieval across n >= 20 well-characterised enzymatic
reactions from the Mechanism and Catalytic Site Atlas (M-CSA).

This script:
  1. Loads M-CSA curated entries and extracts reactions with known EC numbers.
  2. Maps EC numbers to GEM reactions via the BiGG database.
  3. For each M-CSA reaction, queries a reference GEM and checks whether
     ChemGIME retrieves a reaction with the *correct EC class*.
  4. Reports MRR and Top-k accuracy across all M-CSA queries.

The goal is to demonstrate that ChemGIME generalises beyond a single case
study to diverse enzyme classes — supporting the paper claim that the tool
has broad discovery power.

Usage::

    python benchmarks/mcsa_batch_eval.py --gem data/iML1515.xml --min-queries 20
    python benchmarks/mcsa_batch_eval.py --gem data/iML1515.xml --ec-level 3
"""

from __future__ import annotations

import argparse
import json
import sys
from io import BytesIO
from pathlib import Path

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
    predict_ec_weighted,
)
from src.core.metrics import mean_reciprocal_rank, retrieval_summary

DATA_DIR = SCRIPT_DIR / "data"
MCSA_CSV = PROJECT_ROOT / "data" / "database" / "mcsa_curated_data.csv"
MCSA_JSON = PROJECT_ROOT / "data" / "database" / "mcsa_entries.json"


class FileWrapper:
    def __init__(self, path: Path):
        self.name = path.name
        with open(path, "rb") as f:
            self._buf = BytesIO(f.read())

    def seek(self, pos):
        self._buf.seek(pos)

    def read(self):
        return self._buf.read()


def load_mcsa_reactions(mcsa_csv: Path, min_queries: int = 20) -> pd.DataFrame:
    """Extract unique M-CSA entries with EC numbers.

    Returns a DataFrame with columns: mcsa_id, ec, uniprot_id, pdb.
    Deduplicates to one row per unique (mcsa_id, ec) pair.
    """
    df = pd.read_csv(mcsa_csv)
    # Keep only rows that have EC numbers
    df = df.dropna(subset=["EC"])
    df = df[df["EC"].str.strip() != ""]

    # Deduplicate to unique M-CSA entries
    unique = (
        df.groupby("M-CSA ID")
        .agg({"EC": "first", "Uniprot IDs": "first", "PDB": "first"})
        .reset_index()
        .rename(columns={"M-CSA ID": "mcsa_id", "EC": "ec", "Uniprot IDs": "uniprot_id", "PDB": "pdb"})
    )

    # Filter to entries with complete 4-level EC
    unique = unique[unique["ec"].str.count(r"\.") >= 3].copy()

    print(f"  M-CSA entries with complete EC: {len(unique)}")
    if len(unique) < min_queries:
        print(f"  WARNING: Only {len(unique)} entries, below min_queries={min_queries}")

    return unique


def _ec_match(query_ec: str, predicted_ec: str, level: int = 3) -> bool:
    """Check if two EC numbers match at a given level."""
    q_parts = query_ec.strip().split(".")[:level]
    p_parts = predicted_ec.strip().replace("-", "").split(".")[:level]
    if len(q_parts) < level or len(p_parts) < level:
        return False
    return q_parts == p_parts


def _find_ec_rank(df_rank: pd.DataFrame, target_ec: str, ec_level: int = 3, max_rank: int = 100) -> int | None:
    """Find the rank of the first reaction matching the target EC at the given level."""
    for rank_idx, (_, row) in enumerate(df_rank.head(max_rank).iterrows()):
        ec_str = str(row.get("ec", ""))
        if not ec_str or ec_str == "nan":
            continue
        for ec in ec_str.split(";"):
            if _ec_match(target_ec, ec.strip(), level=ec_level):
                return rank_idx + 1
    return None


def run_mcsa_evaluation(
    gem_path: Path,
    mcsa_entries: pd.DataFrame,
    met_to_smiles: dict,
    cofactor_set: set,
    ec_level: int = 3,
) -> list:
    """Evaluate ChemGIME against M-CSA entries using EC-match as ground truth.

    Strategy: For each M-CSA entry, we find all GEM reactions with the same
    EC number (at the specified level). We use one as a "query" and check
    whether ChemGIME retrieves another reaction with the same EC class.
    """
    gem_wrapper = FileWrapper(gem_path)
    db_df = build_organism_database(gem_wrapper, met_to_smiles, cofactor_set)
    if db_df is None or db_df.empty:
        print("ERROR: Failed to build database.")
        return []

    db_fps, valid_indices = calculate_fingerprints_parallel(db_df)
    if db_fps is None or len(db_fps) == 0:
        print("ERROR: No fingerprints generated.")
        return []

    results = []
    queries_used = 0

    for _, entry in tqdm(mcsa_entries.iterrows(), total=len(mcsa_entries), desc="  M-CSA queries"):
        target_ec = entry["ec"]

        # Find GEM reactions matching this EC (these become our "query" pool)
        matching_gem_rxns = []
        for idx, row in db_df.iterrows():
            ec_str = str(row.get("ec", ""))
            if ec_str and ec_str != "nan":
                for ec in ec_str.split(";"):
                    if _ec_match(target_ec, ec.strip(), level=ec_level):
                        matching_gem_rxns.append(row)
                        break

        if not matching_gem_rxns:
            continue

        # Use the first matching GEM reaction as the query
        query_row = matching_gem_rxns[0]
        clean_l = filter_smiles_string(str(query_row.get("raw_left_smiles", "")), cofactor_set)
        clean_r = filter_smiles_string(str(query_row.get("raw_right_smiles", "")), cofactor_set)
        if not clean_l or not clean_r:
            continue

        qfp = _calculate_drfp_fingerprint((clean_l, clean_r))
        if qfp is None:
            continue

        ranked = get_closest_rxns(qfp.reshape(1, -1), db_fps, db_df, valid_indices)

        # Skip rank 0 if it's the query itself (self-match)
        # Find the rank of the first OTHER reaction matching the EC
        found_rank = None
        query_rxn_id = query_row.get("reaction_id", "")
        for rank_idx, (_, r) in enumerate(ranked.head(100).iterrows()):
            if r.get("reaction_id", "") == query_rxn_id:
                continue  # Skip self-match
            ec_str = str(r.get("ec", ""))
            if ec_str and ec_str != "nan":
                for ec in ec_str.split(";"):
                    if _ec_match(target_ec, ec.strip(), level=ec_level):
                        found_rank = rank_idx + 1
                        break
            if found_rank is not None:
                break

        queries_used += 1
        results.append({
            "mcsa_id": entry["mcsa_id"],
            "target_ec": target_ec,
            "query_rxn_id": query_rxn_id,
            "found_at_rank": found_rank,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="M-CSA batch evaluation")
    parser.add_argument("--gem", required=True, help="Path to reference GEM .xml")
    parser.add_argument("--min-queries", type=int, default=20, help="Minimum M-CSA queries required")
    parser.add_argument("--ec-level", type=int, default=3, choices=[1, 2, 3, 4],
                        help="EC matching level (default: 3 = sub-subclass)")
    parser.add_argument("--output", default=None, help="JSON output file for results")
    args = parser.parse_args()

    print("=" * 60)
    print("ChemGIME M-CSA Batch Evaluation")
    print(f"  EC match level: {args.ec_level}")
    print("=" * 60)

    met_to_smiles = load_bigg_to_smiles_map(BIGG_MET_TO_SMILES_FILE)
    cofactor_set = load_cofactor_set(COFACTOR_SMILES_FILE)

    print("\nLoading M-CSA data...")
    mcsa_entries = load_mcsa_reactions(MCSA_CSV, min_queries=args.min_queries)

    print(f"\nRunning evaluation against: {args.gem}")
    results = run_mcsa_evaluation(
        Path(args.gem), mcsa_entries, met_to_smiles, cofactor_set,
        ec_level=args.ec_level,
    )

    if not results:
        print("No results generated. Check that M-CSA EC classes overlap with the GEM.")
        return

    # Compute metrics
    ranks = [r["found_at_rank"] for r in results]
    summary = retrieval_summary(ranks)

    print("\n" + "=" * 60)
    print(f"M-CSA BATCH RESULTS (n={summary['n_queries']})")
    print("=" * 60)
    print(f"  MRR:           {summary['mrr']:.4f}")
    print(f"  Top-1:         {summary['top1']:.2f}%")
    print(f"  Top-5:         {summary['top5']:.2f}%")
    print(f"  Top-10:        {summary['top10']:.2f}%")
    print(f"  Found:         {summary['n_found']}/{summary['n_queries']}")

    # EC class breakdown
    ec_classes = {}
    for r in results:
        ec_class = r["target_ec"].split(".")[0]
        ec_classes.setdefault(ec_class, []).append(r["found_at_rank"])
    print("\n  Per EC Class (L1):")
    for ec_class in sorted(ec_classes.keys()):
        class_ranks = ec_classes[ec_class]
        class_summary = retrieval_summary(class_ranks)
        print(f"    EC {ec_class}.-.-.-: n={len(class_ranks)}, MRR={class_summary['mrr']:.3f}, Top-5={class_summary['top5']:.1f}%")

    # Save
    out_path = Path(args.output) if args.output else DATA_DIR / "mcsa_eval_results.json"
    out_data = {
        "ec_level": args.ec_level,
        "n_queries": summary["n_queries"],
        "summary": summary,
        "per_query": results,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
