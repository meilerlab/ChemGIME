"""
Split-GEM Benchmark Runner Script
==================================

This script runs the ChemGIME benchmark using the Split-GEM methodology.

Workflow:
---------
1. Load the training GEM (80% of reactions) and build the reaction database.
2. Load the test queries CSV (20% of reactions with ground-truth GPRs).
3. For each query, generate a DRFP fingerprint and find the closest reactions.
4. Calculate Top-k accuracy metrics (Top-1, Top-5, Top-10).

The benchmark measures how well ChemGIME can retrieve the correct enzyme/gene
association given only the chemical transformation.
"""

import sys
from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd
from tqdm import tqdm

# --- Import ChemGIME core (no Streamlit dependency) ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import (
    load_bigg_to_smiles_map,
    load_cofactor_set,
    build_organism_database,
    calculate_fingerprints_parallel,
    get_closest_rxns,
    _calculate_drfp_fingerprint,
    filter_smiles_string,
    BIGG_MET_TO_SMILES_FILE,
    COFACTOR_SMILES_FILE,
)
from src.core.metrics import mean_reciprocal_rank, retrieval_summary

# --- Configuration ---
DATA_DIR = SCRIPT_DIR / "data"
TRAIN_GEM_PATH = DATA_DIR / "iML1515_train.xml"
TEST_CSV_PATH = DATA_DIR / "iML1515_test.csv"


class FileWrapper:
    """
    A simple file-like wrapper for build_organism_database.
    The function expects an object with .seek(), .read(), and .name attributes.
    """
    def __init__(self, path: Path):
        self.path = path
        self.name = path.name
        with open(path, 'rb') as f:
            self._content = f.read()
        self._buffer = BytesIO(self._content)
    
    def seek(self, pos):
        self._buffer.seek(pos)
    
    def read(self):
        return self._buffer.read()


def calculate_accuracy(results: list, k: int) -> float:
    """Calculate top-k accuracy."""
    hits = sum(1 for r in results if r['found_at_rank'] is not None and r['found_at_rank'] <= k)
    return hits / len(results) * 100 if results else 0.0


def normalize_gpr(gpr: str) -> set:
    """
    Extract individual gene IDs from a GPR string.
    GPR strings use 'and', 'or', and parentheses.
    We extract the raw gene identifiers for comparison.
    """
    # Remove parentheses and logical operators
    gpr_clean = gpr.replace('(', ' ').replace(')', ' ')
    gpr_clean = gpr_clean.replace(' and ', ' ').replace(' or ', ' ')
    genes = set(gpr_clean.split())
    return genes


def check_gpr_match(query_gpr: str, db_gpr: str) -> bool:
    """
    Check if the query GPR matches the database GPR.
    We match if any gene in the query GPR is also in the DB GPR.
    """
    query_genes = normalize_gpr(query_gpr)
    db_genes = normalize_gpr(db_gpr)
    return bool(query_genes & db_genes)  # Intersection is non-empty


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", default=None, help="Path to test CSV")
    parser.add_argument("--serial", action="store_true", help="Run in serial mode (disable parallel fingerprinting)")
    args = parser.parse_args()

    print("=" * 60)
    print("ChemGIME Split-GEM Benchmark")
    if args.serial:
        # Monkey patch parallel function to use serial map
        print("  Running in SERIAL mode (parallel disabled)")
        global calculate_fingerprints_parallel
        def serial_calculate_fingerprints(df):
            smiles_pairs = list(zip(df['raw_left_smiles'], df['raw_right_smiles']))
            # Use list comprehension or map
            # We must import _calculate_drfp_fingerprint
            results = [_calculate_drfp_fingerprint(p) for p in tqdm(smiles_pairs, desc="  Fingerprinting (Serial)")]
            # Filter results
            valid_indices = [i for i, fp in enumerate(results) if fp is not None]
            if not valid_indices:
                return None, []
            
            fps_array = np.stack([results[i] for i in valid_indices])
            return fps_array, valid_indices
        
        calculate_fingerprints_parallel = serial_calculate_fingerprints
        
    print("=" * 60)
    
    # --- Step 1: Load helper data ---
    print("\n[1/4] Loading helper data...")
    met_to_smiles = load_bigg_to_smiles_map(BIGG_MET_TO_SMILES_FILE)
    cofactor_set = load_cofactor_set(COFACTOR_SMILES_FILE)
    print(f"  Loaded {len(met_to_smiles)} metabolite mappings")
    print(f"  Loaded {len(cofactor_set)} cofactor SMILES")
    
    # --- Step 2: Build database from training GEM ---
    print(f"\n[2/4] Building database from: {TRAIN_GEM_PATH}")
    gem_wrapper = FileWrapper(TRAIN_GEM_PATH)
    db_df = build_organism_database(gem_wrapper, met_to_smiles, cofactor_set)
    
    if db_df is None or db_df.empty:
        print("ERROR: Failed to build database. Exiting.")
        return
    
    print(f"  Database has {len(db_df)} reactions")
    
    # Generate fingerprints for the database
    print("  Generating fingerprints...")
    db_fps, valid_indices = calculate_fingerprints_parallel(db_df)
    
    if db_fps is None or len(db_fps) == 0:
        print("ERROR: Failed to generate fingerprints. Exiting.")
        return
    
    print(f"  Generated {len(db_fps)} valid fingerprints")
    
    # --- Step 3: Load test queries and run benchmark ---
    test_path = Path(args.test_file) if args.test_file else TEST_CSV_PATH
    print(f"\n[3/4] Loading test queries from: {test_path}")
    test_df = pd.read_csv(test_path)
    print(f"  Loaded {len(test_df)} test queries")
    
    results = []

    print("\n  Running queries...")
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="  Queries"):
        query_left = row['left_smiles']
        query_right = row['right_smiles']
        true_gpr = row['true_gpr']
        
        # Filter cofactors from query SMILES
        clean_left = filter_smiles_string(query_left, cofactor_set)
        clean_right = filter_smiles_string(query_right, cofactor_set)
        
        # Skip if filtering removed all SMILES
        if not clean_left or not clean_right:
            results.append({
                'reaction_id': row['reaction_id'],
                'true_gpr': true_gpr,
                'found_at_rank': None,
                'reason': 'empty_after_filter'
            })
            continue
        
        # Generate query fingerprint
        query_fp = _calculate_drfp_fingerprint((clean_left, clean_right))
        
        if query_fp is None:
            results.append({
                'reaction_id': row['reaction_id'],
                'true_gpr': true_gpr,
                'found_at_rank': None,
                'reason': 'fingerprint_failed'
            })
            continue
        
        # Find closest reactions
        query_fp_array = np.array([query_fp])
        ranked_df = get_closest_rxns(query_fp_array, db_fps, db_df, valid_indices)
        
        # Check where the true GPR appears in the ranked list
        found_at_rank = None
        for rank, db_row in ranked_df.head(100).iterrows():  # Check top 100
            db_gpr = db_row.get('gpr', '')
            if db_gpr and check_gpr_match(true_gpr, db_gpr):
                found_at_rank = rank + 1  # 1-indexed
                break
        
        results.append({
            'reaction_id': row['reaction_id'],
            'true_gpr': true_gpr,
            'found_at_rank': found_at_rank,
            'reason': 'success' if found_at_rank else 'not_in_top100'
        })
    
    # --- Step 4: Calculate and print metrics ---
    print("\n[4/4] Calculating metrics...")
    print("=" * 60)
    
    valid_results = [r for r in results if r['reason'] != 'empty_after_filter' and r['reason'] != 'fingerprint_failed']

    # Extract ranks for MRR computation
    valid_ranks = [r['found_at_rank'] for r in valid_results]
    summary = retrieval_summary(valid_ranks, k_values=(1, 5, 10, 50, 100))

    print(f"\n  BENCHMARK RESULTS")
    print(f"   Total Queries:    {len(test_df)}")
    print(f"   Valid Queries:    {summary['n_queries']}")
    print(f"   Found:            {summary['n_found']}")
    print(f"   Skipped (filter): {sum(1 for r in results if r['reason'] == 'empty_after_filter')}")
    print(f"   Skipped (FP):     {sum(1 for r in results if r['reason'] == 'fingerprint_failed')}")
    print()
    print(f"   MRR:              {summary['mrr']:.4f}")
    print(f"   Top-1 Accuracy:   {summary['top1']:.2f}%")
    print(f"   Top-5 Accuracy:   {summary['top5']:.2f}%")
    print(f"   Top-10 Accuracy:  {summary['top10']:.2f}%")
    print(f"   Top-50 Accuracy:  {summary['top50']:.2f}%")
    print(f"   Top-100 Accuracy: {summary['top100']:.2f}%")

    print("=" * 60)
    
    # Save detailed results
    results_df = pd.DataFrame(results)
    results_path = DATA_DIR / "benchmark_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n✅ Detailed results saved to: {results_path}")


if __name__ == "__main__":
    main()
