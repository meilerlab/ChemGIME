"""
M-CSA Novelty Benchmark Evaluation Script
==========================================

This script evaluates ChemGIME's ability to find chemically similar reactions
for novel/unmodeled enzymatic reactions from M-CSA.

Unlike the Split-GEM benchmark (which expects exact GPR matches), this script
evaluates:
1. Minimum Tanimoto distance to closest GEM reaction
2. EC number class similarity (e.g., do both start with "3.1"?)
3. Proportion of queries with "chemically similar" hits (distance <= 0.5)

Usage:
    python run_novelty_benchmark.py [--gem-file PATH] [--test-file PATH]
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

# --- Configuration ---
DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_GEM_PATH = PROJECT_ROOT / "data" / "iML1515.xml"
DEFAULT_TEST_PATH = DATA_DIR / "novel_mcsa_reactions.csv"


class FileWrapper:
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


def ec_class_match(ec1: str, ec2: str, level: int = 2) -> bool:
    """
    Check if two EC numbers match at a given level.
    Level 1: First digit (e.g., "3" matches "3")
    Level 2: First two digits (e.g., "3.1" matches "3.1")
    Level 3: First three digits (e.g., "3.1.3" matches "3.1.3")
    """
    if not ec1 or not ec2:
        return False
    
    # Handle multiple EC numbers (separated by ;)
    ec1_list = [e.strip() for e in str(ec1).split(';')]
    ec2_list = [e.strip() for e in str(ec2).split(';')]
    
    for e1 in ec1_list:
        for e2 in ec2_list:
            parts1 = e1.split('.')
            parts2 = e2.split('.')
            
            if len(parts1) >= level and len(parts2) >= level:
                if parts1[:level] == parts2[:level]:
                    return True
    return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="M-CSA Novelty Benchmark Evaluator")
    parser.add_argument("--gem-file", default=None, help="Path to full GEM file")
    parser.add_argument("--test-file", default=None, help="Path to novelty test CSV")
    parser.add_argument("--similarity-threshold", type=float, default=0.5,
                        help="Tanimoto distance threshold for 'similar' reactions (default: 0.5)")
    args = parser.parse_args()

    print("=" * 70)
    print("ChemGIME M-CSA Novelty Benchmark")
    print("=" * 70)
    
    # --- Step 1: Load helper data ---
    print("\n[1/4] Loading helper data...")
    met_to_smiles = load_bigg_to_smiles_map(BIGG_MET_TO_SMILES_FILE)
    cofactor_set = load_cofactor_set(COFACTOR_SMILES_FILE)
    print(f"  Loaded {len(met_to_smiles)} metabolite mappings")
    print(f"  Loaded {len(cofactor_set)} cofactor SMILES")
    
    # --- Step 2: Build database from FULL GEM ---
    gem_path = Path(args.gem_file) if args.gem_file else DEFAULT_GEM_PATH
    print(f"\n[2/4] Building database from: {gem_path}")
    gem_wrapper = FileWrapper(gem_path)
    db_df = build_organism_database(gem_wrapper, met_to_smiles, cofactor_set)
    
    if db_df is None or db_df.empty:
        print("ERROR: Failed to build database. Exiting.")
        return
    
    print(f"  Database has {len(db_df)} reactions")
    
    # Generate fingerprints
    print("  Generating fingerprints...")
    db_fps, valid_indices = calculate_fingerprints_parallel(db_df)
    
    if db_fps is None or len(db_fps) == 0:
        print("ERROR: Failed to generate fingerprints. Exiting.")
        return
    
    print(f"  Generated {len(db_fps)} valid fingerprints")
    
    # --- Step 3: Load novelty test queries ---
    test_path = Path(args.test_file) if args.test_file else DEFAULT_TEST_PATH
    print(f"\n[3/4] Loading novelty queries from: {test_path}")
    test_df = pd.read_csv(test_path)
    print(f"  Loaded {len(test_df)} novel reaction queries")
    
    results = []
    
    print("\n  Running similarity search...")
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="  Queries"):
        query_left = row['left_smiles']
        query_right = row['right_smiles']
        true_ec = row.get('true_ec', '')
        
        # Filter cofactors
        clean_left = filter_smiles_string(query_left, cofactor_set)
        clean_right = filter_smiles_string(query_right, cofactor_set)
        
        if not clean_left or not clean_right:
            results.append({
                'reaction_id': row['reaction_id'],
                'true_ec': true_ec,
                'min_distance': None,
                'closest_rxn_id': None,
                'closest_ec': None,
                'ec_class_match_l1': False,
                'ec_class_match_l2': False,
                'status': 'empty_after_filter'
            })
            continue
        
        # Generate query fingerprint
        query_fp = _calculate_drfp_fingerprint((clean_left, clean_right))
        
        if query_fp is None:
            results.append({
                'reaction_id': row['reaction_id'],
                'true_ec': true_ec,
                'min_distance': None,
                'closest_rxn_id': None,
                'closest_ec': None,
                'ec_class_match_l1': False,
                'ec_class_match_l2': False,
                'status': 'fingerprint_failed'
            })
            continue
        
        # Find closest reactions
        query_fp_array = np.array([query_fp])
        ranked_df = get_closest_rxns(query_fp_array, db_fps, db_df, valid_indices)
        
        # Get top hit
        top_hit = ranked_df.iloc[0] if len(ranked_df) > 0 else None
        
        if top_hit is not None:
            min_dist = top_hit['tan_distance']
            closest_id = top_hit.get('reaction_id', 'unknown')
            closest_ec = top_hit.get('ec', '')
            
            results.append({
                'reaction_id': row['reaction_id'],
                'true_ec': true_ec,
                'min_distance': min_dist,
                'closest_rxn_id': closest_id,
                'closest_ec': closest_ec,
                'ec_class_match_l1': ec_class_match(true_ec, closest_ec, level=1),
                'ec_class_match_l2': ec_class_match(true_ec, closest_ec, level=2),
                'status': 'success'
            })
        else:
            results.append({
                'reaction_id': row['reaction_id'],
                'true_ec': true_ec,
                'min_distance': None,
                'closest_rxn_id': None,
                'closest_ec': None,
                'ec_class_match_l1': False,
                'ec_class_match_l2': False,
                'status': 'no_hits'
            })
    
    # --- Step 4: Calculate and print metrics ---
    print("\n[4/4] Calculating metrics...")
    print("=" * 70)
    
    valid_results = [r for r in results if r['status'] == 'success']
    
    if not valid_results:
        print("No valid results to analyze.")
        return
    
    distances = [r['min_distance'] for r in valid_results]
    threshold = args.similarity_threshold
    
    # Calculate metrics
    mean_dist = np.mean(distances)
    median_dist = np.median(distances)
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    
    similar_count = sum(1 for d in distances if d <= threshold)
    similar_pct = similar_count / len(valid_results) * 100
    
    ec_l1_matches = sum(1 for r in valid_results if r['ec_class_match_l1'])
    ec_l2_matches = sum(1 for r in valid_results if r['ec_class_match_l2'])
    
    print(f"\n📊 NOVELTY BENCHMARK RESULTS")
    print(f"   Total Queries:     {len(test_df)}")
    print(f"   Valid Queries:     {len(valid_results)}")
    print(f"   Skipped (filter):  {sum(1 for r in results if r['status'] == 'empty_after_filter')}")
    print(f"   Skipped (FP):      {sum(1 for r in results if r['status'] == 'fingerprint_failed')}")
    print()
    print(f"   📏 TANIMOTO DISTANCE (lower = more similar)")
    print(f"      Mean:           {mean_dist:.4f}")
    print(f"      Median:         {median_dist:.4f}")
    print(f"      Min:            {min_dist:.4f}")
    print(f"      Max:            {max_dist:.4f}")
    print()
    print(f"   🎯 SIMILARITY HITS (distance ≤ {threshold})")
    print(f"      Count:          {similar_count}/{len(valid_results)}")
    print(f"      Percentage:     {similar_pct:.1f}%")
    print()
    print(f"   🧬 EC CLASS MATCHING")
    print(f"      Level 1 (X._._._): {ec_l1_matches}/{len(valid_results)} ({ec_l1_matches/len(valid_results)*100:.1f}%)")
    print(f"      Level 2 (X.Y._._): {ec_l2_matches}/{len(valid_results)} ({ec_l2_matches/len(valid_results)*100:.1f}%)")
    print("=" * 70)
    
    # Print detailed results table
    print("\n📋 DETAILED RESULTS:")
    print("-" * 70)
    for r in valid_results:
        ec_match = "✓" if r['ec_class_match_l2'] else "✗"
        print(f"  {r['reaction_id']}: dist={r['min_distance']:.4f} | "
              f"true_ec={r['true_ec']} → closest_ec={r['closest_ec']} [{ec_match}]")
    print("-" * 70)
    
    # Save detailed results
    results_df = pd.DataFrame(results)
    results_path = DATA_DIR / "novelty_benchmark_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n✅ Detailed results saved to: {results_path}")


if __name__ == "__main__":
    main()
