
"""
ChemGIME Threshold Evaluation Script
====================================

Processes the iML1515 benchmark dataset to determine the optimal Tanimoto similarity threshold
for maximizing F1 score in enzyme identification.

Outputs:
1. roc_curve.png
2. f1_vs_threshold.png
3. optimal_threshold_summary.md
"""

import sys
import os
import argparse
from pathlib import Path
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc

# --- Import ChemGIME core (no Streamlit dependency) ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = SCRIPT_DIR / "data"
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

# --- Serial Execution Monkey-Patch ---
# Force serial execution to avoid ProcessPoolExecutor overhead/deadlocks
# which are common in small-batch benchmarks or restricted environments.
def serial_calculate_fingerprints(df):
    print("  (Running in forced SERIAL mode)")
    smiles_pairs = list(zip(df['raw_left_smiles'], df['raw_right_smiles']))
    results = [_calculate_drfp_fingerprint(p) for p in tqdm(smiles_pairs, desc="  Fingerprinting")]

    valid_fps = [fp for fp in results if fp is not None]
    valid_indices = [i for i, fp in enumerate(results) if fp is not None]

    if not valid_fps:
        return None, None

    return np.array(valid_fps), valid_indices

# Apply monkey-patch
import src.core.fingerprints as _fp_module
_fp_module.calculate_fingerprints_parallel = serial_calculate_fingerprints

# --- Configuration ---
TRAIN_GEM_PATH = DATA_DIR / "iML1515_train.xml"
TEST_CSV_PATH = DATA_DIR / "iML1515_test.csv"

# --- Helper Classes/Functions ---

class FileWrapper:
    """Simple file wrapper for build_organism_database compatible with BytesIO."""
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

def normalize_gpr(gpr: str) -> set:
    """Extract gene IDs from GPR string."""
    if not isinstance(gpr, str): return set()
    gpr_clean = gpr.replace('(', ' ').replace(')', ' ')
    gpr_clean = gpr_clean.replace(' and ', ' ').replace(' or ', ' ')
    return set(gpr_clean.split())

def check_gpr_match(query_gpr: str, db_gpr: str) -> bool:
    """Check if any gene in query GPR matches DB GPR."""
    query_genes = normalize_gpr(query_gpr)
    db_genes = normalize_gpr(db_gpr)
    return bool(query_genes & db_genes)

# --- Main Evaluation Logic ---

def main():
    print("=" * 60)
    print("ChemGIME Threshold Evaluation")
    print("=" * 60)

    # 1. Load Helper Data
    print("Loading helper data...")
    if not BIGG_MET_TO_SMILES_FILE.exists():
        print(f"CRITICAL: BiGG mapping missing at {BIGG_MET_TO_SMILES_FILE}")
        return

    if not COFACTOR_SMILES_FILE.exists():
        print(f"WARNING: Cofactor file missing at {COFACTOR_SMILES_FILE}. Cofactor filtering will be disabled.")

    met_to_smiles = load_bigg_to_smiles_map(BIGG_MET_TO_SMILES_FILE)
    cofactor_set = load_cofactor_set(COFACTOR_SMILES_FILE)

    # 2. Build Training DB
    print(f"Building training DB from {TRAIN_GEM_PATH.name}...")
    gem_wrapper = FileWrapper(TRAIN_GEM_PATH)
    db_df = build_organism_database(gem_wrapper, met_to_smiles, cofactor_set)
    if db_df is None or db_df.empty:
        sys.exit("Failed to build training database.")

    # 3. Generate DB Fingerprints
    print("Generating database fingerprints...")
    # Note: calculate_fingerprints_parallel might rely on global variables in ChemGIME?
    # No, it takes df as input.
    db_fps, valid_indices = calculate_fingerprints_parallel(db_df)
    if db_fps is None:
        sys.exit("Database fingerprinting failed.")
    
    print(f"  Database Reactions: {len(db_df)}")
    print(f"  Valid Fingerprints: {len(db_fps)}")

    # 4. Load Test Data
    print(f"Loading test queries from {TEST_CSV_PATH.name}...")
    test_df = pd.read_csv(TEST_CSV_PATH)
    
    # 5. Run Queries & Collect Metrics
    print(f"Processing {len(test_df)} queries...")
    results = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        query_left = row['left_smiles']
        query_right = row['right_smiles']
        true_gpr = row['true_gpr']
        
        # Filter cofactors
        cleaned_left = filter_smiles_string(query_left, cofactor_set)
        cleaned_right = filter_smiles_string(query_right, cofactor_set)

        if not cleaned_left or not cleaned_right:
            # Empty after filter -> effectively a generic/cofactor-only reaction
            # We treat this as a non-retrieval (Distance = 1.0, Score = 0.0)
            results.append({
                'match_correct': False,
                'tanimoto_dist': 1.0,
                'similarity': 0.0
            })
            continue

        query_fp = _calculate_drfp_fingerprint((cleaned_left, cleaned_right))
        if query_fp is None:
            results.append({
                'match_correct': False,
                'tanimoto_dist': 1.0,
                'similarity': 0.0
            })
            continue

        # Find closest match (Rank 1 only)
        # get_closest_rxns returns DataFrame with 'tan_distance' and 'gpr'
        # We need to pass query_fp as array
        query_fp_arr = np.array([query_fp])
        ranked_df = get_closest_rxns(query_fp_arr, db_fps, db_df, valid_indices)

        if ranked_df.empty:
            results.append({'match_correct': False, 'tanimoto_dist': 1.0, 'similarity': 0.0})
            continue

        # Inspect Top 1 Hit
        top_hit = ranked_df.iloc[0]
        top_gpr = top_hit.get('gpr', '')
        # dist is Tanimoto distance (0=identical, 1=distinct)
        dist = top_hit.get('tan_distance', 1.0)
        similarity = 1.0 - dist
        
        is_match = check_gpr_match(true_gpr, top_gpr)

        results.append({
            'match_correct': is_match,
            'tanimoto_dist': dist,
            'similarity': similarity
        })

    results_df = pd.DataFrame(results)
    print(f"\nCollected {len(results_df)} query results.")

    # 6. Threshold Analysis (0.1 to 0.9) (Distance Thresholds)
    # Tanimoto Distance Threshold T means: Accept if dist <= T (Similarity >= 1-T)
    # Range: 0.1 (Strict) -> 0.9 (Loose)
    thresholds = np.arange(0.1, 0.91, 0.01)
    metrics = []

    for t in thresholds:
        # Prediction Logic:
        # If dist <= t: Predict matching GPR (Positive Prediction)
        #   If match_correct: True Positive (TP)
        #   If not match_correct: False Positive (FP)
        # If dist > t: Predict "No Match" (Negative Prediction)
        #   Since a true GPR *always* exists in reality for these queries (they are valid rxns),
        #   predicting "No Match" is a False Negative relative to the goal of "finding the enzyme".

        # TP: Dist <= T AND Correct
        tp = ((results_df['tanimoto_dist'] <= t) & (results_df['match_correct'])).sum()
        
        # FP: Dist <= T AND Incorrect
        fp = ((results_df['tanimoto_dist'] <= t) & (~results_df['match_correct'])).sum()
        
        # FN: Dist > T (We missed the opportunity to predict, or we predicted nothing)
        # Note: If we predicted (Dist <= T) but got it wrong, is it also an FN?
        # In multi-class classification, a misclassification is 1 FP (for the wrong class) and 1 FN (for the right class).
        # Here, we are treating "Correct Retrieval" as the binary task.
        # "Did we find the right enzyme?" for this query.
        # So total positives (P) = Total Queries (N).
        # Recall = TP / N.
        # Precision = TP / (TP + FP).
        
        fn = len(results_df) - tp - fp # Wait no. FN is simple: Total Docs - Retrieved.
        # Wait, if dist > T, we retrieved nothing. That is an FN.
        # If dist <= T but wrong, is it an FN?
        # Yes, we failed to retrieve the *correct* one. We retrieved a *wrong* one.
        # So "TP" is the only success case.
        # Everything else is a failure.
        # But Precision only penalizes "Wrong Retrievals" (FP).
        # Recall penalizes "Missed Retrievals" (FN).
        
        n_predictions = tp + fp
        precision = tp / n_predictions if n_predictions > 0 else 1.0 # High precision if we are too strict? Or 0? usually 1.
        recall = tp / len(results_df) # Recall relative to total queries
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics.append({
            'threshold': t,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })

    metrics_df = pd.DataFrame(metrics)
    best_row = metrics_df.loc[metrics_df['f1'].idxmax()]
    optimal_dist = best_row['threshold']
    max_f1 = best_row['f1']

    print(f"\nOptimal Tanimoto Distance Threshold: {optimal_dist:.2f}")
    print(f"Max F1 Score: {max_f1:.4f}")

    # 7. Generate Plots
    
    # Plot 1: F1 vs Threshold
    plt.figure(figsize=(8, 6))
    plt.plot(metrics_df['threshold'], metrics_df['f1'], label='F1 Score', color='blue', linewidth=2)
    plt.plot(metrics_df['threshold'], metrics_df['precision'], label='Precision', linestyle='--', color='green', alpha=0.7)
    plt.plot(metrics_df['threshold'], metrics_df['recall'], label='Recall', linestyle=':', color='red', alpha=0.7)
    plt.axvline(optimal_dist, color='black', linestyle='--', label=f'Optimal ({optimal_dist:.2f})')
    plt.xlabel('Tanimoto Distance Threshold')
    plt.ylabel('Score')
    plt.title('Performance vs. Tanimoto Distance Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('f1_vs_threshold.png', dpi=300)
    plt.close()
    print("Saved f1_vs_threshold.png")

    # Plot 2: ROC Curve
    # y_true: whether the top hit was correct
    # y_score: similarity (1 - distance) because higher score = more likely correct
    # Wait, ROC usually evaluates the ranking capability.
    # Here, for each query, we have ONE score (the top hit's score) and ONE label (is it correct).
    # This evaluates: "Given a Top-1 prediction with score S, how well does S predict correctness?"
    # This is exactly what we need to verify if the score is a good confidence metric.
    
    y_true = results_df['match_correct'].astype(int)
    y_score = results_df['similarity']
    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Top-1 Match Confidence)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig('roc_curve.png', dpi=300)
    plt.close()
    print("Saved roc_curve.png")

    # 8. Output Summary
    summary_text = f"""
# ChemGIME Optimal Threshold Analysis

## Executive Summary
The empirical evaluation on the iML1515 benchmark dataset identifies **{optimal_dist:.2f}** as the optimal Tanimoto distance threshold for maximizing retrieval performance.

- **Optimal Threshold:** {optimal_dist:.2f} (Distance)
- **Max F1 Score:** {max_f1:.4f}

## Methodology & Chain of Thought
To determine this threshold, we performed a systematic sweep of Tanimoto distance thresholds ($T$) from 0.1 to 0.9. For each threshold, we evaluated the system's ability to correctly identify the true enzyme (GPR) associated with a reaction query.

### Metric Definitions
- **Precision**: The accuracy of positive predictions. Calculated as $TP / (TP + FP)$.
  - *Interpretation*: When the system predicts a match (distance <= T), how often is it correct?
- **Recall**: The ability to find the correct answer among all queries. Calculated as $TP / Total Queries$.
  - *Interpretation*: Out of all test reactions, what fraction did we correctly identify?
- **F1 Score**: The harmonic mean of Precision and Recall. Calculated as $2 * (Precision * Recall) / (Precision + Recall)$.
  - *Why maximize F1?* It ensures a balanced system that is neither too conservative (missing valid hits) nor too permissive (returning incorrect enzymes).

### Evaluation Logic
1.  **Low Thresholds (< 0.3)**: Highly specific but poor recall. The system only accepts near-identical matches, missing many valid homologues.
2.  **High Thresholds (> 0.7)**: High recall but poor precision. The system accepts distant matches which are likely false positives.
3.  **Optimal Point ({optimal_dist:.2f})**: This threshold represents the peak of the performance curve, where the trade-off between strictness and coverage is optimized.

## Detailed Results at Optimal Threshold ({optimal_dist:.2f})

| Metric | Value | Meaning |
|:-------|:------|:--------|
| **Precision** | **{best_row['precision']:.4f}** | {best_row['precision']*100:.1f}% of accepted predictions were correct. |
| **Recall** | **{best_row['recall']:.4f}** | {best_row['recall']*100:.1f}% of total queries were correctly solved. |
| **F1** | **{max_f1:.4f}** | The balanced performance score. |
| **AUC** | **{roc_auc:.4f}** | The model's ability to rank correct matches higher than incorrect ones (0.5 = random, 1.0 = perfect). |

## Recommendation
We recommend configuring ChemGIME with a **Tanimoto cut-off of {optimal_dist:.2f}**. Predictions with a distance greater than {optimal_dist:.2f} should be flagged as "Low Confidence" or filtered out to maintain system reliability.
"""
    with open("optimal_threshold_summary.md", "w") as f:
        f.write(summary_text.strip())
    
    print("\nSummary written to optimal_threshold_summary.md")

if __name__ == "__main__":
    main()
