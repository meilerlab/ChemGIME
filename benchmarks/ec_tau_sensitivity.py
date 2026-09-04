"""
EC level-3 accuracy vs Tanimoto distance threshold (τ) sensitivity analysis.
=============================================================================

Mechanistic explanation for the gap between EC L3 accuracy (57–78 %) and
Top-10 retrieval accuracy (78–90 %) in Table 2.

Method: leave-one-out on the full iML1515 reaction database.
For each reaction with a complete EC sub-subclass annotation (all three
numeric parts resolved), the reaction is held out, distances to the
remaining database reactions are computed, and the similarity-weighted vote
(identical to Eq. 1–2 in §Methods) is evaluated at each τ in the sweep.

Three per-τ statistics are reported:
  - conditional_accuracy  : fraction of held-out reactions that have ≥1
                            voter within τ AND receive the correct EC L3
                            prediction (denominator = queries with ≥1 voter)
  - unconditional_accuracy: same numerator but denominator = all queries
                            (no-voter queries count as wrong)
  - coverage              : fraction of queries with ≥1 voter within τ

Outputs
-------
  benchmarks/data/ec_tau_sensitivity.json
  benchmarks/data/ec_tau_sensitivity_figure.png

Usage
-----
  python benchmarks/ec_tau_sensitivity.py [--fp-type drfp|qrfp] [--cofactors keep|filter]
  python benchmarks/ec_tau_sensitivity.py --all  # DRFP kept + QRFP kept
"""

from __future__ import annotations

import argparse
import json
import sys
from io import BytesIO
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = SCRIPT_DIR / "data"
GEM_PATH = PROJECT_ROOT / "iML1515.xml"
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import (
    BIGG_MET_TO_SMILES_FILE,
    COFACTOR_SMILES_FILE,
    _calculate_drfp_fingerprint,
    build_organism_database,
    filter_smiles_string,
    load_bigg_to_smiles_map,
    load_cofactor_set,
)

TAU_VALUES: list[float] = [round(t, 2) for t in np.arange(0.05, 0.91, 0.05).tolist()]
ALPHA = 2  # quadratic weight exponent (matches manuscript Eq. 1)


class FileWrapper:
    def __init__(self, path: Path):
        self.name = path.name
        with open(path, "rb") as f:
            self._buf = BytesIO(f.read())

    def seek(self, pos: int) -> None:
        self._buf.seek(pos)

    def read(self) -> bytes:
        return self._buf.read()


def _ec_l3_key(ec: str) -> str | None:
    """Return normalised EC L3 key (e.g. '2.7.1.-') or None if incomplete."""
    parts = ec.strip().split(".")
    if len(parts) < 3:
        return None
    if not all(p.isdigit() for p in parts[:3]):
        return None
    return f"{parts[0]}.{parts[1]}.{parts[2]}.-"


def _extract_ec_l3_set(ec_string: str) -> set[str]:
    """Return set of EC L3 keys from a semicolon-joined EC string."""
    result: set[str] = set()
    if not ec_string or ec_string == "nan":
        return result
    for ec in ec_string.split(";"):
        key = _ec_l3_key(ec.strip())
        if key:
            result.add(key)
    return result


def _try_qrfp(left: str, right: str) -> np.ndarray | None:
    """Compute QRFP fingerprint; falls back to None on failure."""
    try:
        from src.core.fingerprints import _calculate_qrfp_fingerprint  # type: ignore[import]
        return _calculate_qrfp_fingerprint((left, right))
    except Exception:
        return None


def _compute_fingerprints(
    db_df: pd.DataFrame, fp_type: str, cofactor_set: set
) -> tuple[np.ndarray, list[int]]:
    """Compute fingerprints for all database reactions."""
    fps: list[np.ndarray] = []
    valid_idx: list[int] = []
    for i, row in db_df.iterrows():
        left = row.get("raw_left_smiles", "")
        right = row.get("raw_right_smiles", "")
        if fp_type == "drfp":
            fp = _calculate_drfp_fingerprint((str(left), str(right)))
        else:
            fp = _try_qrfp(str(left), str(right))
        if fp is not None:
            fps.append(fp)
            valid_idx.append(int(i))
    if not fps:
        return np.empty((0,)), []
    return np.stack(fps, axis=0), valid_idx


def _compute_fingerprints_cofactor_filtered(
    db_df: pd.DataFrame, fp_type: str, cofactor_set: set
) -> tuple[np.ndarray, list[int]]:
    """Compute fingerprints after cofactor filtering."""
    fps: list[np.ndarray] = []
    valid_idx: list[int] = []
    for i, row in db_df.iterrows():
        left = filter_smiles_string(str(row.get("raw_left_smiles", "")), cofactor_set)
        right = filter_smiles_string(str(row.get("raw_right_smiles", "")), cofactor_set)
        if not left or not right:
            continue
        if fp_type == "drfp":
            fp = _calculate_drfp_fingerprint((left, right))
        else:
            fp = _try_qrfp(left, right)
        if fp is not None:
            fps.append(fp)
            valid_idx.append(int(i))
    if not fps:
        return np.empty((0,)), []
    return np.stack(fps, axis=0), valid_idx


def _tanimoto_distances(query_fp: np.ndarray, db_fps: np.ndarray) -> np.ndarray:
    """Compute Tanimoto distance from query_fp to each row of db_fps."""
    q = query_fp.astype(bool)
    d = db_fps.astype(bool)
    intersection = (q & d).sum(axis=1)
    union = (q | d).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sim = np.where(union > 0, intersection / union, 0.0)
    return 1.0 - sim


def _vote_ec_l3(
    distances: np.ndarray,
    ec_l3_sets: list[set[str]],
    exclude_idx: int,
    tau: float,
) -> str | None:
    """
    Similarity-weighted vote for EC L3.

    Returns the predicted EC L3 key, or None if no valid voter exists
    within tau (after excluding exclude_idx).
    """
    scores: dict[str, float] = {}
    for pos, (dist, ec_set) in enumerate(zip(distances, ec_l3_sets)):
        if pos == exclude_idx:
            continue
        if dist > tau:
            continue
        if not ec_set:
            continue
        weight = (1.0 - dist) ** ALPHA
        for key in ec_set:
            scores[key] = scores.get(key, 0.0) + weight
    if not scores:
        return None
    return max(scores, key=scores.__getitem__)


def run_sensitivity(
    gem_path: Path,
    fp_type: str,
    cofactors: str,
    tau_values: Sequence[float] = TAU_VALUES,
) -> dict:
    """Run leave-one-out τ-sensitivity analysis on a GEM."""
    print(f"\n  GEM: {gem_path.name}  FP: {fp_type}  Cofactors: {cofactors}")

    met_to_smiles = load_bigg_to_smiles_map(BIGG_MET_TO_SMILES_FILE)
    cofactor_set = load_cofactor_set(COFACTOR_SMILES_FILE)

    gem_wrapper = FileWrapper(gem_path)
    db_df = build_organism_database(gem_wrapper, met_to_smiles, cofactor_set)
    db_df = db_df.reset_index(drop=True)
    print(f"  Database reactions: {len(db_df)}")

    print("  Computing fingerprints ...")
    if cofactors == "keep":
        fps, valid_idx = _compute_fingerprints(db_df, fp_type, cofactor_set)
    else:
        fps, valid_idx = _compute_fingerprints_cofactor_filtered(db_df, fp_type, cofactor_set)

    if len(fps) == 0:
        raise RuntimeError("No fingerprints generated.")
    print(f"  Valid fingerprints: {len(valid_idx)}")

    # Subset db_df to fingerprint-valid rows
    db_sub = db_df.loc[valid_idx].reset_index(drop=True)

    # Pre-compute EC L3 sets for all valid reactions
    ec_l3_sets: list[set[str]] = [
        _extract_ec_l3_set(str(row.get("ec", ""))) for _, row in db_sub.iterrows()
    ]

    # Identify query set: reactions with ≥1 known EC L3
    query_mask = [bool(s) for s in ec_l3_sets]
    n_queries = sum(query_mask)
    print(f"  Queries with EC L3: {n_queries}")

    # For efficiency, pre-compute distances in one batch per query
    # Shape: (n_queries, n_valid_db_reactions)
    query_indices = [i for i, m in enumerate(query_mask) if m]

    per_tau: dict[float, dict] = {t: {"correct": 0, "no_voter": 0} for t in tau_values}
    n_total = len(query_indices)

    print(f"  Running leave-one-out sweep over {n_total} queries × {len(tau_values)} τ values ...")
    for count, qi in enumerate(query_indices, 1):
        if count % 200 == 0 or count == n_total:
            print(f"    {count}/{n_total}")
        query_fp = fps[qi]
        distances = _tanimoto_distances(query_fp, fps)
        true_ec_l3 = ec_l3_sets[qi]  # set of correct EC L3 keys

        for tau in tau_values:
            predicted = _vote_ec_l3(distances, ec_l3_sets, exclude_idx=qi, tau=tau)
            if predicted is None:
                per_tau[tau]["no_voter"] += 1
            elif predicted in true_ec_l3:
                per_tau[tau]["correct"] += 1

    # Aggregate
    tau_results = []
    for tau in tau_values:
        correct = per_tau[tau]["correct"]
        no_voter = per_tau[tau]["no_voter"]
        wrong = n_total - correct - no_voter
        covered = n_total - no_voter
        cond_acc = (correct / covered * 100.0) if covered > 0 else 0.0
        uncond_acc = correct / n_total * 100.0
        coverage = covered / n_total * 100.0
        tau_results.append(
            {
                "tau": tau,
                "n_queries": n_total,
                "n_correct": correct,
                "n_no_voter": no_voter,
                "n_wrong": wrong,
                "conditional_accuracy_pct": round(cond_acc, 2),
                "unconditional_accuracy_pct": round(uncond_acc, 2),
                "coverage_pct": round(coverage, 2),
            }
        )

    return {
        "gem": gem_path.name,
        "fp_type": fp_type,
        "cofactors": cofactors,
        "n_queries": n_total,
        "n_db_reactions": len(valid_idx),
        "alpha": ALPHA,
        "tau_sweep": tau_results,
    }


def make_figure(results_list: list[dict], out_path: Path) -> None:
    """Generate Supplementary Figure S1: EC L3 accuracy vs τ.

    Saves both <out_path>.pdf (OUP submission) and <out_path>.png (draft).
    Font: Helvetica/Arial 8 pt minimum per OUP production requirements.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # OUP figure font requirements: Helvetica/Arial >= 8 pt
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    })

    # Accuracy and coverage are both percentages on the same 0-100 scale, so
    # they share one axis; a twinx() here would imply two different scales.
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax2 = ax1

    colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    cov_colours = ["#aec7e8", "#ffbb78", "#98df8a", "#ff9896"]

    for idx, res in enumerate(results_list):
        label = f"{res['fp_type'].upper()} ({res['cofactors']})"
        taus = [r["tau"] for r in res["tau_sweep"]]
        cond = [r["conditional_accuracy_pct"] for r in res["tau_sweep"]]
        cov = [r["coverage_pct"] for r in res["tau_sweep"]]

        ax1.plot(taus, cond, color=colours[idx % len(colours)],
                 linewidth=2, label=f"{label} EC L3 acc. (conditional)")
        ax2.plot(taus, cov, color=cov_colours[idx % len(cov_colours)],
                 linewidth=1.5, linestyle="--",
                 label=f"{label} coverage")

    ax1.axvline(x=0.5, color="black", linestyle=":", linewidth=1.5,
                label="Default τ = 0.5")
    ax1.set_xlabel("Tanimoto distance threshold τ", fontsize=11)
    ax1.set_ylabel("Percent (%)", fontsize=11)
    ax1.set_xlim(0.0, 0.95)
    ax1.set_ylim(0, 105)
    ax1.set_xticks(TAU_VALUES[::2])

    # Combined legend
    ax1.legend(loc="lower right", fontsize=8, framealpha=0.85)

    fig.tight_layout()
    # Save PDF for OUP submission and PNG for draft use
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure saved: {pdf_path} (submission) + {out_path} (draft)")


def main() -> None:
    parser = argparse.ArgumentParser(description="EC L3 accuracy vs τ sensitivity")
    parser.add_argument("--fp-type", choices=["drfp", "qrfp"], default="drfp")
    parser.add_argument("--cofactors", choices=["keep", "filter"], default="keep")
    parser.add_argument("--all", action="store_true",
                        help="Run DRFP-kept and QRFP-kept in sequence")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    conditions = (
        [("drfp", "keep"), ("qrfp", "keep")]
        if args.all
        else [(args.fp_type, args.cofactors)]
    )

    all_results = []
    for fp_type, cofactors in conditions:
        res = run_sensitivity(GEM_PATH, fp_type, cofactors)
        all_results.append(res)
        out_json = DATA_DIR / f"ec_tau_sensitivity_{fp_type}_{cofactors}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"  Saved: {out_json}")

        # Print summary table
        print(f"\n  τ-sensitivity summary ({fp_type.upper()}, cofactors {cofactors}):")
        print(f"  {'τ':>5}  {'Cond. EC L3 acc (%)':>20}  {'Uncond. acc (%)':>17}  {'Coverage (%)':>13}")
        for row in res["tau_sweep"]:
            print(
                f"  {row['tau']:>5.2f}  {row['conditional_accuracy_pct']:>20.1f}"
                f"  {row['unconditional_accuracy_pct']:>17.1f}  {row['coverage_pct']:>13.1f}"
            )

    # Generate figure
    fig_path = DATA_DIR / "ec_tau_sensitivity_figure.png"
    make_figure(all_results, fig_path)

    # Print key values at default τ=0.5 for manuscript text
    print("\n  === Key values at default τ = 0.50 ===")
    for res in all_results:
        row_50 = next(r for r in res["tau_sweep"] if abs(r["tau"] - 0.5) < 0.01)
        print(
            f"  {res['fp_type'].upper()} {res['cofactors']}: "
            f"cond_acc={row_50['conditional_accuracy_pct']:.1f}%, "
            f"coverage={row_50['coverage_pct']:.1f}%"
        )


if __name__ == "__main__":
    main()
