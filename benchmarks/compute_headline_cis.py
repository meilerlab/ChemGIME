"""
compute_headline_cis.py — Bootstrap 95% CIs for Split-GEM (Table 1) and
M-CSA (Table 2) headline benchmarks.

Reads per-query rank arrays from tests/benchmark_output/*.json and computes
mean MRR, Top-1, Top-5, Top-10 with bootstrap 95% percentile CIs.

Usage:
    python benchmarks/compute_headline_cis.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = PROJECT_ROOT / "tests" / "benchmark_output"
N_BOOT = 10_000
SEED = 42
RNG = np.random.default_rng(SEED)


def _bootstrap_ci(values: np.ndarray, stat_fn, n_boot: int = N_BOOT):
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    point = stat_fn(values)
    boot = np.empty(n_boot)
    idx = RNG.integers(0, n, size=(n_boot, n))
    for i in range(n_boot):
        boot[i] = stat_fn(values[idx[i]])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, lo, hi


def _rr(ranks: np.ndarray) -> np.ndarray:
    rr = np.where(ranks > 0, 1.0 / np.maximum(ranks, 1), 0.0)
    return rr


def _topk(ranks: np.ndarray, k: int) -> np.ndarray:
    return ((ranks >= 1) & (ranks <= k)).astype(float) * 100.0


def metrics_with_ci(ranks: np.ndarray) -> dict:
    rr = _rr(ranks)
    mrr, mrr_lo, mrr_hi = _bootstrap_ci(rr, np.mean)
    out = {"n": int(len(ranks)), "mrr": (mrr, mrr_lo, mrr_hi)}
    for k in (1, 5, 10, 100):
        v = _topk(ranks, k)
        m, lo, hi = _bootstrap_ci(v, np.mean)
        out[f"top{k}"] = (m, lo, hi)
    return out


def _fmt(t):
    m, lo, hi = t
    if abs(m) < 1.0 and 0 <= m <= 1:
        return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"
    return f"{m:.1f} [{lo:.1f}, {hi:.1f}]"


def split_gem(condition_file: Path, rank_field: str = "rank"):
    data = json.load(open(condition_file))
    ranks = np.array(
        [d.get(rank_field, 0) if d.get(rank_field) is not None else 0
         for d in data["details"]],
        dtype=int,
    )
    return metrics_with_ci(ranks)


def mcsa(condition_file: Path):
    """Match manuscript denominator: exclude all_cofactors queries
    (they have no GEM-side fingerprint signal under cofactor-filtered)."""
    data = json.load(open(condition_file))
    ranks = []
    for d in data["details"]:
        if d.get("status") == "all_cofactors":
            continue  # excluded from denominator (n_valid)
        if not d.get("found", False):
            ranks.append(0)
        else:
            r = d.get("rank")
            ranks.append(int(r) if r else 0)
    return metrics_with_ci(np.array(ranks, dtype=int))


def main() -> None:
    out = {"split_gem": {}, "split_gem_strict": {}, "mcsa": {}}

    # ── Split-GEM, both criteria
    sg_files = {
        ("iML1515", "DRFP", "filtered"): "split_gem_iML1515_drfp_filtered.json",
        ("iML1515", "DRFP", "kept"):     "split_gem_iML1515_drfp_kept.json",
        ("iML1515", "QRFP", "filtered"): "split_gem_iML1515_qrfp_filtered.json",
        ("iML1515", "QRFP", "kept"):     "split_gem_iML1515_qrfp_kept.json",
        ("iJN746",  "DRFP", "filtered"): "split_gem_iJN746_drfp_filtered.json",
        ("iJN746",  "DRFP", "kept"):     "split_gem_iJN746_drfp_kept.json",
        ("iJN746",  "QRFP", "filtered"): "split_gem_iJN746_qrfp_filtered.json",
        ("iJN746",  "QRFP", "kept"):     "split_gem_iJN746_qrfp_kept.json",
    }
    for cond, fname in sg_files.items():
        path = BENCH_DIR / fname
        if not path.exists():
            print(f"  [MISS] {fname}")
            continue
        out["split_gem"]["__".join(cond)] = split_gem(path, "rank")
        out["split_gem_strict"]["__".join(cond)] = split_gem(path, "strict_rank")

    # ── M-CSA
    mcsa_files = {
        ("iML1515", "DRFP", "filtered"): "mcsa_novelty_iML1515_drfp_filtered.json",
        ("iML1515", "DRFP", "kept"):     "mcsa_novelty_iML1515_drfp_kept.json",
        ("iML1515", "QRFP", "filtered"): "mcsa_novelty_iML1515_qrfp_filtered.json",
        ("iML1515", "QRFP", "kept"):     "mcsa_novelty_iML1515_qrfp_kept.json",
    }
    for cond, fname in mcsa_files.items():
        path = BENCH_DIR / fname
        if not path.exists():
            print(f"  [MISS] {fname}")
            continue
        out["mcsa"]["__".join(cond)] = mcsa(path)

    # Pretty-print summary
    def section(title, data):
        print(f"\n=== {title} (bootstrap 95% CI; n_boot={N_BOOT}, seed={SEED}) ===")
        header = f"{'condition':<32}{'n':>5}  {'MRR':<24}{'Top-1':<24}{'Top-5':<24}{'Top-10':<24}"
        print(header)
        print("-" * len(header))
        for k, m in data.items():
            row = f"{k:<32}{m['n']:>5}  "
            for fld in ("mrr", "top1", "top5", "top10"):
                row += f"{_fmt(m[fld]):<24}"
            print(row)

    section("Split-GEM (permissive)", out["split_gem"])
    section("Split-GEM (strict)", out["split_gem_strict"])
    section("M-CSA", out["mcsa"])

    # Serialise raw numbers for downstream use
    json_out = {
        section: {
            cond: {fld: list(vals) if isinstance(vals, tuple) else vals
                   for fld, vals in metrics.items()}
            for cond, metrics in section_data.items()
        }
        for section, section_data in out.items()
    }
    out_path = PROJECT_ROOT / "benchmarks" / "data" / "headline_cis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(json_out, open(out_path, "w"), indent=2)
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
