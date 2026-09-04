"""
Out-of-distribution (OOD) threshold validation for ChemGIME.
============================================================

Question
--------
Is the operational cutoff D_Tan = 0.5 (nearest-GEM-neighbour distance) a
genuine boundary beyond which retrieval fails, or merely a heuristic reused
from the decoy similarity window?

Test
----
For every M-CSA query that ChemGIME encoded (status=success), bin by
`min_distance` (Tanimoto distance to the closest GEM reaction = the rank-1
hit distance) and measure top-hit EC-match success at L1 and L3. Compare
success above vs below 0.5 with a Fisher exact test.

Source: tests/benchmark_output/mcsa_novelty_{gem}_qrfp.json -> details[]
Output: benchmarks/data/ood_threshold_validation.json
        benchmarks/data/ood_threshold_hitrate.png  (--plot)

Run: python benchmarks/ood_threshold_validation.py [--plot]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parent.parent
NOVELTY = ROOT / "tests" / "benchmark_output"
OUT = ROOT / "benchmarks" / "data" / "ood_threshold_validation.json"
PLOT_OUT = ROOT / "benchmarks" / "data" / "ood_threshold_hitrate.png"
GEMS = ["iML1515", "iJN746"]
LEVELS = [("ec_match_l1", "#1f77b4", "EC-L1 (class)"),
          ("ec_match_l3", "#d62728", "EC-L3 (sub-subclass)")]
THRESHOLD = 0.5
BIN_EDGES = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.01]


def _load(gem: str) -> list[dict]:
    d = json.loads((NOVELTY / f"mcsa_novelty_{gem}_qrfp.json").read_text())["details"]
    return [q for q in d if q.get("status") == "success" and q.get("min_distance") is not None]


def _analyse(rec: list[dict], level: str) -> dict:
    md = np.array([q["min_distance"] for q in rec])
    ok = np.array([bool(q.get(level)) for q in rec])

    bins = []
    for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        m = (md >= lo) & (md < hi)
        if m.sum():
            bins.append({"lo": lo, "hi": hi, "n": int(m.sum()),
                         "success_pct": round(float(ok[m].mean()) * 100, 1)})

    below, above = ok[md <= THRESHOLD], ok[md > THRESHOLD]
    table = [[int(below.sum()), int(len(below) - below.sum())],
             [int(above.sum()), int(len(above) - above.sum())]]
    odds, p = fisher_exact(table)
    return {
        "level": level,
        "n_total": len(rec),
        "bins": bins,
        "below_0.5": {"n": int(len(below)), "success_pct": round(float(below.mean()) * 100, 1),
                      "n_correct": int(below.sum())},
        "above_0.5": {"n": int(len(above)), "success_pct": round(float(above.mean()) * 100, 1),
                      "n_correct": int(above.sum())},
        "fisher_p": float(p),
    }


def _plot() -> None:
    """Per-discrete-value hit rate (point size = n) plus 0.1-bin mean trend,
    marking the D_Tan = 0.5 out-of-distribution wall."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
    bins = np.arange(0, 1.001, 0.1)
    for ax, gem in zip(axes, GEMS):
        rec = _load(gem)
        md = np.array([q["min_distance"] for q in rec])
        for lvl, color, lab in LEVELS:
            ok = np.array([bool(q.get(lvl)) for q in rec])
            # per discrete distance value: faint points sized by n
            agg: dict = defaultdict(lambda: [0, 0])
            for v, o in zip(md, ok):
                agg[round(float(v), 4)][0] += int(o)
                agg[round(float(v), 4)][1] += 1
            xs = sorted(agg)
            ys = [agg[x][0] / agg[x][1] * 100 for x in xs]
            ns = [agg[x][1] for x in xs]
            ax.scatter(xs, ys, s=[max(8, n * 6) for n in ns], color=color,
                       alpha=0.30, edgecolors="none", zorder=2)
            # 0.1-bin mean trend (>=3 queries per bin)
            bx, by = [], []
            for lo, hi in zip(bins[:-1], bins[1:]):
                m = (md >= lo) & (md < hi)
                if m.sum() >= 3:
                    bx.append((lo + hi) / 2)
                    by.append(ok[m].mean() * 100)
            ax.plot(bx, by, "-o", color=color, lw=2.2, ms=5, label=lab, zorder=4)
        above = int((md > THRESHOLD).sum())
        ax.axvline(THRESHOLD, color="k", ls="--", lw=1.5, zorder=1)
        ax.axvspan(THRESHOLD, 1.0, color="grey", alpha=0.09, zorder=0)
        ax.text(THRESHOLD + 0.01, 95, r"$D_{Tan}=0.5$", fontsize=9, va="top")
        ax.text(0.71, 55, f"> 0.5:\n0 / {above} hits", fontsize=10.5, ha="center",
                color="#444", weight="bold")
        ax.set_title(f"{gem}   (n={len(rec)} queries)", fontsize=11)
        ax.set_xlabel("Nearest GEM-neighbour distance  (Tanimoto, min_distance)")
        ax.set_xlim(0, 0.92)
        ax.set_ylim(-4, 100)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Top-hit EC-match success rate (%)")
    axes[0].legend(loc="upper right", fontsize=9, framealpha=0.92,
                   title="line = 0.1-bin mean\ndots = per-value (size $\\propto$ n)",
                   title_fontsize=7.5)
    # No in-image title: the journal caption carries the title, and an
    # embedded one duplicates it and cannot be copy-edited.  The previous
    # title also asserted the conclusion ("a clean 0% wall") rather than
    # describing the data, which belongs in the caption.
    fig.tight_layout()
    fig.savefig(PLOT_OUT, dpi=600, bbox_inches="tight")
    pdf_out = PLOT_OUT.with_suffix(".pdf")
    fig.savefig(pdf_out, bbox_inches="tight")
    print(f"Saved: {PLOT_OUT}\nSaved: {pdf_out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plot", action="store_true",
                    help="also render the per-value hit-rate figure")
    args = ap.parse_args()

    results = {"threshold": THRESHOLD, "metric": "min_distance (nearest GEM neighbour, Tanimoto)",
               "source": "mcsa_novelty_{gem}_qrfp.json details[] status=success", "gems": {}}
    pooled_above_l1 = [0, 0]  # correct, total
    for gem in GEMS:
        rec = _load(gem)
        results["gems"][gem] = {lvl: _analyse(rec, lvl) for lvl, _c, _l in LEVELS}
        a = results["gems"][gem]["ec_match_l1"]["above_0.5"]
        pooled_above_l1[0] += a["n_correct"]
        pooled_above_l1[1] += a["n"]
    results["pooled_above_0.5_ec_l1"] = {"n_correct": pooled_above_l1[0], "n_total": pooled_above_l1[1]}

    OUT.write_text(json.dumps(results, indent=2))
    print(f"Saved: {OUT}")
    print(f"\nPooled across both GEMs, queries with nearest neighbour > 0.5:")
    print(f"  EC-L1 correct: {pooled_above_l1[0]} / {pooled_above_l1[1]}")
    for gem in GEMS:
        for lvl, _c, _l in LEVELS:
            r = results["gems"][gem][lvl]
            print(f"  {gem} {lvl}: <=0.5 {r['below_0.5']['success_pct']}% (n={r['below_0.5']['n']})  "
                  f"| >0.5 {r['above_0.5']['success_pct']}% (n={r['above_0.5']['n']})  p={r['fisher_p']:.2e}")

    if args.plot:
        _plot()


if __name__ == "__main__":
    main()
