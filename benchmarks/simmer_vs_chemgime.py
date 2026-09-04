#!/usr/bin/env python3
"""
simmer_vs_chemgime.py
=====================
Sub-sampling benchmark: SIMMER positive controls × ChemGIME-GEM.

For each of the 33 positive-control drug-metabolism reactions from
Bustion et al. (eLife 2023), restrict the search space to a **single
organism GEM** and measure where the correct enzyme lands in ChemGIME's
ranked output.  Compare against SIMMER's published 88 % Top-20 accuracy
measured across 286 997 genomes × 8 914 MetaCyc reactions.

Usage
-----
    python benchmarks/simmer_vs_chemgime.py \
        --config benchmarks/simmer_benchmark_config.yaml \
        [--gem-dir /path/to/agora2/] \
        [--output-dir benchmarks/simmer_comparison/] \
        [--skip-missing]           # skip queries whose GEM is absent

Requires
--------
- ChemGIME src/ on PYTHONPATH (added automatically below)
- AGORA2 (or BiGG) SBML files for the 7 organisms
- drfp, rdkit, cobra, scikit-learn, pandas, pyyaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

# ── Ensure ChemGIME src/ is importable ──────────────────────────────────
_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
sys.path.insert(0, str(_PROJECT / "src"))
sys.path.insert(0, str(_PROJECT))

from core.config import BIGG_MET_TO_SMILES_FILE, COFACTOR_SMILES_FILE
from core.data_loading import load_bigg_to_smiles_map, load_cofactor_set
from core.ec_prediction import predict_ec_weighted
from core.fingerprints import (
    _calculate_drfp_fingerprint,
    calculate_fingerprints_parallel,
)
from core.gem_parser import build_organism_database
from core.metrics import reciprocal_rank, retrieval_summary, top_k_accuracy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("benchmark")


# ── Data structures ─────────────────────────────────────────────────────


@dataclass
class QueryResult:
    query_id: str
    dm: str
    drug: str
    product: str
    organism: str
    enzyme: str
    ec: str

    # ChemGIME outputs
    cg_rank: Optional[int] = None
    cg_distance: Optional[float] = None
    cg_ec_l1: Optional[str] = None
    cg_ec_l3: Optional[str] = None
    cg_gem_size: Optional[int] = None  # reactions in the GEM
    cg_time_s: Optional[float] = None

    # SIMMER reference (from paper Figure 5)
    simmer_top20_hit: bool = True  # default True (88 % overall)

    notes: str = ""


# ── Helpers ─────────────────────────────────────────────────────────────


class _FileWrapper:
    """Adapts a Path to the file-like interface build_organism_database expects."""

    def __init__(self, path: Path):
        self._path = path
        self.name = path.name
        self._bytes = path.read_bytes()
        self._pos = 0

    def seek(self, pos: int) -> None:
        self._pos = pos

    def read(self) -> bytes:
        data = self._bytes[self._pos :]
        self._pos = len(self._bytes)
        return data


def _find_enzyme_rank(
    ranked_df: pd.DataFrame, enzyme_id: str, ec_code: str
) -> Optional[int]:
    """Return 1-based rank of the first row whose GPR or EC matches the target.

    Matching strategy (mirrors what SIMMER reports):
      1. GPR string contains the enzyme gene locus (e.g. 'BT_4554', 'b1534')
      2. EC number at L3 resolution matches
    Either is sufficient for a hit.
    """
    enzyme_lower = enzyme_id.lower()
    ec_prefix = ".".join(ec_code.split(".")[:3]) if ec_code else ""

    for rank_0, row in enumerate(ranked_df.itertuples()):
        gpr = str(getattr(row, "gpr", "")).lower()
        rxn_ec = str(getattr(row, "ec", ""))
        # Match by GPR gene name
        if enzyme_lower and enzyme_lower in gpr:
            return rank_0 + 1
        # Match by EC sub-sub-class
        if ec_prefix and rxn_ec.startswith(ec_prefix):
            return rank_0 + 1
    return None


# ── Main pipeline ───────────────────────────────────────────────────────


def run_benchmark(config_path: str, gem_dir_override: str | None, output_dir: str, skip_missing: bool):
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    gem_base = Path(gem_dir_override) if gem_dir_override else Path(cfg.get("gem_dir", "."))
    gems_cfg = cfg["gems"]
    queries = cfg["queries"]
    topk = cfg.get("benchmark", {}).get("top_k_cutoffs", [1, 5, 10, 20])
    ec_thresh = cfg.get("benchmark", {}).get("ec_threshold", 0.5)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load shared reference data ──
    log.info("Loading metabolite-to-SMILES map and cofactor set ...")
    met_map = load_bigg_to_smiles_map(BIGG_MET_TO_SMILES_FILE)
    cofactors = load_cofactor_set(COFACTOR_SMILES_FILE)

    # ── Pre-load GEMs (caching per organism) ──
    gem_cache: dict[str, tuple[pd.DataFrame, np.ndarray, list[int]]] = {}

    def _load_gem(gem_alias: str):
        if gem_alias in gem_cache:
            return gem_cache[gem_alias]
        info = gems_cfg[gem_alias]
        sbml_path = gem_base / info["file"]
        if not sbml_path.exists():
            log.warning("GEM not found: %s", sbml_path)
            return None
        log.info("Parsing GEM: %s  (%s)", gem_alias, sbml_path.name)
        wrapper = _FileWrapper(sbml_path)
        db_df = build_organism_database(wrapper, met_map, cofactors)
        if db_df is None or db_df.empty:
            log.warning("Empty database for %s", gem_alias)
            return None
        fps, valid = calculate_fingerprints_parallel(db_df)
        if fps is None:
            log.warning("Fingerprinting failed for %s", gem_alias)
            return None
        gem_cache[gem_alias] = (db_df, fps, valid)
        log.info("  -> %d reactions, %d valid fingerprints", len(db_df), len(valid))
        return gem_cache[gem_alias]

    # ── Run each query ──
    results: list[QueryResult] = []

    for q in queries:
        qid = q["id"]
        gem_alias = q["gem"]
        organism = gems_cfg[gem_alias]["organism"]

        log.info("── %s  (%s, GEM=%s) ──", qid, q["drug"], gem_alias)
        res = QueryResult(
            query_id=qid,
            dm=q["dm"],
            drug=q["drug"],
            product=q["product"],
            organism=organism,
            enzyme=q["enzyme"],
            ec=q["ec"],
        )

        gem_data = _load_gem(gem_alias)
        if gem_data is None:
            if skip_missing:
                res.notes = "GEM not found; skipped"
                results.append(res)
                continue
            else:
                log.error("GEM missing and --skip-missing not set. Aborting.")
                sys.exit(1)

        db_df, db_fps, valid_idx = gem_data
        res.cg_gem_size = len(db_df)

        # Fingerprint the query reaction
        left = q["left_smiles"]
        right = q["right_smiles"]
        t0 = time.monotonic()
        qfp = _calculate_drfp_fingerprint((left, right))
        if qfp is None:
            res.notes = "Query fingerprint failed"
            results.append(res)
            continue

        # Rank by Jaccard distance
        from core.similarity import get_closest_rxns

        ranked = get_closest_rxns(qfp.reshape(1, -1), db_fps, db_df, valid_idx)
        elapsed = time.monotonic() - t0
        res.cg_time_s = round(elapsed, 3)

        if ranked.empty:
            res.notes = "No ranked results"
            results.append(res)
            continue

        res.cg_distance = float(ranked.iloc[0]["tan_distance"])

        # EC prediction
        ec_pred = predict_ec_weighted(ranked, distance_threshold=ec_thresh)
        if ec_pred:
            res.cg_ec_l1 = ec_pred.get("L1", (None,))[0]
            res.cg_ec_l3 = ec_pred.get("L3", (None,))[0]

        # Find rank of the positive-control enzyme
        rank = _find_enzyme_rank(ranked, q["enzyme"], q["ec"])
        res.cg_rank = rank

        log.info(
            "  rank=%s  distance=%.4f  ec_l3=%s  (%.2fs)",
            rank if rank else "NOT FOUND",
            res.cg_distance,
            res.cg_ec_l3 or "?",
            elapsed,
        )
        results.append(res)

    # ── Compute summary metrics ──
    _print_summary(results, topk)
    _save_outputs(results, out)


# ── Reporting ───────────────────────────────────────────────────────────


def _print_summary(results: list[QueryResult], topk: list[int]):
    ranks = [r.cg_rank for r in results]
    n = len(results)
    n_found = sum(1 for r in ranks if r is not None)
    n_skipped = sum(1 for r in results if "skipped" in r.notes)

    print()
    print("=" * 72)
    print("  SIMMER vs. ChemGIME-GEM  —  Sub-sampling Benchmark Results")
    print("=" * 72)
    print(f"  Total positive controls:    {n}")
    print(f"  GEMs skipped (missing):     {n_skipped}")
    print(f"  Enzyme found in output:     {n_found} / {n - n_skipped}")
    print()

    # SIMMER baseline
    simmer_top20 = sum(1 for r in results if r.simmer_top20_hit and "skipped" not in r.notes)
    active = n - n_skipped
    print("  SIMMER (published, Top-20 / 8914 rxns / 287k genomes)")
    print(f"    Accuracy:  {simmer_top20}/{active}  =  {simmer_top20/max(active,1):.0%}")
    print()

    # ChemGIME at each cutoff
    print("  ChemGIME-GEM (organism-restricted)")
    for k in topk:
        hits = sum(1 for r in ranks if r is not None and r <= k)
        print(f"    Top-{k:>2d}:    {hits}/{active}  =  {hits/max(active,1):.0%}")

    # MRR
    valid_ranks = [r for r in ranks if r is not None]
    if valid_ranks:
        mrr = np.mean([1.0 / r for r in valid_ranks])
        print(f"    MRR:       {mrr:.4f}")

    # Highlight focus cases
    focus_ids = {"PC18_quercetin3glc", "PC20_ginsenoside_rb1", "PC07_brivudine", "PC08_sorivudine"}
    focus = [r for r in results if r.query_id in focus_ids]
    if focus:
        print()
        print("  --- Focus Cases ---")
        print(f"  {'ID':<25s}  {'Drug':<22s}  {'CG Rank':<10s}  {'CG Dist':<10s}  {'EC L3':<12s}")
        for r in focus:
            rk = str(r.cg_rank) if r.cg_rank else "N/F"
            dist = f"{r.cg_distance:.4f}" if r.cg_distance is not None else "—"
            ec3 = r.cg_ec_l3 or "—"
            print(f"  {r.query_id:<25s}  {r.drug:<22s}  {rk:<10s}  {dist:<10s}  {ec3:<12s}")

    print("=" * 72)


def _save_outputs(results: list[QueryResult], out_dir: Path):
    # CSV
    rows = [asdict(r) for r in results]
    df = pd.DataFrame(rows)
    csv_path = out_dir / "benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    log.info("CSV saved: %s", csv_path)

    # JSON
    json_path = out_dir / "benchmark_results.json"
    with open(json_path, "w") as fh:
        json.dump(rows, fh, indent=2, default=str)
    log.info("JSON saved: %s", json_path)

    # Per-enzyme-group summary
    groups = df.groupby("enzyme").agg(
        n_queries=("query_id", "count"),
        n_found=("cg_rank", lambda s: s.notna().sum()),
        median_rank=("cg_rank", "median"),
        mean_distance=("cg_distance", "mean"),
    )
    summary_path = out_dir / "per_enzyme_summary.csv"
    groups.to_csv(summary_path)
    log.info("Per-enzyme summary: %s", summary_path)


# ── CLI ─────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description="SIMMER vs. ChemGIME sub-sampling benchmark"
    )
    ap.add_argument(
        "--config",
        default=str(_HERE / "simmer_benchmark_config.yaml"),
        help="Path to benchmark YAML config",
    )
    ap.add_argument(
        "--gem-dir",
        default=None,
        help="Override base directory for GEM SBML files",
    )
    ap.add_argument(
        "--output-dir",
        default=str(_HERE / "simmer_comparison"),
        help="Directory for output CSV/JSON",
    )
    ap.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip queries whose GEM file is absent",
    )
    args = ap.parse_args()
    run_benchmark(args.config, args.gem_dir, args.output_dir, args.skip_missing)


if __name__ == "__main__":
    main()
