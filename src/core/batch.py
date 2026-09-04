"""
ChemGIME Batch Processing (F4)
================================
Multi-GEM, multi-query batch analysis without Streamlit.

Leverages the decoupled parser architecture: any combination of GEM files
(SBML/XML) can be screened against any set of query reactions (CSV/TSV).

Usage::

    from src.core.batch import run_batch

    results = run_batch(
        gem_paths=["organism_A.xml", "organism_B.xml"],
        query_paths=["drug_1.csv", "drug_2.csv"],
    )

CLI::

    python -m src.core.batch \\
        --gems model_A.xml model_B.xml \\
        --queries query_1.csv query_2.csv \\
        --output results/ \\
        --format json csv html
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class _FileWrapper:
    """File-like wrapper for build_organism_database."""
    def __init__(self, path: Path):
        self.name = path.name
        with open(path, "rb") as f:
            self._buf = BytesIO(f.read())

    def seek(self, pos):
        self._buf.seek(pos)

    def read(self):
        return self._buf.read()


def run_batch(
    gem_paths: List[str | Path],
    query_paths: List[str | Path],
    *,
    num_results: int = 100,
    ec_threshold: float = 0.5,
    output_dir: Optional[str | Path] = None,
    output_formats: tuple = ("json",),
) -> List[Dict[str, Any]]:
    """Run ChemGIME analysis for all (GEM, query) pairs.

    Parameters
    ----------
    gem_paths : list of paths
        One or more GEM files (SBML/XML).
    query_paths : list of paths
        One or more query reaction files (CSV/TSV with left_smiles, right_smiles).
    num_results : int
        Number of top results to retain per query.
    ec_threshold : float
        Tanimoto distance threshold for EC prediction.
    output_dir : path, optional
        Directory to write results. Created if needed.
    output_formats : tuple
        Output formats: any combination of "json", "csv", "html".

    Returns
    -------
    list of dict
        One entry per (GEM, query) pair with keys:
        ``gem``, ``query``, ``top_reactions``, ``ec_predictions``, ``timing``.
    """
    from .config import BIGG_MET_TO_SMILES_FILE, COFACTOR_SMILES_FILE
    from .data_loading import load_bigg_to_smiles_map, load_cofactor_set
    from .ec_prediction import predict_ec_weighted
    from .fingerprints import _calculate_drfp_fingerprint, calculate_fingerprints_parallel
    from .gem_parser import build_organism_database, filter_smiles_string
    from .reporting import generate_html_report
    from .similarity import get_closest_rxns
    from .export import export_results

    met_to_smiles = load_bigg_to_smiles_map(BIGG_MET_TO_SMILES_FILE)
    cofactor_set = load_cofactor_set(COFACTOR_SMILES_FILE)

    all_results = []

    for gem_path in gem_paths:
        gem_path = Path(gem_path)
        logger.info(f"Processing GEM: {gem_path.name}")
        t0 = time.time()

        gem_wrapper = _FileWrapper(gem_path)
        db_df = build_organism_database(gem_wrapper, met_to_smiles, cofactor_set)
        if db_df is None or db_df.empty:
            logger.warning(f"No reactions in {gem_path.name}, skipping.")
            continue

        db_fps, valid_indices = calculate_fingerprints_parallel(db_df)
        if db_fps is None or len(db_fps) == 0:
            logger.warning(f"No fingerprints for {gem_path.name}, skipping.")
            continue

        gem_build_time = time.time() - t0
        logger.info(f"  Database built: {len(db_df)} rxns, {len(db_fps)} FPs ({gem_build_time:.1f}s)")

        for query_path in query_paths:
            query_path = Path(query_path)
            t1 = time.time()
            logger.info(f"  Query: {query_path.name}")

            try:
                from .data_loading import load_data
                query_df = load_data(query_path).rename(columns={"reaction": "id"}).iloc[0:1]
                if query_df is None or query_df.empty:
                    logger.warning(f"  Empty query file: {query_path.name}")
                    continue

                query_row = query_df.iloc[0]
                qfp = _calculate_drfp_fingerprint((
                    query_row.get("left_smiles"),
                    query_row.get("right_smiles"),
                ))
                if qfp is None:
                    logger.warning(f"  Fingerprint failed for {query_path.name}")
                    continue

                df_rank = get_closest_rxns(
                    qfp.reshape(1, -1), db_fps, db_df, valid_indices
                )
                ec_preds = predict_ec_weighted(df_rank, ec_threshold)

                result = {
                    "gem": gem_path.name,
                    "query": query_path.name,
                    "query_id": query_row.get("id", query_path.stem),
                    "n_reactions": len(db_df),
                    "top_reactions": df_rank.head(num_results),
                    "ec_predictions": ec_preds,
                    "timing": {
                        "gem_build_s": round(gem_build_time, 2),
                        "query_s": round(time.time() - t1, 2),
                    },
                }
                all_results.append(result)

                # Write outputs
                if output_dir:
                    out = Path(output_dir)
                    out.mkdir(parents=True, exist_ok=True)
                    stem = f"{gem_path.stem}_{query_path.stem}"

                    if "json" in output_formats or "csv" in output_formats:
                        export_results(
                            df_rank.head(num_results),
                            ec_preds,
                            gem_name=gem_path.name,
                            query_name=query_path.name,
                            output_dir=out,
                            formats=output_formats,
                            prefix=stem,
                        )

                    if "html" in output_formats:
                        html = generate_html_report(
                            df_rank, query_df, gem_path.name, query_path.name,
                            num_results, ec_preds, cofactor_set,
                        )
                        (out / f"report_{stem}.html").write_text(html, encoding="utf-8")

            except Exception as e:
                logger.error(f"  Error processing {query_path.name}: {e}", exc_info=True)
                continue

    logger.info(f"Batch complete: {len(all_results)} results from {len(gem_paths)} GEMs x {len(query_paths)} queries.")
    return all_results


# ── CLI entry point ──────────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(
        prog="python -m src.core.batch",
        description="ChemGIME batch processing: multi-GEM x multi-query analysis",
    )
    parser.add_argument("--gems", nargs="+", required=True, help="GEM file(s) (.xml/.sbml)")
    parser.add_argument("--queries", nargs="+", required=True, help="Query file(s) (.csv/.tsv)")
    parser.add_argument("--output", default="batch_results", help="Output directory")
    parser.add_argument("--format", nargs="+", default=["json", "csv"],
                        choices=["json", "csv", "html"], help="Output formats")
    parser.add_argument("--num-results", type=int, default=100, help="Top-N results per query")
    parser.add_argument("--ec-threshold", type=float, default=0.5, help="EC prediction distance threshold")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    results = run_batch(
        gem_paths=args.gems,
        query_paths=args.queries,
        num_results=args.num_results,
        ec_threshold=args.ec_threshold,
        output_dir=args.output,
        output_formats=tuple(args.format),
    )

    print(f"\nBatch complete: {len(results)} analyses written to {args.output}/")


if __name__ == "__main__":
    _cli()
