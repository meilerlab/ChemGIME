"""
ChemGIME Structured Output Export (F5)
=======================================
Programmatic JSON and CSV export for downstream pipeline integration.

Resolves the "experimentalist bottleneck": results are machine-readable
and can feed directly into gene-knockout prioritisation or assay design.

Usage::

    from src.core.export import export_results

    export_results(
        df_rank.head(100),
        ec_predictions,
        gem_name="iML1515.xml",
        query_name="gemcitabine.csv",
        output_dir="results/",
        formats=("json", "csv"),
    )
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _ec_predictions_to_dict(ec_predictions: dict) -> dict:
    """Convert EC prediction results to a JSON-serialisable dict."""
    result = {}
    for level, (pred, conf, df) in ec_predictions.items():
        result[level] = {
            "predicted": pred,
            "confidence": round(conf, 4),
            "top_votes": df.to_dict(orient="records") if isinstance(df, pd.DataFrame) else [],
        }
    return result


def _sanitise_df_for_export(df: pd.DataFrame) -> pd.DataFrame:
    """Select and rename columns for clean export."""
    col_map = {
        "reaction_id": "reaction_id",
        "name": "reaction_name",
        "gpr": "gene_protein_rule",
        "ec": "ec_number",
        "subsystem": "subsystem",
        "tan_distance": "tanimoto_distance",
    }
    available = {k: v for k, v in col_map.items() if k in df.columns}
    out = df[list(available.keys())].copy()
    out.columns = [available[c] for c in out.columns]
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


def export_results(
    df_rank: pd.DataFrame,
    ec_predictions: dict,
    *,
    gem_name: str = "",
    query_name: str = "",
    output_dir: str | Path = ".",
    formats: tuple = ("json", "csv"),
    prefix: Optional[str] = None,
) -> dict:
    """Export analysis results to JSON and/or CSV.

    Parameters
    ----------
    df_rank : pd.DataFrame
        Ranked reactions (output of get_closest_rxns, already sliced to top-N).
    ec_predictions : dict
        EC prediction results from predict_ec_weighted.
    gem_name : str
        Name of the GEM file (for metadata).
    query_name : str
        Name of the query file (for metadata).
    output_dir : path
        Directory to write files.
    formats : tuple
        Output formats: "json", "csv", or both.
    prefix : str, optional
        Filename prefix. Defaults to ``{gem_stem}_{query_stem}``.

    Returns
    -------
    dict
        Mapping of format → output file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gem_stem = Path(gem_name).stem if gem_name else "unknown_gem"
    query_stem = Path(query_name).stem if query_name else "unknown_query"
    prefix = prefix or f"{gem_stem}_{query_stem}"

    clean_df = _sanitise_df_for_export(df_rank)
    paths = {}

    if "csv" in formats:
        csv_path = output_dir / f"{prefix}_results.csv"
        clean_df.to_csv(csv_path, index=False)
        paths["csv"] = str(csv_path)
        logger.info(f"CSV exported: {csv_path}")

    if "json" in formats:
        json_path = output_dir / f"{prefix}_results.json"
        payload = {
            "metadata": {
                "tool": "ChemGIME",
                "version": "2.0",
                "timestamp": datetime.now().isoformat(),
                "gem_file": gem_name,
                "query_file": query_name,
                "n_results": len(clean_df),
            },
            "ec_predictions": _ec_predictions_to_dict(ec_predictions),
            "top_reactions": clean_df.to_dict(orient="records"),
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        paths["json"] = str(json_path)
        logger.info(f"JSON exported: {json_path}")

    return paths


def export_batch_summary(
    all_results: list,
    output_path: str | Path,
) -> Path:
    """Export a batch summary CSV: one row per (GEM, query) pair.

    Parameters
    ----------
    all_results : list of dict
        Output of run_batch().
    output_path : path
        Where to write the summary CSV.

    Returns
    -------
    Path
        The written file path.
    """
    rows = []
    for r in all_results:
        ec_preds = r.get("ec_predictions", {})
        l3_pred, l3_conf = "N/A", 0.0
        if "L3" in ec_preds:
            l3_pred = ec_preds["L3"][0]
            l3_conf = ec_preds["L3"][1]

        top_df = r.get("top_reactions")
        best_dist = top_df.iloc[0]["tan_distance"] if top_df is not None and not top_df.empty else None

        rows.append({
            "gem": r.get("gem", ""),
            "query": r.get("query", ""),
            "query_id": r.get("query_id", ""),
            "n_reactions": r.get("n_reactions", 0),
            "best_tanimoto_distance": best_dist,
            "ec_l3_prediction": l3_pred,
            "ec_l3_confidence": round(l3_conf, 4),
            "gem_build_time_s": r.get("timing", {}).get("gem_build_s"),
            "query_time_s": r.get("timing", {}).get("query_s"),
        })

    output_path = Path(output_path)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    logger.info(f"Batch summary exported: {output_path}")
    return output_path
