"""
MicrobeRX vs ChemGIME — Head-to-Head EC-Attribution Benchmark
=============================================================

Empirical comparison addressing Major Weakness 5 of the peer-review audit:
"Table 5 is excellent as an architectural map; the scoring row juxtaposition
(ChemGIME 10/10 vs MicrobeRX 13/18) reads as a head-to-head metric ...
Run MicrobeRX on the 130 M-CSA queries and report a common enzyme-attribution
metric."

Design
------
Both systems are evaluated on the IDENTICAL 130 M-CSA query set used in
the ChemGIME headline benchmark.  The common metric is EC-attribution
accuracy: does the system predict the correct enzyme class for a reaction?

  ChemGIME  — receives substrate + product SMILES; searches iML1515;
              ranks by Tanimoto distance on DRFP/QRFP fingerprints.
              Ground truth: GPR-overlap with target UniProt.
              Existing results are loaded from the stored benchmark JSON.

  MicrobeRX — receives substrate SMILES ONLY (harder task: product unknown);
              applies 109,403 CSRR rules from AGORA2+Recon3D;
              ranks candidate metabolites by confidence_score =
              similarity_substrates + similarity_products + atom_efficiency.
              Ground truth: does the top-ranked CSRR carry the correct
              EC class?  Matched at EC levels 1, 2, 3.

Metrics (both systems, same 130-query denominator)
---------------------------------------------------
  MRR       — Mean Reciprocal Rank of first correct-EC prediction
  Top-1/5/10 — % of queries with correct-EC prediction in top k
  Coverage  — % of queries where the system returns ≥1 prediction

EC matching level: L3 (sub-subclass) as the primary metric, consistent
with the EC-accuracy column in ChemGIME's Table 2.

Output
------
  benchmarks/data/microberx_mcsa_results.json   — full per-query results
  benchmarks/data/microberx_mcsa_statistics.json — aggregate summary table

Usage
-----
    python benchmarks/microberx_mcsa_eval.py
    python benchmarks/microberx_mcsa_eval.py --ec-level 4
    python benchmarks/microberx_mcsa_eval.py --biosystem gutmicrobes
    python benchmarks/microberx_mcsa_eval.py --confidence-cutoff 0.0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
VENDOR_DIR = PROJECT_ROOT / "vendor" / "MicrobeRX"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(VENDOR_DIR))

# Silence MicrobeRX's verbose INFO logging during import
import logging as _logging
_logging.getLogger().setLevel(_logging.WARNING)

from rdkit import Chem
import rdkit
rdkit.RDLogger.DisableLog("rdApp.*")

DB_DIR = PROJECT_ROOT / "data" / "database"
MCSA_FULL_JSON = DB_DIR / "mcsa_entries_full.json"
MOL_CACHE_DIR = DB_DIR / "mcsa_mol_cache"
DEFAULT_QUERY_SET = (PROJECT_ROOT / "tests" / "benchmark_output" /
                     "mcsa_novelty_iML1515_drfp_kept.json")
DATA_DIR = SCRIPT_DIR / "data"

log = logging.getLogger("microberx_mcsa_eval")


# ════════════════════════════════════════════════════════════════════════
#  Re-use M-CSA query loading from decoy_eval (same 130-query canonical set)
# ════════════════════════════════════════════════════════════════════════

def _chebi_mol_to_smiles(mol_url: str, chebi_id: str) -> Optional[str]:
    MOL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = MOL_CACHE_DIR / f"{chebi_id}.mol"
    if cache_path.exists():
        mol_block = cache_path.read_text(encoding="utf-8")
    else:
        full_url = mol_url if mol_url.startswith("http") else f"https://{mol_url}"
        try:
            req = urllib.request.urlopen(full_url, timeout=15)
            mol_block = req.read().decode("utf-8")
            cache_path.write_text(mol_block, encoding="utf-8")
        except Exception as exc:
            log.debug("mol download failed for ChEBI %s: %s", chebi_id, exc)
            return None
    mol = Chem.MolFromMolBlock(mol_block)
    return Chem.MolToSmiles(mol) if mol is not None else None


def _build_substrate_smiles(entry: dict) -> Optional[str]:
    """Return dot-joined SMILES of reactant compounds (substrate side only)."""
    compounds = entry.get("reaction", {}).get("compounds", [])
    reactants = []
    for comp in compounds:
        chebi_id = str(comp.get("chebi_id", ""))
        mol_url = comp.get("mol_file", "")
        if not chebi_id or not mol_url:
            continue
        if comp.get("type") != "reactant":
            continue
        smi = _chebi_mol_to_smiles(mol_url, chebi_id)
        if smi:
            reactants.append(smi)
    return ".".join(reactants) if reactants else None


def load_queries(query_set_path: Path, mcsa_full_path: Path) -> list[dict]:
    mcsa_full = json.load(open(mcsa_full_path))
    by_id = {e.get("mcsa_id"): e for e in mcsa_full}

    if query_set_path.exists():
        details = json.load(open(query_set_path))["details"]
        queries = []
        for d in details:
            full = by_id.get(d["mcsa_id"])
            if full is None:
                continue
            queries.append({
                "mcsa_id": d["mcsa_id"],
                "enzyme_name": d.get("enzyme_name", ""),
                "target_ec": d["target_ec"],
                "target_uniprot": d.get("target_uniprot", ""),
                "reaction": full.get("reaction", {}),
            })
        return queries

    # Fallback: derive from GEM UniProt overlap
    g2u_path = DB_DIR / "iML1515_gene_to_uniprot.json"
    g2u = json.load(open(g2u_path)) if g2u_path.exists() else {}
    gem_uniprots = set(g2u.values())
    queries = []
    for e in mcsa_full:
        up = e.get("reference_uniprot_id", "")
        ecs = [ec for ec in e.get("all_ecs", []) if ec and ec.count(".") >= 3]
        if up in gem_uniprots and ecs and e.get("reaction", {}).get("compounds"):
            queries.append({
                "mcsa_id": e.get("mcsa_id"),
                "enzyme_name": e.get("enzyme_name", ""),
                "target_ec": ecs[0],
                "target_uniprot": up,
                "reaction": e.get("reaction", {}),
            })
    return queries


# ════════════════════════════════════════════════════════════════════════
#  EC matching utilities
# ════════════════════════════════════════════════════════════════════════

def _ec_fields(ec_str: str) -> list[str]:
    """Split a semicolon/comma-joined EC field into individual EC strings."""
    if not ec_str or str(ec_str).lower() in ("nan", "none", ""):
        return []
    return [e.strip() for e in str(ec_str).replace(",", ";").split(";") if e.strip()]


def ec_matches_at_level(target: str, candidate: str, level: int) -> bool:
    """True if target and candidate agree on the first `level` EC fields."""
    tp = target.strip().split(".")
    cp = candidate.strip().split(".")
    if len(tp) < level or len(cp) < level:
        return False
    for i in range(level):
        if not tp[i].isdigit() or not cp[i].isdigit():
            return False
        if tp[i] != cp[i]:
            return False
    return True


def find_ec_rank(target_ec: str, ranked_ec_rows: list[str], level: int) -> Optional[int]:
    """Return 1-based rank of first row whose EC matches target at `level`."""
    for rank, ec_str in enumerate(ranked_ec_rows, start=1):
        for cand_ec in _ec_fields(ec_str):
            if ec_matches_at_level(target_ec, cand_ec, level):
                return rank
    return None


# ════════════════════════════════════════════════════════════════════════
#  MicrobeRX prediction with tqdm shim (notebook tqdm → plain tqdm)
# ════════════════════════════════════════════════════════════════════════

def _patch_tqdm():
    """Replace tqdm.notebook with plain tqdm so MicrobeRX runs in a script."""
    import types
    try:
        from tqdm import tqdm as plain_tqdm
        notebook_mod = types.ModuleType("tqdm.notebook")
        notebook_mod.tqdm = plain_tqdm
        sys.modules["tqdm.notebook"] = notebook_mod
    except ImportError:
        pass


def run_microberx_prediction(substrate_smiles: str,
                              query_name: str,
                              biosystem: str,
                              confidence_cutoff: float) -> Optional[object]:
    """Run MicrobeRX MetabolitePredictor on a single substrate SMILES.

    Returns the predictor object (with .predicted_metabolites populated),
    or None if the substrate cannot be parsed or prediction fails.
    """
    from microberx.MetabolitePredictor import MetabolitePredictor

    # Take only the first (dominant) component of a dot-joined substrate
    components = [s for s in substrate_smiles.split(".") if s]
    if not components:
        return None

    # Use largest molecule by heavy-atom count as the query
    best_mol, best_n = None, -1
    for smi in components:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        n = mol.GetNumHeavyAtoms()
        if n > best_n:
            best_n, best_mol = n, mol

    if best_mol is None:
        return None

    try:
        predictor = MetabolitePredictor(
            query=best_mol,
            query_name=query_name,
            cut_off=confidence_cutoff,
            biosystem=biosystem,
        )
        predictor.run_prediction()
        return predictor
    except Exception as exc:
        log.warning("MicrobeRX prediction failed for %s: %s", query_name, exc)
        return None


# ════════════════════════════════════════════════════════════════════════
#  Rank derivation from predicted_metabolites DataFrame
# ════════════════════════════════════════════════════════════════════════

def extract_ranked_ec_list(predictor) -> list[str]:
    """Return EC strings in confidence-score order, one entry per unique
    (bigg_reaction, ec) combination, de-duplicated by max confidence.

    This is the correct ranking unit for enzyme-attribution evaluation:
    each row represents one CSRR rule with one EC annotation, ordered by
    how well the rule matched the query substrate.
    """
    import pandas as pd
    df = predictor.predicted_metabolites
    if df.empty:
        return []

    # Keep the best confidence score per (bigg_reaction, ec) pair
    if "ec" not in df.columns:
        return []

    df = df[["bigg_reaction", "ec", "confidence_score"]].copy()
    df["ec"] = df["ec"].fillna("").astype(str)

    # Deduplicate: one row per unique (bigg_reaction, ec), max confidence
    best = (df.groupby(["bigg_reaction", "ec"], sort=False)["confidence_score"]
              .max()
              .reset_index()
              .sort_values("confidence_score", ascending=False, ignore_index=True))

    return best["ec"].tolist()


# ════════════════════════════════════════════════════════════════════════
#  Main evaluation loop
# ════════════════════════════════════════════════════════════════════════

def run_evaluation(args) -> dict:
    _patch_tqdm()

    queries = load_queries(Path(args.query_set), MCSA_FULL_JSON)
    print(f"\nLoaded {len(queries)} M-CSA queries")
    print(f"MicrobeRX biosystem: {args.biosystem}   "
          f"confidence cutoff: {args.confidence_cutoff}   "
          f"EC level: {args.ec_level}\n")

    per_query = []

    try:
        from tqdm import tqdm as _tqdm
        query_iter = _tqdm(queries, desc="MicrobeRX eval", unit="query")
    except ImportError:
        query_iter = queries

    for q in query_iter:
        substrate_smi = _build_substrate_smiles(q)
        rec = {
            "mcsa_id": q["mcsa_id"],
            "enzyme_name": q["enzyme_name"],
            "target_ec": q["target_ec"],
            "target_uniprot": q["target_uniprot"],
        }

        if not substrate_smi:
            rec["status"] = "smiles_failed"
            per_query.append(rec)
            continue

        predictor = run_microberx_prediction(
            substrate_smi,
            query_name=f"mcsa_{q['mcsa_id']}",
            biosystem=args.biosystem,
            confidence_cutoff=args.confidence_cutoff,
        )

        if predictor is None:
            rec["status"] = "prediction_failed"
            per_query.append(rec)
            continue

        n_predictions = len(predictor.predicted_metabolites)
        if n_predictions == 0:
            rec["status"] = "no_predictions"
            rec["n_predictions"] = 0
            per_query.append(rec)
            continue

        ranked_ecs = extract_ranked_ec_list(predictor)
        ranks = {}
        for level in [1, 2, 3, 4]:
            ranks[f"ec_rank_L{level}"] = find_ec_rank(
                q["target_ec"], ranked_ecs, level)

        rec.update({
            "status": "ok",
            "n_predictions": n_predictions,
            "n_unique_rules": len(ranked_ecs),
            "substrate_smiles": substrate_smi,
            **ranks,
        })
        per_query.append(rec)

    aggregate = _aggregate(per_query, args.ec_level)
    return {
        "benchmark": "MicrobeRX M-CSA EC-attribution evaluation",
        "config": {
            "biosystem": args.biosystem,
            "confidence_cutoff": args.confidence_cutoff,
            "ec_level": args.ec_level,
            "query_set": Path(args.query_set).name,
            "n_queries": len(queries),
        },
        "aggregate": aggregate,
        "per_query": per_query,
    }


def _reciprocal_rank(rank: Optional[int]) -> float:
    return 0.0 if rank is None else 1.0 / rank


def _aggregate(per_query: list[dict], primary_level: int) -> dict:
    n_total = len(per_query)
    n_ok = sum(1 for r in per_query if r["status"] == "ok")
    n_no_pred = sum(1 for r in per_query if r["status"] == "no_predictions")
    n_smiles_fail = sum(1 for r in per_query if r["status"] == "smiles_failed")
    n_pred_fail = sum(1 for r in per_query if r["status"] == "prediction_failed")

    agg = {
        "n_total": n_total,
        "n_evaluable": n_ok,
        "n_no_predictions": n_no_pred,
        "n_smiles_failed": n_smiles_fail,
        "n_prediction_failed": n_pred_fail,
        "coverage_pct": round(n_ok / n_total * 100, 1) if n_total else 0,
    }

    for level in [1, 2, 3, 4]:
        key = f"ec_rank_L{level}"
        # Unconditional: all 130 queries in denominator (no-prediction = rank 0)
        all_ranks = [r.get(key) for r in per_query]
        rrs = [_reciprocal_rank(r) for r in all_ranks]
        mrr = float(np.mean(rrs)) if rrs else 0.0
        top1 = sum(1 for r in all_ranks if r == 1) / n_total * 100
        top5 = sum(1 for r in all_ranks if r is not None and r <= 5) / n_total * 100
        top10 = sum(1 for r in all_ranks if r is not None and r <= 10) / n_total * 100
        agg[f"L{level}"] = {
            "mrr": round(mrr, 3),
            "top1_pct": round(top1, 1),
            "top5_pct": round(top5, 1),
            "top10_pct": round(top10, 1),
        }

    return agg


# ════════════════════════════════════════════════════════════════════════
#  Statistics file for manuscript Table 5
# ════════════════════════════════════════════════════════════════════════

def build_statistics(result: dict) -> dict:
    """Build a statistics summary matching the format needed for Table 5."""
    agg = result["aggregate"]
    cfg = result["config"]
    primary = f"L{cfg['ec_level']}"
    return {
        "system": "MicrobeRX",
        "input": "Substrate SMILES only",
        "query_set": "130 M-CSA queries (same as ChemGIME headline benchmark)",
        "biosystem": cfg["biosystem"],
        "confidence_cutoff": cfg["confidence_cutoff"],
        "n_queries": cfg["n_queries"],
        "n_evaluable": agg["n_evaluable"],
        "coverage_pct": agg["coverage_pct"],
        "primary_ec_level": cfg["ec_level"],
        "mrr": agg[primary]["mrr"],
        "top1_pct": agg[primary]["top1_pct"],
        "top5_pct": agg[primary]["top5_pct"],
        "top10_pct": agg[primary]["top10_pct"],
        "all_levels": {f"L{l}": agg[f"L{l}"] for l in [1, 2, 3, 4]},
    }


# ════════════════════════════════════════════════════════════════════════
#  Reporting
# ════════════════════════════════════════════════════════════════════════

def print_summary(result: dict) -> None:
    agg = result["aggregate"]
    cfg = result["config"]
    print("\n" + "=" * 68)
    print("MicrobeRX M-CSA EC-ATTRIBUTION BENCHMARK — SUMMARY")
    print("=" * 68)
    print(f"  Biosystem: {cfg['biosystem']}   "
          f"Confidence cutoff: {cfg['confidence_cutoff']}   "
          f"EC level: {cfg['ec_level']}")
    print(f"  Total queries: {agg['n_total']}   "
          f"Evaluable (≥1 prediction): {agg['n_evaluable']} "
          f"({agg['coverage_pct']:.1f}%)")
    print(f"  No predictions returned: {agg['n_no_predictions']}   "
          f"SMILES failed: {agg['n_smiles_failed']}")
    print()
    print(f"  {'Level':<8} {'MRR':>7} {'Top-1%':>8} {'Top-5%':>8} {'Top-10%':>9}")
    print(f"  {'-'*46}")
    for level in [1, 2, 3, 4]:
        m = agg[f"L{level}"]
        marker = " *" if level == cfg["ec_level"] else ""
        print(f"  EC L{level}   {m['mrr']:>7.3f} {m['top1_pct']:>8.1f} "
              f"{m['top5_pct']:>8.1f} {m['top10_pct']:>9.1f}{marker}")
    print()
    print("  (* = primary comparison metric)")
    print()
    print("  ChemGIME reference (iML1515, QRFP kept, substrate+product input):")
    print("       MRR=0.825  Top-1=77.7%  Top-5=87.7%  Top-10=90.0%")
    print("=" * 68)


# ════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(
        description="MicrobeRX vs ChemGIME EC-attribution benchmark on 130 M-CSA queries")
    p.add_argument("--query-set", default=str(DEFAULT_QUERY_SET))
    p.add_argument("--biosystem", choices=["all", "human", "gutmicrobes"],
                   default="all",
                   help="MicrobeRX biosystem filter (default: all)")
    p.add_argument("--confidence-cutoff", type=float, default=0.6,
                   help="MicrobeRX confidence_score threshold (default: 0.6)")
    p.add_argument("--ec-level", type=int, choices=[1, 2, 3, 4], default=3,
                   help="EC hierarchy level for primary matching (default: 3 = sub-subclass)")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    result = run_evaluation(args)
    print_summary(result)

    tag = f"{args.biosystem}_cutoff{args.confidence_cutoff}"
    out_path = (Path(args.output) if args.output
                else DATA_DIR / f"microberx_mcsa_results_{tag}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(f"\nFull results saved to: {out_path}")

    stats = build_statistics(result)
    stats_path = out_path.with_name(out_path.stem.replace("results", "statistics") + ".json")
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, default=str)
    print(f"Statistics saved to:   {stats_path}")


if __name__ == "__main__":
    main()
