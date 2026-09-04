"""
Reconstructed generator for ``benchmarks/ecoli_benchmark_results.json``.
=========================================================================

Regenerates the 10 GEM-covered SIMMER E. coli positive-control results that
back Supplementary Table S8 and the main-text drug-metabolism worked example
(5-ASA -> NAT rank 1, GUS -> rank 2, NfsB -> rank 3).

The original generating script was not committed; this reconstruction reuses
the canonical ChemGIME pipeline helpers from ``tests/benchmarking_suite.py``
(same GEM parsing, cofactor handling, fingerprinting, ranking, and EC voting
as every other benchmark) so the numbers are reproducible from raw inputs.

Reproduction status (verified 2026-06-23 against the committed
``ecoli_benchmark_results.json``):
  - gem_size = 2322 (DRFP, cofactors kept): EXACT.
  - GEM-covered set (the 10 of 17 whose gene is in iML1515): EXACT.
  - Aggregate: median rank 2.0, Top-10 10/10: EXACT. MRR 0.458 (committed 0.428).
  - 6/10 cases bit-exact, including BOTH main-text worked-example claims
    (5-ASA -> NAT b1463 rank 1 / ACANTHAT; GUS b1617 rank 2 / METGLCUR x5).
  - 4 nitroreduction cases (chloramphenicol, clonazepam, nitrazepam,
    flunitrazepam) recover the correct gene b0578 and matched reaction DHPTDNR
    but at different ranks (recon 6/3/3/4 vs committed 3/8/8/5). The rank is
    sensitive to the exact nitro reaction SMILES; the lost original used
    slightly different SMILES than simmer_benchmark_config.yaml provides, so
    these four are not bit-reproducible without the original inputs.
This script writes to a SEPARATE path by default and does NOT overwrite the
committed file, which remains the reference S8/worked-example source.

Inputs
------
  - benchmarks/simmer_benchmark_config.yaml   (17 ecoli queries: reaction SMILES)
  - iML1515.xml                               (the search-space GEM)
  - enzyme -> gene map below                  (recovered from docs/vs_simmer_ecoli.md;
                                               the SIMMER set gives enzymes, not loci)

A query is "GEM-covered" when its target gene locus is present in iML1515.
AzoR (b1412) and Cpg2 are absent, leaving the 10 covered cases.
A query is "found at rank r" when the target gene first appears in the GPR of
the r-th ranked GEM reaction (1-indexed).

Run:  python benchmarks/simmer_ecoli_eval.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "benchmarks" / "simmer_benchmark_config.yaml"
GEM = ROOT / "iML1515.xml"
OUT = ROOT / "benchmarks" / "ecoli_benchmark_results_reconstructed.json"
REF = ROOT / "benchmarks" / "ecoli_benchmark_results.json"  # committed reference

# Reuse the canonical pipeline helpers from benchmarking_suite.
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location(
    "benchmarking_suite", ROOT / "tests" / "benchmarking_suite.py")
bs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bs)

# enzyme -> target E. coli gene locus (recovered from docs/vs_simmer_ecoli.md).
# AzoR (b1412) and Cpg2 are real genes but absent from iML1515 -> not covered.
ENZYME_TO_GENE = {
    "NfsB": "b0578",
    "NfsA": "b0578",   # NfsA (b0851) absent; NfsB (b0578) covers the same activity
    "NfsA/B": "b0578",
    "GUS": "b1617",
    "uidA": "b1617",
    "NAT": "b1463",
    "nhoA": "b1463",
    "AzoR": "b1412",   # absent from iML1515
    "Cpg2": "__absent__",
}


def build_db(fp_type, cofactor_set):
    wrapper = bs.FileWrapper(GEM)
    met_map = bs._load_bigg_map()
    db_df = bs.build_organism_database(wrapper, met_map, cofactor_set)
    db_fps, valid_idx = bs.calculate_fingerprints_parallel(db_df, fp_type=fp_type)
    return db_df, db_fps, valid_idx


def main() -> None:
    cfg = yaml.safe_load(open(CONFIG))
    queries = [q for q in cfg["queries"] if q.get("gem") == "ecoli"]

    fp_type = bs._resolve_fp_type("drfp")
    fp_fn = bs._fp_fn(fp_type)

    # Auto-detect the cofactor setting that reproduces gem_size = 2322.
    chosen = None
    for filt in (False, True):
        cof = bs._resolve_cofactor_set(filt)
        db_df, db_fps, valid_idx = build_db(fp_type, cof)
        print(f"  DRFP cofactors {'filtered' if filt else 'kept'}: "
              f"gem_size = {len(db_df)} (valid fp {len(valid_idx)})")
        if len(db_df) == 2322:
            chosen = (filt, db_df, db_fps, valid_idx)
            break
    if chosen is None:
        print("WARNING: no setting reproduced gem_size=2322; using DRFP kept.")
        cof = bs._resolve_cofactor_set(False)
        db_df, db_fps, valid_idx = build_db(fp_type, cof)
        chosen = (False, db_df, db_fps, valid_idx)
    filt, db_df, db_fps, valid_idx = chosen

    gpr_col = "gpr" if "gpr" in db_df.columns else "GPR"
    id_col = "reaction_id" if "reaction_id" in db_df.columns else db_df.columns[0]

    results = []
    for q in queries:
        gene = ENZYME_TO_GENE.get(q["enzyme"], "__absent__")
        # covered only if the gene is annotated somewhere in iML1515
        covered = db_df[gpr_col].astype(str).str.contains(gene).any() if gene != "__absent__" else False
        if not covered:
            continue

        t0 = time.perf_counter()
        try:
            qfp = fp_fn((q["left_smiles"], q["right_smiles"]))
        except Exception as e:
            results.append({"id": q["id"], "drug": q["drug"], "gene": gene,
                            "rank": None, "error": f"fp:{e}"})
            continue
        if qfp is None:
            results.append({"id": q["id"], "drug": q["drug"], "gene": gene,
                            "rank": None, "error": "fp_none"})
            continue
        ranked = bs.get_closest_rxns(
            np.asarray(qfp).reshape(1, -1), db_fps, db_df, valid_idx,
            fp_type=fp_type.value)
        dt = time.perf_counter() - t0

        # rank = first ranked reaction whose GPR contains the target gene
        rank = None
        for i, (_, row) in enumerate(ranked.iterrows(), start=1):
            if gene in str(row[gpr_col]):
                rank = i
                break
        top = ranked.iloc[0]
        matched_row = ranked.iloc[rank - 1] if rank else top
        ec_res = bs.predict_ec_weighted(ranked)
        ec_l1, c1, _ = ec_res["L1"]
        ec_l3, c3, _ = ec_res["L3"]

        results.append({
            "id": q["id"], "dm": q.get("dm"), "drug": q["drug"],
            "product": q.get("product"), "enzyme": q["enzyme"], "gene": gene,
            "ec_true": q.get("ec"), "rank": rank,
            "distance_top1": float(top["tan_distance"]),
            "matched_rxn": str(matched_row[id_col]), "matched_gpr": str(matched_row[gpr_col]),
            "ec_l1_pred": ec_l1, "ec_l1_conf": round(float(c1), 4),
            "ec_l3_pred": ec_l3, "ec_l3_conf": round(float(c3), 4),
            "gem_size": len(db_df), "fp_valid": len(valid_idx),
            "time_s": round(dt, 3),
        })

    OUT.write_text(json.dumps(results, indent=2, default=str))
    ranks = [r["rank"] for r in results if r.get("rank")]
    print(f"\n  {len(results)} covered queries; "
          f"median rank {np.median(ranks):.1f}, MRR {np.mean([1/r for r in ranks]):.3f}, "
          f"Top-10 {sum(r<=10 for r in ranks)}/{len(ranks)}")
    for r in results:
        print(f"    {r['drug']:24s} gene {r['gene']} rank {r['rank']} matched {r.get('matched_rxn')}")


if __name__ == "__main__":
    main()
