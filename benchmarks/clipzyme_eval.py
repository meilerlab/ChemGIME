"""
CLIPZyme Constrained + Unconstrained Evaluation
=================================================

Axis 1:  Constrained — CLIPZyme scores only iML1515 proteins (via UniProt bridge).
Axis 1b: Unconstrained — CLIPZyme scores all 260K proteins in its screening set.

Both axes use the same 443 Split-GEM test queries and evaluate by GPR match,
identical to ChemGIME's benchmark.

Usage (run in clipzyme conda env):
    python benchmarks/clipzyme_eval.py
"""

from __future__ import annotations

import json
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Paths
CHECKPOINT = PROJECT_ROOT / "vendor" / "clipzyme" / "files" / "clipzyme_model.ckpt"
SCREENING_SET = PROJECT_ROOT / "vendor" / "clipzyme" / "files" / "clipzyme_screening_set.p"
BRIDGE = SCRIPT_DIR / "data" / "clipzyme_iml1515_bridge.json"
TEST_CSV = SCRIPT_DIR / "data" / "iML1515_test_atommapped.csv"
EC2UNIPROT = PROJECT_ROOT / "vendor" / "clipzyme" / "files" / "ec2uniprot.p"
OUTPUT_CONSTRAINED = SCRIPT_DIR / "data" / "clipzyme_constrained_results.json"
OUTPUT_UNCONSTRAINED = SCRIPT_DIR / "data" / "clipzyme_unconstrained_results.json"


def normalize_gpr(gpr_str: str) -> str:
    if not gpr_str or pd.isna(gpr_str):
        return ""
    return gpr_str.strip().lower().replace("  ", " ")


def extract_genes_from_gpr(gpr_str: str) -> set[str]:
    if not gpr_str or pd.isna(gpr_str):
        return set()
    return set(re.findall(r'\b(b\d{4})\b', str(gpr_str)))


def check_gpr_match(query_gpr: str, candidate_gpr: str) -> bool:
    """Check if two GPR strings share at least one gene."""
    q_genes = extract_genes_from_gpr(query_gpr)
    c_genes = extract_genes_from_gpr(candidate_gpr)
    if not q_genes or not c_genes:
        return False
    return bool(q_genes & c_genes)


def main():
    print("=" * 70)
    print("CLIPZyme Evaluation: Axis 1 (Constrained) + Axis 1b (Unconstrained)")
    print("=" * 70)

    # ---- Load data ----
    print("\n[1/5] Loading data...")

    # Bridge
    with open(BRIDGE) as f:
        bridge = json.load(f)
    print(f"  Bridge: {bridge['overlap_stats']['overlap_count']} iML1515 proteins in screening set")

    # Screening set
    print("  Loading screening set (261K × 1280)...")
    screenset = pickle.load(open(SCREENING_SET, "rb"))
    screen_hiddens = screenset["hiddens"]  # torch.Tensor (261907, 1280)
    screen_unis = screenset["uniprots"]    # list of 261907 UniProt IDs
    if hasattr(screen_unis, "tolist"):
        screen_unis = screen_unis.tolist()
    print(f"  Screening set: {screen_hiddens.shape}")

    # Test reactions
    test_df = pd.read_csv(TEST_CSV)
    test_mapped = test_df[test_df["mapped_smiles"].notna() & (test_df["mapped_smiles"] != "")]
    print(f"  Test reactions: {len(test_df)} total, {len(test_mapped)} with atom maps")

    # Build reverse lookup: screening_idx → UniProt
    # and UniProt → gene mapping from bridge
    uni_to_genes = {}
    for gene, info in bridge["gene_to_screen"].items():
        uni = info["uniprot"]
        uni_to_genes.setdefault(uni, set()).add(gene)

    # Constrained indices
    constrained_indices = bridge["all_iml1515_screen_indices"]
    print(f"  Constrained search space: {len(constrained_indices)} proteins")

    # ---- Load CLIPZyme model ----
    print("\n[2/5] Loading CLIPZyme model...")
    import os
    os.chdir(str(PROJECT_ROOT / "vendor" / "clipzyme"))
    from clipzyme import CLIPZyme
    model = CLIPZyme(checkpoint_path=str(CHECKPOINT))
    model = model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # Pre-compute constrained hiddens
    constrained_hiddens = screen_hiddens[constrained_indices].to(device)  # (N_constrained, 1280)
    constrained_unis = [screen_unis[i] for i in constrained_indices]
    full_hiddens = screen_hiddens.to(device)  # (261907, 1280)

    # ---- Encode reactions ----
    print(f"\n[3/5] Encoding {len(test_mapped)} test reactions...")
    reaction_embeddings = {}
    encode_times = []

    for _, row in tqdm(test_mapped.iterrows(), total=len(test_mapped), desc="  Encoding"):
        rxn_id = row["reaction_id"]
        mapped_smi = row["mapped_smiles"]
        try:
            t0 = time.perf_counter()
            with torch.no_grad():
                emb = model.extract_reaction_features(reaction=mapped_smi)  # (1, 1280)
            t1 = time.perf_counter()
            reaction_embeddings[rxn_id] = emb.to(device)
            encode_times.append(t1 - t0)
        except Exception as e:
            pass  # Skip failed encodings

    print(f"  Successfully encoded: {len(reaction_embeddings)}/{len(test_mapped)}")
    if encode_times:
        print(f"  Mean encode time: {np.mean(encode_times)*1000:.1f} ms")
        print(f"  Median encode time: {np.median(encode_times)*1000:.1f} ms")

    # ---- Axis 1: Constrained Evaluation ----
    print(f"\n[4/5] Axis 1: Constrained evaluation ({len(constrained_indices)} proteins)...")
    constrained_results = []
    constrained_query_times = []

    for _, row in tqdm(test_mapped.iterrows(), total=len(test_mapped), desc="  Constrained"):
        rxn_id = row["reaction_id"]
        true_gpr = str(row.get("true_gpr", ""))

        if rxn_id not in reaction_embeddings:
            continue

        rxn_emb = reaction_embeddings[rxn_id]  # (1, 1280)

        t0 = time.perf_counter()
        scores = (constrained_hiddens @ rxn_emb.T).squeeze()  # (N_constrained,)
        ranked_idx = torch.argsort(scores, descending=True)
        t1 = time.perf_counter()
        constrained_query_times.append(t1 - t0)

        # Find rank of first GPR match
        found_rank = None
        for rank, idx in enumerate(ranked_idx.cpu().numpy()):
            protein_uni = constrained_unis[idx]
            # Get genes for this protein
            protein_genes = uni_to_genes.get(protein_uni, set())
            if not protein_genes:
                continue
            # Build a pseudo-GPR from the protein's genes
            candidate_gpr = " or ".join(sorted(protein_genes))
            if check_gpr_match(true_gpr, candidate_gpr):
                found_rank = rank + 1  # 1-indexed
                break

        constrained_results.append({
            "reaction_id": rxn_id,
            "true_gpr": true_gpr,
            "found_at_rank": found_rank,
        })

    # ---- Axis 1b: Unconstrained Evaluation ----
    print(f"\n[5/5] Axis 1b: Unconstrained evaluation ({len(screen_unis)} proteins)...")
    unconstrained_results = []
    unconstrained_query_times = []

    # Build set of target UniProts for each reaction
    for _, row in tqdm(test_mapped.iterrows(), total=len(test_mapped), desc="  Unconstrained"):
        rxn_id = row["reaction_id"]
        true_gpr = str(row.get("true_gpr", ""))

        if rxn_id not in reaction_embeddings:
            continue

        # Get target UniProts for this reaction from bridge
        rxn_bridge = bridge["reaction_to_proteins"].get(rxn_id, {})
        target_unis = set(rxn_bridge.get("uniprots", []))

        rxn_emb = reaction_embeddings[rxn_id]

        t0 = time.perf_counter()
        scores = (full_hiddens @ rxn_emb.T).squeeze()  # (261907,)
        ranked_idx = torch.argsort(scores, descending=True)
        t1 = time.perf_counter()
        unconstrained_query_times.append(t1 - t0)

        # Find rank of first target UniProt in full ranking
        found_rank = None
        if target_unis:
            for rank, idx in enumerate(ranked_idx.cpu().numpy()):
                if screen_unis[idx] in target_unis:
                    found_rank = rank + 1
                    break

        # Also find rank via GPR gene match (broader)
        found_rank_gpr = None
        true_genes = extract_genes_from_gpr(true_gpr)
        if true_genes:
            for rank, idx in enumerate(ranked_idx[:10000].cpu().numpy()):
                protein_uni = screen_unis[idx]
                protein_genes = uni_to_genes.get(protein_uni, set())
                if protein_genes & true_genes:
                    found_rank_gpr = rank + 1
                    break

        unconstrained_results.append({
            "reaction_id": rxn_id,
            "true_gpr": true_gpr,
            "target_uniprots": list(target_unis),
            "found_at_rank_uniprot": found_rank,
            "found_at_rank_gpr_10k": found_rank_gpr,
        })

    # ---- Compute and print metrics ----
    print("\n" + "=" * 70)
    print("AXIS 1: CONSTRAINED RESULTS")
    print("=" * 70)

    ranks_c = [r["found_at_rank"] for r in constrained_results]
    n_c = len(ranks_c)
    n_found_c = sum(1 for r in ranks_c if r is not None)
    rrs_c = [1.0 / r if r else 0.0 for r in ranks_c]
    mrr_c = np.mean(rrs_c) if rrs_c else 0.0

    def top_k_pct(ranks, k):
        return sum(1 for r in ranks if r is not None and r <= k) / len(ranks) * 100 if ranks else 0

    print(f"  Queries:     {n_c}")
    print(f"  Found:       {n_found_c}/{n_c} ({n_found_c/n_c*100:.1f}%)")
    print(f"  MRR:         {mrr_c:.4f}")
    print(f"  Top-1:       {top_k_pct(ranks_c, 1):.2f}%")
    print(f"  Top-5:       {top_k_pct(ranks_c, 5):.2f}%")
    print(f"  Top-10:      {top_k_pct(ranks_c, 10):.2f}%")
    print(f"  Top-20:      {top_k_pct(ranks_c, 20):.2f}%")
    print(f"  Top-100:     {top_k_pct(ranks_c, 100):.2f}%")
    if constrained_query_times:
        print(f"  Query time:  {np.median(constrained_query_times)*1000:.2f} ms (median)")

    print("\n" + "=" * 70)
    print("AXIS 1b: UNCONSTRAINED RESULTS (260K screening set)")
    print("=" * 70)

    ranks_u_uni = [r["found_at_rank_uniprot"] for r in unconstrained_results]
    ranks_u_gpr = [r["found_at_rank_gpr_10k"] for r in unconstrained_results]
    n_u = len(ranks_u_uni)
    n_with_target = sum(1 for r in unconstrained_results if r["target_uniprots"])
    n_found_uni = sum(1 for r in ranks_u_uni if r is not None)
    n_found_gpr = sum(1 for r in ranks_u_gpr if r is not None)

    rrs_u = [1.0 / r if r else 0.0 for r in ranks_u_uni]
    mrr_u = np.mean(rrs_u) if rrs_u else 0.0

    print(f"  Queries:             {n_u}")
    print(f"  With target UniProt: {n_with_target}/{n_u}")
    print(f"  Found (UniProt):     {n_found_uni}/{n_u} ({n_found_uni/n_u*100:.1f}%)")
    print(f"  Found (GPR, top10K): {n_found_gpr}/{n_u} ({n_found_gpr/n_u*100:.1f}%)")
    print(f"  MRR (UniProt):       {mrr_u:.6f}")
    print(f"  Top-10:              {top_k_pct(ranks_u_uni, 10):.2f}%")
    print(f"  Top-100:             {top_k_pct(ranks_u_uni, 100):.2f}%")
    print(f"  Top-1000:            {top_k_pct(ranks_u_uni, 1000):.2f}%")
    print(f"  Top-10000:           {top_k_pct(ranks_u_uni, 10000):.2f}%")

    found_uni_ranks = [r for r in ranks_u_uni if r is not None]
    if found_uni_ranks:
        print(f"  Median rank (found): {np.median(found_uni_ranks):.0f}")
        print(f"  Mean rank (found):   {np.mean(found_uni_ranks):.0f}")

    if unconstrained_query_times:
        print(f"  Query time:          {np.median(unconstrained_query_times)*1000:.2f} ms (median)")

    # ---- Comparison table ----
    print("\n" + "=" * 70)
    print("HEAD-TO-HEAD COMPARISON (Constrained, same search space)")
    print("=" * 70)
    print(f"  {'Metric':<15} {'ChemGIME (DRFP)':<20} {'CLIPZyme (Constrained)':<25}")
    print(f"  {'MRR':<15} {'0.2772':<20} {f'{mrr_c:.4f}':<25}")
    print(f"  {'Top-1':<15} {'23.23%':<20} {f'{top_k_pct(ranks_c, 1):.2f}%':<25}")
    print(f"  {'Top-5':<15} {'31.30%':<20} {f'{top_k_pct(ranks_c, 5):.2f}%':<25}")
    print(f"  {'Top-10':<15} {'33.01%':<20} {f'{top_k_pct(ranks_c, 10):.2f}%':<25}")
    print(f"  {'Top-20':<15} {'49.39%':<20} {f'{top_k_pct(ranks_c, 20):.2f}%':<25}")
    print(f"  {'Top-100':<15} {'57.70%':<20} {f'{top_k_pct(ranks_c, 100):.2f}%':<25}")
    print(f"  {'Hardware':<15} {'CPU only':<20} {'GPU (CUDA)':<25}")

    # ---- Save results ----
    constrained_out = {
        "n_queries": n_c,
        "n_found": n_found_c,
        "mrr": mrr_c,
        "top1": top_k_pct(ranks_c, 1),
        "top5": top_k_pct(ranks_c, 5),
        "top10": top_k_pct(ranks_c, 10),
        "top20": top_k_pct(ranks_c, 20),
        "top100": top_k_pct(ranks_c, 100),
        "median_encode_time_ms": np.median(encode_times) * 1000 if encode_times else None,
        "median_query_time_ms": np.median(constrained_query_times) * 1000 if constrained_query_times else None,
        "per_query": constrained_results,
    }

    unconstrained_out = {
        "n_queries": n_u,
        "n_with_target": n_with_target,
        "n_found_uniprot": n_found_uni,
        "n_found_gpr_10k": n_found_gpr,
        "mrr_uniprot": mrr_u,
        "top10": top_k_pct(ranks_u_uni, 10),
        "top100": top_k_pct(ranks_u_uni, 100),
        "top1000": top_k_pct(ranks_u_uni, 1000),
        "top10000": top_k_pct(ranks_u_uni, 10000),
        "median_rank_found": float(np.median(found_uni_ranks)) if found_uni_ranks else None,
        "median_query_time_ms": np.median(unconstrained_query_times) * 1000 if unconstrained_query_times else None,
        "per_query": unconstrained_results,
    }

    with open(OUTPUT_CONSTRAINED, "w") as f:
        json.dump(constrained_out, f, indent=2)
    with open(OUTPUT_UNCONSTRAINED, "w") as f:
        json.dump(unconstrained_out, f, indent=2)

    print(f"\nResults saved:")
    print(f"  {OUTPUT_CONSTRAINED}")
    print(f"  {OUTPUT_UNCONSTRAINED}")


if __name__ == "__main__":
    main()
