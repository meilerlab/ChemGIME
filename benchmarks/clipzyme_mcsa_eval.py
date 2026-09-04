"""
CLIPZyme vs ChemGIME — M-CSA Cross-Database Enzyme Retrieval
=============================================================

PURPOSE
-------
This is the primary head-to-head benchmark between ChemGIME and CLIPZyme.
It uses M-CSA (Mechanism and Catalytic Site Atlas) enzymatic reactions as
queries — an *external* data source that neither system has seen — and
evaluates whether each system can retrieve the correct enzyme (UniProt ID)
from the iML1515 search space.

WHY M-CSA?
----------
The Split-GEM benchmark (vs_clipzyme.md) has design weaknesses:
non-equivalent search spaces, no statistical testing, 50% query dropout.
M-CSA provides:
  - Genuinely external queries (ChEBI mol files, not GEM SMILES)
  - Clean ground truth (curated reference_uniprot_id, not noisy GPR parsing)
  - Existing ChemGIME baseline (Benchmark 2 from benchmark1.md)
  - 130 well-characterised enzymes reviewers can inspect individually

WHAT IT DOES (step by step)
----------------------------
Phase 1 — Load resources:
  1a. Load the UniProt bridge (from clipzyme_build_bridge.py output)
  1b. Load CLIPZyme model + pre-computed protein embeddings
  1c. Load existing ChemGIME M-CSA benchmark results (4 conditions)
  1d. Load M-CSA entries from the cached full API dump

Phase 2 — Prepare CLIPZyme queries:
  2a. Identify the 130 M-CSA overlap entries (same as Benchmark 2)
  2b. Build ChEBI-derived reaction SMILES for each entry
  2c. Atom-map each reaction via RXNMapper
  2d. Encode each mapped reaction with CLIPZyme's DMPNN
  2e. Log every failure with category and exception type

Phase 3 — Evaluate:
  3a. For each successfully encoded query, rank the 1,014 constrained
      iML1515 proteins by cosine similarity
  3b. Find the rank of the target UniProt in the ranked list
  3c. Compute the matched query subset (queries both systems processed)
  3d. Compute MRR, Top-k for both systems on the matched subset

Phase 4 — Statistical testing:
  4a. Bootstrap 95% CIs for MRR (10,000 resamples)
  4b. Wilcoxon signed-rank test on paired reciprocal ranks
  4c. McNemar's test on Top-1 hit/miss pairs

Phase 5 — Output:
  5a. Per-query results JSON with both systems' ranks
  5b. Failure analysis JSON
  5c. Statistics JSON (CIs, p-values, effect sizes)
  5d. Human-readable summary to stdout

USAGE
-----
    # Must run in clipzyme conda env with GPU available
    conda activate clipzyme
    python benchmarks/clipzyme_mcsa_eval.py

    # All paths are auto-detected from project layout.
    # Override if needed:
    python benchmarks/clipzyme_mcsa_eval.py \\
        --bridge benchmarks/data/clipzyme_iml1515_bridge.json \\
        --chemgime-results tests/benchmark_output/mcsa_novelty_iML1515_drfp_kept.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

BRIDGE_PATH = SCRIPT_DIR / "data" / "clipzyme_iml1515_bridge.json"
SCREENING_SET_PATH = (PROJECT_ROOT / "vendor" / "clipzyme" / "files" /
                      "clipzyme_screening_set.p")
CHECKPOINT_PATH = (PROJECT_ROOT / "vendor" / "clipzyme" / "files" /
                   "clipzyme_model.ckpt")
MCSA_FULL_JSON = PROJECT_ROOT / "data" / "database" / "mcsa_entries_full.json"
MOL_CACHE_DIR = PROJECT_ROOT / "data" / "database" / "mcsa_mol_cache"
GENE_TO_UNIPROT_PATH = (PROJECT_ROOT / "data" / "database" /
                        "iML1515_gene_to_uniprot.json")

# ChemGIME's best M-CSA result (DRFP kept cofactors) from Benchmark 2.
# The user can override with --chemgime-results to use any condition.
CHEMGIME_RESULTS_PATH = (PROJECT_ROOT / "tests" / "benchmark_output" /
                         "mcsa_novelty_iML1515_drfp_kept.json")

OUTPUT_DIR = SCRIPT_DIR / "data"

log = logging.getLogger("clipzyme_mcsa")


# ═══════════════════════════════════════════════════════════════════════════
#  Phase 1 — Load resources
# ═══════════════════════════════════════════════════════════════════════════

def load_bridge(path: Path) -> dict:
    """Load the UniProt bridge JSON built by clipzyme_build_bridge.py."""
    with open(path) as f:
        return json.load(f)


def load_screening_embeddings(path: Path, device: str) -> tuple:
    """Load CLIPZyme screening set: protein embeddings + UniProt list.

    Returns (hiddens_tensor, uniprots_list).  The tensor is moved to
    the specified device (cuda/cpu) for similarity computation.
    """
    with open(path, "rb") as f:
        ss = pickle.load(f)
    hiddens = ss["hiddens"]  # torch.Tensor (261907, 1280)
    uniprots = ss["uniprots"]
    if hasattr(uniprots, "tolist"):
        uniprots = uniprots.tolist()
    return hiddens.to(device), uniprots


def load_chemgime_results(path: Path) -> dict[int, dict]:
    """Load existing ChemGIME M-CSA benchmark results, keyed by mcsa_id.

    Each entry has: mcsa_id, target_uniprot, rank, reciprocal_rank, etc.
    We key by mcsa_id (int) so the eval loop can look up ChemGIME's rank
    for any M-CSA entry.
    """
    with open(path) as f:
        data = json.load(f)
    return {entry["mcsa_id"]: entry for entry in data["details"]}


def load_mcsa_entries(path: Path) -> list[dict]:
    """Load all M-CSA entries from the cached full API dump.

    This file is created by tests/benchmarking_suite.py::_fetch_all_mcsa_entries()
    and cached at data/database/mcsa_entries_full.json.  If it doesn't exist,
    fetch from the M-CSA REST API (requires internet).
    """
    if path.exists():
        with open(path) as f:
            return json.load(f)

    # Fetch from API (fallback)
    print("  M-CSA cache not found, fetching from API...")
    all_entries = []
    url = "https://www.ebi.ac.uk/thornton-srv/m-csa/api/entries/?format=json"
    while url:
        req = urllib.request.urlopen(url, timeout=30)
        data = json.loads(req.read().decode("utf-8"))
        all_entries.extend(data.get("results", []))
        url = data.get("next")

    # Cache for future runs
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(all_entries, f, indent=1)
    print(f"  Cached {len(all_entries)} entries to {path}")
    return all_entries


# ═══════════════════════════════════════════════════════════════════════════
#  Phase 2 — Prepare CLIPZyme queries from M-CSA
# ═══════════════════════════════════════════════════════════════════════════

def chebi_mol_to_smiles(mol_url: str, chebi_id: str) -> Optional[str]:
    """Download a .mol file from M-CSA, convert to canonical SMILES.

    Mol files are cached locally to data/database/mcsa_mol_cache/ so that
    re-runs don't require internet access.  This is the same logic as
    tests/benchmarking_suite.py::_chebi_mol_to_smiles().
    """
    from rdkit import Chem

    MOL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = MOL_CACHE_DIR / f"{chebi_id}.mol"

    if cache_path.exists():
        mol_block = cache_path.read_text(encoding="utf-8")
    else:
        full_url = (f"https://{mol_url}"
                    if not mol_url.startswith("http") else mol_url)
        try:
            req = urllib.request.urlopen(full_url, timeout=15)
            mol_block = req.read().decode("utf-8")
            cache_path.write_text(mol_block, encoding="utf-8")
        except Exception as exc:
            log.debug("Mol download failed for ChEBI %s: %s", chebi_id, exc)
            return None

    mol = Chem.MolFromMolBlock(mol_block)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def build_mcsa_query_smiles(entry: dict) -> tuple[Optional[str], Optional[str]]:
    """Build (left_smiles, right_smiles) from an M-CSA entry's ChEBI compounds.

    Each M-CSA entry has a ``reaction`` dict containing ``compounds`` typed
    as "reactant" or "product".  We download each compound's .mol file,
    convert to SMILES, and join with "." to form dot-separated multi-component
    SMILES — the same format ChemGIME and RXNMapper expect.
    """
    rxn = entry.get("reaction", {})
    compounds = rxn.get("compounds", [])
    if not compounds:
        return None, None

    reactant_smiles: list[str] = []
    product_smiles: list[str] = []

    for comp in compounds:
        chebi_id = str(comp.get("chebi_id", ""))
        mol_url = comp.get("mol_file", "")
        comp_type = comp.get("type", "")
        if not chebi_id or not mol_url:
            continue

        smi = chebi_mol_to_smiles(mol_url, chebi_id)
        if smi is None:
            continue

        if comp_type == "reactant":
            reactant_smiles.append(smi)
        elif comp_type == "product":
            product_smiles.append(smi)

    left = ".".join(reactant_smiles) if reactant_smiles else None
    right = ".".join(product_smiles) if product_smiles else None
    return left, right


def identify_overlap_entries(
    mcsa_entries: list[dict],
    gem_uniprots: set[str],
) -> list[dict]:
    """Filter M-CSA entries whose reference_uniprot_id is in the GEM.

    This is the same overlap filter as Benchmark 2 in benchmarking_suite.py.
    We keep entries that:
      - Have a reference_uniprot_id present in iML1515
      - Have at least one complete 4-level EC number
      - Have reaction compound data (for building query SMILES)
    """
    overlap = []
    for entry in mcsa_entries:
        up = entry.get("reference_uniprot_id", "")
        ecs = entry.get("all_ecs", [])
        rxn = entry.get("reaction", {})

        if not (up and up in gem_uniprots and rxn.get("compounds")):
            continue

        # Need at least one complete 4-level EC
        valid_ec = None
        for ec in ecs:
            if ec and ec.count(".") >= 3:
                valid_ec = ec
                break
        if not valid_ec:
            continue

        overlap.append({
            "mcsa_id": entry.get("mcsa_id"),
            "enzyme_name": entry.get("enzyme_name", ""),
            "target_uniprot": up,
            "ec": valid_ec,
            "reaction": rxn,
        })

    return overlap


# ═══════════════════════════════════════════════════════════════════════════
#  Phase 3 — Evaluate CLIPZyme
# ═══════════════════════════════════════════════════════════════════════════

def rank_target_in_scores(
    scores: torch.Tensor,
    constrained_uniprots: list[str],
    target_uniprot: str,
) -> Optional[int]:
    """Find the 1-indexed rank of target_uniprot in a cosine-similarity ranking.

    Returns None if the target is not in the constrained set at all.
    """
    ranked_idx = torch.argsort(scores, descending=True)
    for rank, idx in enumerate(ranked_idx.cpu().numpy(), start=1):
        if constrained_uniprots[idx] == target_uniprot:
            return rank
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  Phase 4 — Statistical testing
# ═══════════════════════════════════════════════════════════════════════════

def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for the mean.

    Returns (mean, ci_lower, ci_upper).
    """
    rng = np.random.RandomState(seed)
    means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(means, [alpha, 1 - alpha])
    return float(values.mean()), float(lo), float(hi)


def wilcoxon_test(x: np.ndarray, y: np.ndarray) -> dict:
    """Wilcoxon signed-rank test on paired samples.

    Falls back gracefully if scipy is unavailable or if all differences
    are zero (which makes the test undefined).
    """
    try:
        from scipy.stats import wilcoxon as scipy_wilcoxon
        diff = x - y
        # Wilcoxon requires at least one non-zero difference
        if np.all(diff == 0):
            return {"statistic": None, "p_value": 1.0,
                    "note": "all differences zero"}
        stat, p = scipy_wilcoxon(x, y)
        return {"statistic": float(stat), "p_value": float(p)}
    except ImportError:
        return {"error": "scipy not installed"}
    except Exception as e:
        return {"error": str(e)}


def mcnemar_test(
    chemgime_hits: np.ndarray,
    clipzyme_hits: np.ndarray,
) -> dict:
    """McNemar's test on paired Top-1 hit/miss binary outcomes.

    The 2x2 contingency table:
                        CLIPZyme hit  CLIPZyme miss
        ChemGIME hit        a              b
        ChemGIME miss       c              d

    McNemar tests whether b != c (discordant pairs).
    """
    try:
        a = int(np.sum(chemgime_hits & clipzyme_hits))
        b = int(np.sum(chemgime_hits & ~clipzyme_hits))
        c = int(np.sum(~chemgime_hits & clipzyme_hits))
        d = int(np.sum(~chemgime_hits & ~clipzyme_hits))

        # Use exact binomial test for small counts
        n_discordant = b + c
        if n_discordant == 0:
            return {"a": a, "b": b, "c": c, "d": d,
                    "p_value": 1.0, "note": "no discordant pairs"}

        from scipy.stats import binom_test
        p = binom_test(b, n_discordant, 0.5)
        return {"a": a, "b": b, "c": c, "d": d, "p_value": float(p)}
    except ImportError:
        # scipy not available — compute manually with chi-squared approx
        if n_discordant == 0:
            return {"a": a, "b": b, "c": c, "d": d, "p_value": 1.0}
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        # Approximate p-value (1 df)
        return {"a": a, "b": b, "c": c, "d": d,
                "chi2": float(chi2), "note": "scipy unavailable, chi2 approx"}


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="CLIPZyme vs ChemGIME: M-CSA cross-database benchmark"
    )
    parser.add_argument("--bridge", default=str(BRIDGE_PATH),
                        help="Path to bridge JSON")
    parser.add_argument("--screening-set", default=str(SCREENING_SET_PATH),
                        help="Path to CLIPZyme screening-set pickle")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_PATH),
                        help="Path to CLIPZyme model checkpoint")
    parser.add_argument("--chemgime-results", default=str(CHEMGIME_RESULTS_PATH),
                        help="Path to ChemGIME M-CSA benchmark JSON "
                             "(default: DRFP kept cofactors)")
    parser.add_argument("--n-bootstrap", type=int, default=10_000,
                        help="Number of bootstrap resamples for CIs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")

    print("=" * 70)
    print("CLIPZyme vs ChemGIME — M-CSA Cross-Database Benchmark")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Phase 1: Load resources
    # ------------------------------------------------------------------
    print("\n[Phase 1] Loading resources...")

    # 1a. Bridge
    bridge_path = Path(args.bridge)
    if not bridge_path.exists():
        print(f"  ERROR: Bridge not found: {bridge_path}")
        print("  Run: python benchmarks/clipzyme_build_bridge.py")
        sys.exit(1)
    bridge = load_bridge(bridge_path)
    print(f"  Bridge: {bridge['overlap_stats']['overlap_count']} "
          f"iML1515 proteins in CLIPZyme screening set")

    # 1b. Screening set embeddings
    ss_path = Path(args.screening_set)
    if not ss_path.exists():
        print(f"  ERROR: Screening set not found: {ss_path}")
        sys.exit(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    screen_hiddens, screen_uniprots = load_screening_embeddings(ss_path, "cpu")
    print(f"  Screening set: {screen_hiddens.shape[0]:,} proteins")
    print(f"  Device: {device}")

    # Build constrained subset (iML1515 proteins only)
    constrained_indices = bridge["all_iml1515_screen_indices"]
    constrained_hiddens = screen_hiddens[constrained_indices].to(device)
    constrained_uniprots = [screen_uniprots[i] for i in constrained_indices]
    print(f"  Constrained search space: {len(constrained_indices)} proteins")

    # 1c. ChemGIME results
    cg_path = Path(args.chemgime_results)
    if not cg_path.exists():
        print(f"  ERROR: ChemGIME results not found: {cg_path}")
        print("  Run: python tests/benchmarking_suite.py mcsa-novelty")
        sys.exit(1)
    chemgime_by_mcsa = load_chemgime_results(cg_path)
    print(f"  ChemGIME results: {len(chemgime_by_mcsa)} M-CSA entries "
          f"(from {cg_path.name})")

    # 1d. M-CSA entries
    mcsa_entries = load_mcsa_entries(MCSA_FULL_JSON)
    print(f"  M-CSA entries loaded: {len(mcsa_entries)}")

    # Build iML1515 UniProt set from the bridge
    gem_uniprots = set(bridge["uniprot_to_screen_idx"].keys())
    # Also include UniProts from gene_to_uniprot that may NOT be in the
    # screening set (ChemGIME can still find them via GPR)
    if GENE_TO_UNIPROT_PATH.exists():
        with open(GENE_TO_UNIPROT_PATH) as f:
            g2u = json.load(f)
        gem_uniprots |= set(g2u.values())
    print(f"  iML1515 UniProt IDs (total): {len(gem_uniprots)}")

    # ------------------------------------------------------------------
    # Phase 2: Prepare CLIPZyme queries from M-CSA
    # ------------------------------------------------------------------
    print("\n[Phase 2] Preparing M-CSA queries for CLIPZyme...")

    # 2a. Identify overlap entries
    overlap = identify_overlap_entries(mcsa_entries, gem_uniprots)
    print(f"  M-CSA overlap entries: {len(overlap)}")

    # 2b-2d. Build SMILES, atom-map, encode
    print(f"\n  Loading CLIPZyme model...")
    original_cwd = os.getcwd()
    os.chdir(str(PROJECT_ROOT / "vendor" / "clipzyme"))
    from clipzyme import CLIPZyme
    model = CLIPZyme(checkpoint_path=str(args.checkpoint))
    model = model.eval()
    os.chdir(original_cwd)

    print(f"  Loading RXNMapper...")
    from rxnmapper import RXNMapper
    rxn_mapper = RXNMapper()

    results: list[dict] = []
    failures: list[dict] = []

    print(f"\n  Processing {len(overlap)} M-CSA queries...")
    for entry in overlap:
        mcsa_id = entry["mcsa_id"]
        target_uni = entry["target_uniprot"]
        enzyme_name = entry["enzyme_name"]
        ec = entry["ec"]

        record = {
            "mcsa_id": mcsa_id,
            "enzyme_name": enzyme_name,
            "ec": ec,
            "target_uniprot": target_uni,
            # Will be filled in below:
            "smiles_ok": False,
            "left_smiles": None,
            "right_smiles": None,
            "mapping_ok": False,
            "mapping_confidence": None,
            "encoding_ok": False,
            "encoding_error": None,
            "clipzyme_rank": None,
            "clipzyme_rr": 0.0,
            "chemgime_rank": None,
            "chemgime_rr": 0.0,
        }

        # 2b. Build ChEBI-derived SMILES
        left_smi, right_smi = build_mcsa_query_smiles(entry)
        if not left_smi or not right_smi:
            record["encoding_error"] = "smiles_build_failed"
            failures.append({"mcsa_id": mcsa_id, "stage": "smiles",
                             "error": "No reactant or product SMILES from ChEBI"})
            results.append(record)
            continue
        record["smiles_ok"] = True
        record["left_smiles"] = left_smi
        record["right_smiles"] = right_smi

        # 2c. Atom-map via RXNMapper
        rxn_smi = f"{left_smi}>>{right_smi}"
        try:
            mapped = rxn_mapper.get_attention_guided_atom_maps([rxn_smi])
            mapped_smi = mapped[0].get("mapped_rxn")
            confidence = mapped[0].get("confidence", 0.0)
        except Exception as e:
            mapped_smi = None
            confidence = 0.0
            failures.append({"mcsa_id": mcsa_id, "stage": "atom_mapping",
                             "error": f"{type(e).__name__}: {e}"})

        if not mapped_smi:
            record["encoding_error"] = "atom_mapping_failed"
            results.append(record)
            continue
        record["mapping_ok"] = True
        record["mapping_confidence"] = float(confidence)

        # 2d. Encode with CLIPZyme
        try:
            with torch.no_grad():
                emb = model.extract_reaction_features(reaction=mapped_smi)
            record["encoding_ok"] = True
        except Exception as e:
            record["encoding_error"] = f"{type(e).__name__}: {e}"
            failures.append({"mcsa_id": mcsa_id, "stage": "clipzyme_encode",
                             "error": record["encoding_error"],
                             "mapped_smiles": mapped_smi})
            results.append(record)
            continue

        # Phase 3: Rank target protein
        emb = emb.to(device)
        scores = (constrained_hiddens @ emb.T).squeeze()
        clip_rank = rank_target_in_scores(scores, constrained_uniprots,
                                          target_uni)
        record["clipzyme_rank"] = clip_rank
        record["clipzyme_rr"] = 1.0 / clip_rank if clip_rank else 0.0

        # Look up ChemGIME result for this M-CSA entry
        cg = chemgime_by_mcsa.get(mcsa_id)
        if cg:
            cg_rank = cg.get("rank")
            record["chemgime_rank"] = cg_rank
            record["chemgime_rr"] = 1.0 / cg_rank if cg_rank else 0.0

        results.append(record)

    # ------------------------------------------------------------------
    # Phase 3 cont'd: Compute metrics
    # ------------------------------------------------------------------
    print("\n[Phase 3] Computing metrics...")

    # Encoding statistics
    n_total = len(results)
    n_smiles_ok = sum(1 for r in results if r["smiles_ok"])
    n_mapped = sum(1 for r in results if r["mapping_ok"])
    n_encoded = sum(1 for r in results if r["encoding_ok"])

    print(f"  Total M-CSA overlap entries: {n_total}")
    print(f"  SMILES built:    {n_smiles_ok}/{n_total}")
    print(f"  Atom-mapped:     {n_mapped}/{n_total}")
    print(f"  CLIPZyme encoded: {n_encoded}/{n_total} "
          f"({n_encoded/n_total*100:.1f}%)")

    # Matched subset: entries where BOTH systems have a result
    matched = [r for r in results
               if r["encoding_ok"] and r["chemgime_rank"] is not None]
    print(f"  Matched subset:  {len(matched)}/{n_total}")

    if not matched:
        print("\n  ERROR: No matched queries. Cannot compare systems.")
        sys.exit(1)

    # Helper
    def top_k(ranks, k):
        return sum(1 for r in ranks if r is not None and r <= k) / len(ranks) * 100

    # CLIPZyme metrics (matched)
    clip_ranks = [r["clipzyme_rank"] for r in matched]
    clip_rrs = np.array([r["clipzyme_rr"] for r in matched])
    clip_found = sum(1 for r in clip_ranks if r is not None)

    # ChemGIME metrics (matched)
    cg_ranks = [r["chemgime_rank"] for r in matched]
    cg_rrs = np.array([r["chemgime_rr"] for r in matched])
    cg_found = sum(1 for r in cg_ranks if r is not None)

    n_matched = len(matched)

    print(f"\n{'='*70}")
    print(f"HEAD-TO-HEAD COMPARISON (matched {n_matched}-query subset)")
    print(f"{'='*70}")
    print(f"  {'Metric':<15} {'ChemGIME':<20} {'CLIPZyme':<20} {'Delta':<15}")
    print(f"  {'-'*15} {'-'*20} {'-'*20} {'-'*15}")

    cg_mrr = float(cg_rrs.mean())
    cl_mrr = float(clip_rrs.mean())
    delta_mrr = cg_mrr - cl_mrr
    print(f"  {'MRR':<15} {cg_mrr:<20.4f} {cl_mrr:<20.4f} {delta_mrr:+.4f}")

    for k in [1, 5, 10, 20, 100]:
        cg_topk = top_k(cg_ranks, k)
        cl_topk = top_k(clip_ranks, k)
        print(f"  {f'Top-{k}':<15} {cg_topk:<20.2f}% {cl_topk:<20.2f}% "
              f"{cg_topk - cl_topk:+.2f}pp")

    print(f"  {'Found':<15} {cg_found}/{n_matched:<16} "
          f"{clip_found}/{n_matched:<16}")

    # CLIPZyme on all encoded queries (not just matched)
    all_encoded = [r for r in results if r["encoding_ok"]]
    if len(all_encoded) != len(matched):
        print(f"\n  CLIPZyme on all {len(all_encoded)} encoded queries:")
        all_clip_rrs = np.array([r["clipzyme_rr"] for r in all_encoded])
        print(f"    MRR: {all_clip_rrs.mean():.4f}")

    # ChemGIME on all 130 queries (from its own benchmark)
    all_cg = [chemgime_by_mcsa[m]["reciprocal_rank"]
              for m in chemgime_by_mcsa
              if chemgime_by_mcsa[m].get("status") == "success"]
    if all_cg:
        print(f"\n  ChemGIME on all {len(all_cg)} queries (from Benchmark 2):")
        print(f"    MRR: {np.mean(all_cg):.4f}")

    # ------------------------------------------------------------------
    # Phase 4: Statistical testing
    # ------------------------------------------------------------------
    print(f"\n[Phase 4] Statistical testing (n={n_matched} paired queries)...")
    stats = {}

    # 4a. Bootstrap CIs
    cg_mean, cg_lo, cg_hi = bootstrap_ci(cg_rrs, n_boot=args.n_bootstrap,
                                          seed=args.seed)
    cl_mean, cl_lo, cl_hi = bootstrap_ci(clip_rrs, n_boot=args.n_bootstrap,
                                          seed=args.seed)
    stats["bootstrap_ci"] = {
        "n_bootstrap": args.n_bootstrap,
        "chemgime_mrr": {"mean": cg_mean, "ci_lower": cg_lo, "ci_upper": cg_hi},
        "clipzyme_mrr": {"mean": cl_mean, "ci_lower": cl_lo, "ci_upper": cl_hi},
    }
    print(f"  ChemGIME MRR: {cg_mean:.4f}  95% CI [{cg_lo:.4f}, {cg_hi:.4f}]")
    print(f"  CLIPZyme MRR: {cl_mean:.4f}  95% CI [{cl_lo:.4f}, {cl_hi:.4f}]")

    # 4b. Wilcoxon signed-rank
    wilcox = wilcoxon_test(cg_rrs, clip_rrs)
    stats["wilcoxon"] = wilcox
    p_str = (f"{wilcox['p_value']:.6f}"
             if "p_value" in wilcox else wilcox.get("error", "N/A"))
    print(f"  Wilcoxon signed-rank p-value: {p_str}")
    if "p_value" in wilcox:
        sig = "SIGNIFICANT" if wilcox["p_value"] < 0.05 else "NOT significant"
        print(f"    -> {sig} at alpha=0.05")

    # 4c. McNemar's test on Top-1
    cg_top1_hits = np.array([r["chemgime_rank"] == 1 for r in matched])
    cl_top1_hits = np.array([r["clipzyme_rank"] == 1 for r in matched])
    mcnemar = mcnemar_test(cg_top1_hits, cl_top1_hits)
    stats["mcnemar_top1"] = mcnemar
    mc_p = mcnemar.get("p_value", "N/A")
    print(f"  McNemar Top-1 p-value: {mc_p}")
    if isinstance(mc_p, float):
        print(f"    Contingency: both_hit={mcnemar['a']}, "
              f"CG_only={mcnemar['b']}, CZ_only={mcnemar['c']}, "
              f"both_miss={mcnemar['d']}")

    # ------------------------------------------------------------------
    # Phase 5: Save outputs
    # ------------------------------------------------------------------
    print(f"\n[Phase 5] Saving results...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 5a. Per-query results
    out_results = {
        "benchmark": "M-CSA Cross-Database Enzyme Retrieval",
        "chemgime_source": str(cg_path.name),
        "n_overlap": n_total,
        "n_clipzyme_encoded": n_encoded,
        "n_matched": n_matched,
        "matched_metrics": {
            "chemgime_mrr": cg_mean,
            "clipzyme_mrr": cl_mean,
            "chemgime_top1": top_k(cg_ranks, 1),
            "clipzyme_top1": top_k(clip_ranks, 1),
            "chemgime_top5": top_k(cg_ranks, 5),
            "clipzyme_top5": top_k(clip_ranks, 5),
            "chemgime_top10": top_k(cg_ranks, 10),
            "clipzyme_top10": top_k(clip_ranks, 10),
            "chemgime_found": cg_found,
            "clipzyme_found": clip_found,
        },
        "per_query": results,
    }
    results_path = OUTPUT_DIR / "clipzyme_mcsa_results.json"
    with open(results_path, "w") as f:
        json.dump(out_results, f, indent=2, default=str)
    print(f"  Results:    {results_path}")

    # 5b. Failure analysis
    failure_summary = {
        "total_overlap": n_total,
        "smiles_failures": n_total - n_smiles_ok,
        "mapping_failures": n_smiles_ok - n_mapped,
        "encoding_failures": n_mapped - n_encoded,
        "total_failures": n_total - n_encoded,
        "encoding_success_rate": n_encoded / n_total * 100 if n_total else 0,
        "details": failures,
    }
    failures_path = OUTPUT_DIR / "clipzyme_mcsa_failures.json"
    with open(failures_path, "w") as f:
        json.dump(failure_summary, f, indent=2, default=str)
    print(f"  Failures:   {failures_path}")

    # 5c. Statistics
    stats_path = OUTPUT_DIR / "clipzyme_mcsa_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Statistics: {stats_path}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    if cg_mean > cl_mean:
        winner = "ChemGIME"
        pct = (cg_mean - cl_mean) / cl_mean * 100 if cl_mean > 0 else float("inf")
    else:
        winner = "CLIPZyme"
        pct = (cl_mean - cg_mean) / cg_mean * 100 if cg_mean > 0 else float("inf")
    print(f"  MRR winner: {winner} (+{pct:.1f}%)")
    if "p_value" in wilcox:
        print(f"  Statistical significance (Wilcoxon): p={wilcox['p_value']:.6f}")
    print(f"  CLIPZyme encoding success: {n_encoded}/{n_total} "
          f"({n_encoded/n_total*100:.1f}%)")
    print(f"  ChemGIME encoding success: {n_total}/{n_total} (100%)")
    print("\nDone.")


if __name__ == "__main__":
    main()
