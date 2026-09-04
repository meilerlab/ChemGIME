"""
CLIPZyme <-> iML1515 UniProt Bridge Builder
============================================

PURPOSE
-------
CLIPZyme ranks *proteins* (by UniProt ID); ChemGIME ranks *reactions*
(by GEM reaction ID).  To compare both systems on the same ground truth
we need a mapping that connects:

    iML1515 gene ID  -->  UniProt ID  -->  CLIPZyme screening-set index

This script builds that bridge and writes it to a single JSON file that
both ``clipzyme_eval.py`` and ``clipzyme_mcsa_eval.py`` consume.

WHY THIS SCRIPT MUST EXIST
---------------------------
The bridge JSON was previously generated ad-hoc and only the output was
committed.  Without a reproducible script, reviewers cannot verify the
mapping or regenerate it after data updates.  This was flagged as a
critical reproducibility gap in the benchmarking review.

HOW IT WORKS (step by step)
----------------------------
1. Load the iML1515 GEM via cobrapy and extract every gene's UniProt
   annotation.  iML1515 stores these natively in gene.annotation["uniprot"].
   As a fallback, load the pre-computed mapping from
   ``data/database/iML1515_gene_to_uniprot.json``.

2. Load CLIPZyme's pre-computed screening set pickle, which contains
   261,907 (UniProt ID, protein embedding) pairs.

3. For each iML1515 gene with a UniProt ID, check whether that UniProt
   exists in CLIPZyme's screening set.  Record the screening-set index.

4. For each GEM reaction, trace GPR -> genes -> UniProts -> screening
   indices.  This tells us which CLIPZyme proteins correspond to each
   GEM reaction.

5. Write the bridge JSON with all mappings.

USAGE
-----
    # Any Python env with cobra installed (no GPU needed)
    python benchmarks/clipzyme_build_bridge.py

    # With explicit paths
    python benchmarks/clipzyme_build_bridge.py \\
        --gem iML1515.xml \\
        --screening-set vendor/clipzyme/files/clipzyme_screening_set.p \\
        --output benchmarks/data/clipzyme_iml1515_bridge.json

OUTPUT
------
    benchmarks/data/clipzyme_iml1515_bridge.json

    Top-level keys:
      - overlap_stats:             summary counts
      - all_iml1515_screen_indices: list[int] -- screening-set indices for
                                    all iML1515 proteins found in CLIPZyme
      - uniprot_to_screen_idx:     dict[str, int] -- UniProt -> screening idx
      - gene_to_screen:            dict[str, {...}] -- gene -> UniProt + indices
      - reaction_to_proteins:      dict[str, {...}] -- reaction -> GPR + UniProts
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


# ---------------------------------------------------------------------------
# Step 1: Gene -> UniProt mapping
# ---------------------------------------------------------------------------

def load_gene_to_uniprot(gem_path: Path) -> dict[str, str]:
    """Extract gene_id -> UniProt_id from the GEM model.

    Strategy: try native COBRA annotations first (iML1515 has them).
    Fall back to the pre-computed JSON mapping if annotations are absent
    or incomplete (< 100 genes resolved), which handles GEMs like iJN746.
    """
    import cobra

    model = cobra.io.read_sbml_model(str(gem_path))
    gene_to_uni: dict[str, str] = {}

    # Try native annotations
    for gene in model.genes:
        ann = getattr(gene, "annotation", {}) or {}
        val = ann.get("uniprot", [])
        if isinstance(val, str) and val:
            gene_to_uni[gene.id] = val
        elif isinstance(val, list) and val:
            gene_to_uni[gene.id] = val[0]

    if len(gene_to_uni) >= 100:
        print(f"  Gene->UniProt from COBRA annotations: {len(gene_to_uni)}")
        return gene_to_uni, model

    # Fallback: pre-computed mapping
    fallback = PROJECT_ROOT / "data" / "database" / "iML1515_gene_to_uniprot.json"
    if fallback.exists():
        print(f"  COBRA annotations insufficient ({len(gene_to_uni)}), "
              f"loading fallback: {fallback.name}")
        with open(fallback) as f:
            gene_to_uni = json.load(f)
    else:
        print(f"  WARNING: No fallback mapping at {fallback}")

    return gene_to_uni, model


# ---------------------------------------------------------------------------
# Step 2: Load CLIPZyme screening set
# ---------------------------------------------------------------------------

def load_screening_set(path: Path) -> tuple[list[str], int]:
    """Load UniProt IDs from CLIPZyme's pre-computed screening set.

    We only need the UniProt list (not the embeddings) to build the bridge.
    Returns (uniprots_list, total_count).
    """
    with open(path, "rb") as f:
        ss = pickle.load(f)

    uniprots = ss["uniprots"]
    if hasattr(uniprots, "tolist"):
        uniprots = uniprots.tolist()

    return uniprots, len(uniprots)


# ---------------------------------------------------------------------------
# Step 3: Build the overlap mapping
# ---------------------------------------------------------------------------

def build_uniprot_to_screen_idx(
    gene_to_uni: dict[str, str],
    screen_uniprots: list[str],
) -> dict[str, int]:
    """For each iML1515 UniProt, find its index in CLIPZyme's screening set.

    The screening set has 261K entries.  Building a reverse lookup (UniProt
    -> index) once is O(N); scanning per gene would be O(N*M).
    """
    # Build reverse index: UniProt -> screening-set index
    # Some UniProts appear multiple times; take the first occurrence
    screen_lookup: dict[str, int] = {}
    for idx, uni in enumerate(screen_uniprots):
        if uni not in screen_lookup:
            screen_lookup[uni] = idx

    # Intersect with iML1515 UniProts
    iml1515_uniprots = set(gene_to_uni.values())
    overlap: dict[str, int] = {}
    for uni in iml1515_uniprots:
        if uni in screen_lookup:
            overlap[uni] = screen_lookup[uni]

    return overlap


# ---------------------------------------------------------------------------
# Step 4: Map reactions -> proteins via GPR
# ---------------------------------------------------------------------------

def extract_genes_from_gpr(gpr_str: str) -> list[str]:
    """Extract E. coli gene IDs (bNNNN pattern) from a GPR string.

    GPR rules use boolean logic: ``(b0001 and b0002) or b0003``.
    We extract all gene identifiers regardless of AND/OR structure,
    because for retrieval ground truth we consider ANY gene match a hit.
    """
    if not gpr_str:
        return []
    return sorted(set(re.findall(r'\b(b\d{4})\b', gpr_str)))


def build_reaction_to_proteins(
    model,
    gene_to_uni: dict[str, str],
    uni_to_screen: dict[str, int],
) -> dict[str, dict]:
    """For each GEM reaction with a GPR, collect UniProt + screening indices.

    This mapping is used by downstream evaluation scripts to determine
    which CLIPZyme proteins correspond to each GEM reaction.
    """
    rxn_map: dict[str, dict] = {}

    for rxn in model.reactions:
        gpr = rxn.gene_reaction_rule.strip()
        if not gpr:
            continue

        genes = extract_genes_from_gpr(gpr)
        if not genes:
            continue

        uniprots = []
        screen_indices = []
        covered_genes = []

        for g in genes:
            uni = gene_to_uni.get(g)
            if uni and uni in uni_to_screen:
                covered_genes.append(g)
                uniprots.append(uni)
                screen_indices.append(uni_to_screen[uni])

        rxn_map[rxn.id] = {
            "gpr": gpr,
            "all_genes": genes,
            "covered_genes": covered_genes,
            "uniprots": uniprots,
            "screening_indices": screen_indices,
            "has_coverage": len(covered_genes) > 0,
        }

    return rxn_map


# ---------------------------------------------------------------------------
# Step 5: Assemble and write
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build the iML1515 <-> CLIPZyme UniProt bridge mapping"
    )
    parser.add_argument(
        "--gem",
        default=str(PROJECT_ROOT / "iML1515.xml"),
        help="Path to iML1515 SBML model (default: %(default)s)",
    )
    parser.add_argument(
        "--screening-set",
        default=str(PROJECT_ROOT / "vendor" / "clipzyme" / "files" /
                     "clipzyme_screening_set.p"),
        help="Path to CLIPZyme screening-set pickle (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default=str(SCRIPT_DIR / "data" / "clipzyme_iml1515_bridge.json"),
        help="Output bridge JSON path (default: %(default)s)",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("CLIPZyme <-> iML1515 UniProt Bridge Builder")
    print("=" * 65)

    # -- Step 1: Gene -> UniProt --
    print("\n[1/5] Loading GEM and extracting gene->UniProt mapping...")
    gem_path = Path(args.gem)
    if not gem_path.exists():
        print(f"  ERROR: GEM file not found: {gem_path}")
        sys.exit(1)
    gene_to_uni, model = load_gene_to_uniprot(gem_path)
    print(f"  Genes with UniProt: {len(gene_to_uni)}")
    print(f"  Unique UniProt IDs: {len(set(gene_to_uni.values()))}")

    # -- Step 2: Load screening set --
    print("\n[2/5] Loading CLIPZyme screening set...")
    ss_path = Path(args.screening_set)
    if not ss_path.exists():
        print(f"  ERROR: Screening set not found: {ss_path}")
        print("  Download from: https://zenodo.org/records/15161343")
        sys.exit(1)
    screen_uniprots, screen_size = load_screening_set(ss_path)
    print(f"  Screening set size: {screen_size:,}")

    # -- Step 3: Build overlap --
    print("\n[3/5] Computing UniProt overlap...")
    uni_to_screen = build_uniprot_to_screen_idx(gene_to_uni, screen_uniprots)
    iml1515_uni_count = len(set(gene_to_uni.values()))
    overlap_pct = len(uni_to_screen) / iml1515_uni_count * 100
    print(f"  iML1515 UniProt IDs: {iml1515_uni_count}")
    print(f"  Found in screening set: {len(uni_to_screen)} ({overlap_pct:.1f}%)")

    # -- Step 4: Map reactions -> proteins --
    print("\n[4/5] Mapping reactions -> proteins via GPR...")
    rxn_map = build_reaction_to_proteins(model, gene_to_uni, uni_to_screen)
    n_with_coverage = sum(1 for v in rxn_map.values() if v["has_coverage"])
    print(f"  Reactions with GPR: {len(rxn_map)}")
    print(f"  Reactions with screening coverage: {n_with_coverage}")

    # -- Step 5: Build gene-level mapping and write --
    print("\n[5/5] Writing bridge file...")
    gene_to_screen: dict[str, dict] = {}
    for gene_id, uni in gene_to_uni.items():
        if uni in uni_to_screen:
            gene_to_screen[gene_id] = {
                "uniprot": uni,
                "screening_indices": [uni_to_screen[uni]],
            }

    bridge = {
        "overlap_stats": {
            "iml1515_uniprot_count": iml1515_uni_count,
            "screening_set_size": screen_size,
            "overlap_count": len(uni_to_screen),
            "overlap_pct": round(overlap_pct, 1),
            "test_reactions_with_coverage": n_with_coverage,
            "test_reactions_total": len(rxn_map),
        },
        "all_iml1515_screen_indices": sorted(uni_to_screen.values()),
        "uniprot_to_screen_idx": uni_to_screen,
        "gene_to_screen": gene_to_screen,
        "reaction_to_proteins": rxn_map,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(bridge, f, indent=2)

    print(f"\n  Bridge written to: {out_path}")
    print(f"  Overlap: {len(uni_to_screen)} / {iml1515_uni_count} "
          f"iML1515 proteins ({overlap_pct:.1f}%)")
    print("\nDone.")


if __name__ == "__main__":
    main()
