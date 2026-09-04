"""
Prepare Atom-Mapped Reaction SMILES for CLIPZyme
=================================================

Converts ChemGIME's dot-separated unmapped SMILES (left_smiles / right_smiles)
to atom-mapped reaction SMILES required by CLIPZyme's DMPNN reaction encoder.

Uses RXNMapper (Schwaller et al. 2021) for attention-guided atom mapping.

Usage:
    # Run in clipzyme conda env
    python benchmarks/clipzyme_prepare_reactions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

TEST_CSV = SCRIPT_DIR / "data" / "iML1515_test.csv"
OUTPUT_CSV = SCRIPT_DIR / "data" / "iML1515_test_atommapped.csv"


def main():
    print("=" * 60)
    print("Atom-Mapping iML1515 Test Reactions for CLIPZyme")
    print("=" * 60)

    test_df = pd.read_csv(TEST_CSV)
    print(f"Test reactions: {len(test_df)}")

    # Initialize RXNMapper
    print("\nLoading RXNMapper...")
    from rxnmapper import RXNMapper
    rxn_mapper = RXNMapper()

    # Convert each reaction
    results = []
    success = 0
    failed = 0

    # Build reaction SMILES in format: reactants>>products
    # ChemGIME uses dot-separated SMILES for left and right
    rxn_smiles_list = []
    for _, row in test_df.iterrows():
        left = str(row["left_smiles"]).strip()
        right = str(row["right_smiles"]).strip()
        # RXNMapper expects reactants>>products with dots between molecules
        rxn_smi = f"{left}>>{right}"
        rxn_smiles_list.append(rxn_smi)

    # Batch process for efficiency
    print(f"\nMapping {len(rxn_smiles_list)} reactions...")
    batch_size = 10
    all_mapped = []

    for i in tqdm(range(0, len(rxn_smiles_list), batch_size), desc="  Batches"):
        batch = rxn_smiles_list[i:i + batch_size]
        try:
            mapped = rxn_mapper.get_attention_guided_atom_maps(batch)
            all_mapped.extend(mapped)
        except Exception as e:
            # Try one-by-one for failed batches
            for rxn in batch:
                try:
                    mapped = rxn_mapper.get_attention_guided_atom_maps([rxn])
                    all_mapped.extend(mapped)
                except Exception:
                    all_mapped.append({"mapped_rxn": None, "confidence": 0.0})

    # Build output
    for idx, (_, row) in enumerate(test_df.iterrows()):
        mapped_info = all_mapped[idx] if idx < len(all_mapped) else {"mapped_rxn": None, "confidence": 0.0}
        mapped_rxn = mapped_info.get("mapped_rxn")
        confidence = mapped_info.get("confidence", 0.0)

        if mapped_rxn:
            success += 1
        else:
            failed += 1

        results.append({
            "reaction_id": row["reaction_id"],
            "left_smiles": row["left_smiles"],
            "right_smiles": row["right_smiles"],
            "true_gpr": row.get("true_gpr", ""),
            "true_ec": row.get("true_ec", ""),
            "mapped_smiles": mapped_rxn or "",
            "mapping_confidence": confidence,
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nResults:")
    print(f"  Success: {success}/{len(test_df)} ({success/len(test_df)*100:.1f}%)")
    print(f"  Failed:  {failed}/{len(test_df)}")
    print(f"  Mean confidence: {out_df[out_df['mapped_smiles'] != '']['mapping_confidence'].mean():.4f}")
    print(f"\nSaved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
