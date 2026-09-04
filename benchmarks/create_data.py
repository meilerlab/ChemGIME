"""
Split-GEM Benchmark Data Creation Script
=========================================

This script creates the benchmark dataset for evaluating ChemGIME's 
GPR (Gene-Protein-Reaction) prediction capability using a "Split-GEM" approach.

The Split-GEM Methodology:
--------------------------
1. Load a complete, well-annotated GEM (E. coli iML1515).
2. Filter reactions to keep only those with valid GPRs and SMILES.
3. Randomly split these reactions into:
   - Database Set (80%): Used as the search space for ChemGIME.
   - Query Set (20%): Used as test queries with ground-truth GPRs.
4. Save the database as a valid SBML file and queries as a CSV.

This allows us to measure how accurately ChemGIME can retrieve the correct 
enzyme/gene association given only the chemical transformation.
"""

import argparse
import random
from pathlib import Path

import cobra
import pandas as pd
from tqdm import tqdm

# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
CHEMGIME_ROOT = SCRIPT_DIR.parent
DATA_DIR = SCRIPT_DIR / "data"
GEM_PATH = CHEMGIME_ROOT / "data" / "iML1515.xml"
BIGG_TO_SMILES_PATH = CHEMGIME_ROOT / "data" / "database" / "bigg_to_smiles.tsv"

TRAIN_RATIO = 0.80
RANDOM_SEED = 42


def load_bigg_to_smiles_map(map_file: Path) -> dict:
    """Loads the BiGG metabolite ID to SMILES mapping."""
    print(f"Loading BiGG-to-SMILES map from: {map_file}")
    df = pd.read_csv(map_file, sep='\t', dtype=str, na_values=['null', 'nan', ''])
    
    # Create a mapping from bigg_id -> smiles
    # Filter out entries with null/empty smiles
    df = df.dropna(subset=['smiles'])
    df = df[df['smiles'] != 'null']
    
    bigg_map = {}
    for _, row in df.iterrows():
        bigg_id = row['bigg_id']
        smiles = row['smiles']
        # Handle prefixed IDs like 'biggM:M_atp' or 'biggM:atp'
        # We want to map 'atp_c', 'atp_e', etc. (with compartment suffixes)
        # The TSV contains base IDs like 'atp', 'biggM:atp', 'biggM:M_atp'
        # We need to extract the base ID
        if bigg_id.startswith('biggM:M_'):
            base_id = bigg_id[8:]  # Remove 'biggM:M_'
        elif bigg_id.startswith('biggM:'):
            base_id = bigg_id[6:]  # Remove 'biggM:'
        else:
            base_id = bigg_id
        
        if base_id not in bigg_map:
            bigg_map[base_id] = smiles
    
    print(f"  Loaded {len(bigg_map)} unique metabolite mappings.")
    return bigg_map


def get_smiles_for_metabolite(met_id: str, bigg_map: dict) -> str | None:
    """
    Get SMILES for a metabolite ID.
    Metabolite IDs in models often have compartment suffixes (e.g., 'atp_c').
    """
    # Remove compartment suffix
    base_id = met_id.rsplit('_', 1)[0] if '_' in met_id else met_id
    return bigg_map.get(base_id)


def build_reaction_smiles(rxn: 'cobra.Reaction', bigg_map: dict) -> tuple:
    """
    Build dot-separated SMILES for reactants and products.
    Returns (left_smiles, right_smiles) or (None, None) if incomplete.
    """
    left_smiles_list = []
    right_smiles_list = []
    
    for met in rxn.reactants:
        smiles = get_smiles_for_metabolite(met.id, bigg_map)
        if smiles:
            left_smiles_list.append(smiles)
    
    for met in rxn.products:
        smiles = get_smiles_for_metabolite(met.id, bigg_map)
        if smiles:
            right_smiles_list.append(smiles)
    
    # Require at least one SMILES on each side
    if not left_smiles_list or not right_smiles_list:
        return None, None
    
    return '.'.join(left_smiles_list), '.'.join(right_smiles_list)


def filter_valid_reactions(model: 'cobra.Model', bigg_map: dict) -> list:
    """
    Filter reactions to keep only those suitable for benchmarking.
    Criteria:
    - Has a non-empty GPR
    - Is not a boundary/exchange reaction
    - Has valid SMILES for at least one reactant and one product
    """
    valid_reactions = []
    
    for rxn in tqdm(model.reactions, desc="Filtering reactions"):
        # Skip boundary/exchange reactions
        if rxn.boundary:
            continue
        
        # Skip reactions without GPR
        if not rxn.gene_reaction_rule or rxn.gene_reaction_rule.strip() == '':
            continue
        
        # Check if we can build SMILES
        left_smiles, right_smiles = build_reaction_smiles(rxn, bigg_map)
        if left_smiles is None:
            continue
        
        valid_reactions.append({
            'reaction': rxn,
            'left_smiles': left_smiles,
            'right_smiles': right_smiles,
            'gpr': rxn.gene_reaction_rule,
            'ec': rxn.annotation.get('ec-code', ''),
        })
    
    return valid_reactions


def create_split_model(model: 'cobra.Model', reaction_ids: set) -> 'cobra.Model':
    """
    Create a new model containing only the specified reactions.
    """
    new_model = cobra.Model(model.id + "_train")
    original_name = model.name if model.name else model.id
    new_model.name = str(original_name) + " (Training Subset)"
    
    # Add reactions to the new model
    reactions_to_add = [rxn.copy() for rxn in model.reactions if rxn.id in reaction_ids]
    new_model.add_reactions(reactions_to_add)
    
    return new_model


def main():
    parser = argparse.ArgumentParser(description="Create Split-GEM benchmark data")
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, help='Random seed')
    parser.add_argument('--train-ratio', type=float, default=TRAIN_RATIO, 
                        help='Ratio of reactions for training (default: 0.80)')
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # Create output directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load the BiGG-to-SMILES map
    bigg_map = load_bigg_to_smiles_map(BIGG_TO_SMILES_PATH)
    
    # Step 2: Load the GEM
    print(f"\nLoading GEM from: {GEM_PATH}")
    model = cobra.io.read_sbml_model(str(GEM_PATH))
    print(f"  Model: {model.id} with {len(model.reactions)} reactions")
    
    # Step 3: Filter valid reactions
    print("\nFiltering reactions...")
    valid_reactions = filter_valid_reactions(model, bigg_map)
    print(f"  Found {len(valid_reactions)} valid reactions (with GPR and SMILES)")
    
    # Step 4: Shuffle and split
    random.shuffle(valid_reactions)
    split_idx = int(len(valid_reactions) * args.train_ratio)
    train_reactions = valid_reactions[:split_idx]
    test_reactions = valid_reactions[split_idx:]
    
    print(f"\nSplit: {len(train_reactions)} training, {len(test_reactions)} test reactions")
    
    # Step 5: Create training model (80%)
    train_rxn_ids = {r['reaction'].id for r in train_reactions}
    train_model = create_split_model(model, train_rxn_ids)
    
    train_model_path = DATA_DIR / "iML1515_train.xml"
    print(f"\nSaving training model to: {train_model_path}")
    cobra.io.write_sbml_model(train_model, str(train_model_path))
    print(f"  Training model has {len(train_model.reactions)} reactions")
    
    # Step 6: Create test CSV (20%)
    test_data = []
    for r in test_reactions:
        ec = r['ec']
        if isinstance(ec, list):
            ec = ';'.join(ec)
        test_data.append({
            'reaction_id': r['reaction'].id,
            'left_smiles': r['left_smiles'],
            'right_smiles': r['right_smiles'],
            'true_gpr': r['gpr'],
            'true_ec': ec,
        })
    
    test_df = pd.DataFrame(test_data)
    test_csv_path = DATA_DIR / "iML1515_test.csv"
    print(f"\nSaving test queries to: {test_csv_path}")
    test_df.to_csv(test_csv_path, index=False)
    print(f"  Test set has {len(test_df)} queries")
    
    print("\n✅ Benchmark data creation complete!")
    print(f"   Training GEM: {train_model_path}")
    print(f"   Test Queries: {test_csv_path}")


if __name__ == "__main__":
    main()
