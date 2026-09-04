import pandas as pd
import sys

# --- 1. Define File Names and Columns ---
CHEM_XREF_FILE = 'chem_xref.tsv'
CHEM_PROP_FILE = 'chem_prop.tsv'
OUTPUT_FILE = 'bigg_to_smiles.tsv'

# Columns to extract from chem_prop.tsv
# Standard MetaNetX headers are: #MNX_ID, name, reference, formula, charge, ... , SMILES
PROP_COLS_TO_USE = ['ID', 'formula', 'charge', 'SMILES']
PROP_COL_RENAME = {
    'ID': 'mnx_id',
    'SMILES': 'smiles'
    # 'formula' and 'charge' are already lowercase
}

# Columns to extract from chem_xref.tsv
# Standard MetaNetX headers are: #XREF, MNX_ID, source, description
XREF_COLS_TO_USE = ['source', 'ID']
XREF_COL_RENAME = {
    'source': 'xref',
    'ID': 'mnx_id'
}

try:
    # --- 2. Load chem_prop.tsv (Properties Table) ---
    # We load only the columns we need and handle the '#' comment character
    print(f"Loading {CHEM_PROP_FILE}...")
    props_df = pd.read_csv(
        CHEM_PROP_FILE,
        sep='\t',
        comment='#',
        usecols=PROP_COLS_TO_USE
    )
    # Rename columns for merging and for the final output
    props_df.rename(columns=PROP_COL_RENAME, inplace=True)

    # --- 3. Load chem_xref.tsv (Cross-Reference Table) ---
    print(f"Loading {CHEM_XREF_FILE}...")
    xref_df = pd.read_csv(
        CHEM_XREF_FILE,
        sep='\t',
        comment='#',
        usecols=XREF_COLS_TO_USE
    )
    xref_df.rename(columns=XREF_COL_RENAME, inplace=True)

except FileNotFoundError as e:
    print(f"Error: File not found ({e.filename}).")
    print("Please make sure 'chem_xref.tsv' and 'chem_prop.tsv' are in the same directory.")
    sys.exit(1)
except ValueError as e:
    print(f"Error loading file. Check if columns are missing/named incorrectly.")
    print(f"Details: {e}")
    sys.exit(1)


# --- 4. Filter for BiGG IDs ---
# Per your note: "check for bigg:, bigg.metabolite:, or other... variations"
# We use .startswith('bigg') to catch all relevant prefixes
print("Filtering for BiGG identifiers...")
bigg_df = xref_df[xref_df['xref'].str.startswith('bigg', na=False)].copy()

# --- 5. Clean BiGG ID Column ---
# The 'xref' column is 'bigg:glc__D'. We create a 'bigg_id' column with 'glc__D'.
# This regex removes 'bigg:' or 'bigg.metabolite:' from the start.
bigg_df['bigg_id'] = bigg_df['xref'].str.replace(r'^bigg(\.metabolite)?:', '', regex=True)

# Keep only the columns needed for the join
bigg_df = bigg_df[['bigg_id', 'mnx_id']]

# --- 6. Perform the JOIN Operation ---
# We use a 'left' join starting from our BiGG list (bigg_df).
# This correctly handles "Gotcha 2.3.2" (Non-Chemical Entities).
# e.g., 'biomass' (BIOMASS) will be kept, even if its properties are 'null'.
print("Joining BiGG IDs with chemical properties...")
final_df = pd.merge(bigg_df, props_df, on='mnx_id', how='left')

# --- 7. Finalize and Save ---
# Re-order columns to match your example output table
final_cols = ['bigg_id', 'mnx_id', 'smiles', 'formula', 'charge']
final_df = final_df[final_cols]

# Save the result as a TSV file
print(f"Saving results to {OUTPUT_FILE}...")
final_df.to_csv(OUTPUT_FILE, sep='\t', index=False, na_rep='null')

print("\nDone. 'bigg_to_smiles.tsv' has been created successfully.")
