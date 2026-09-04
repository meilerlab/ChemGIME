import os
import csv
import re

# Configuration
PREDICTIONS_DIR = "/media/data/dev/microbiomeML/paper_metabolism/ChemGIME/predictions"
TARGET_DRUGS = ["budesonid", "salmeterol"]
TOP_K = 5

def parse_csv_report(filepath):
    """Parses a single CSV report to extract reaction data."""
    rows = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    clean_row = {k.strip(): v for k, v in row.items() if k}
                    
                    if "Tanimoto Distance" not in clean_row:
                        continue
                        
                    dist = float(clean_row["Tanimoto Distance"])
                    name = clean_row.get("Enzyme Name", "Unknown")
                    accession = clean_row.get("Accession_ID", "")
                    evalue = clean_row.get("E-Value", "1.0")
                    ec = clean_row.get("EC Number", "N/A")
                    
                    rows.append({
                        'tanimoto': dist,
                        'name': name,
                        'accession': accession,
                        'evalue': float(evalue) if evalue and evalue.lower() != 'nan' else 1.0,
                        'ec': ec
                    })
                except ValueError:
                    continue
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        
    return rows

def get_strain_from_path(path):
    parts = path.split(os.sep)
    for p in parts:
        if "-all-Reactions" in p:
            return p.replace("-all-Reactions", "").replace("_", " ")
    return None

def clean_filename(filename):
    """Simplifies filename to represent the reaction/drug."""
    name = filename.replace("_results.csv", "").replace("report_", "")
    return name

def main():
    all_selected_candidates = []
    
    for root, dirs, files in os.walk(PREDICTIONS_DIR):
        strain_name = get_strain_from_path(root)
        if not strain_name:
            continue
            
        for file in files:
            if file.endswith("_results.csv") and any(d in file.lower() for d in TARGET_DRUGS):
                filepath = os.path.join(root, file)
                reactions = parse_csv_report(filepath)
                
                # Sort by Tanimoto (ascending, lower is better?)
                # Actually, Tanimoto Distance: 0 is identical, 1 is different. So Ascending.
                reactions.sort(key=lambda x: x['tanimoto'])
                
                # Take Top K
                top_k = reactions[:TOP_K]
                
                for r in top_k:
                    r['strain'] = strain_name
                    r['reaction'] = clean_filename(file)
                    all_selected_candidates.append(r)

    # Dedup
    unique_map = {}
    for c in all_selected_candidates:
        key = (c['strain'], c['accession'])
        if key not in unique_map:
            unique_map[key] = c
            
    final_candidates = list(unique_map.values())
    
    # Sort for table: Strain, then Tanimoto
    final_candidates.sort(key=lambda x: (x['strain'], x['tanimoto']))

    print("| Strain | Accession ID | Enzyme Name | Reaction Context | Tanimoto Distance | Homology E-Value |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for c in final_candidates:
        evalue_str = f"{c['evalue']:.2e}" if c['evalue'] < 0.01 else f"{c['evalue']:.4f}"
        print(f"| *{c['strain']}* | {c['accession']} | {c['name']} | {c['reaction']} | {c['tanimoto']:.4f} | {evalue_str} |")

if __name__ == "__main__":
    main()
