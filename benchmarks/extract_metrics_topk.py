import os
import csv
from collections import Counter
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

def main():
    print(f"Analyzing predictions in {PREDICTIONS_DIR}...")
    
    file_count = 0
    all_selected_candidates = []
    
    candidates_by_strain = Counter()
    
    for root, dirs, files in os.walk(PREDICTIONS_DIR):
        strain_name = get_strain_from_path(root)
        if not strain_name:
            continue
            
        for file in files:
            if file.endswith("_results.csv") and any(d in file.lower() for d in TARGET_DRUGS):
                filepath = os.path.join(root, file)
                file_count += 1
                reactions = parse_csv_report(filepath)
                reactions.sort(key=lambda x: x['tanimoto'])
                top_k = reactions[:TOP_K]
                for r in top_k:
                    r['strain'] = strain_name
                    all_selected_candidates.append(r)

    print(f"Total Files Processed: {file_count}")
    
    unique_map = {}
    for c in all_selected_candidates:
        key = (c['strain'], c['accession'])
        if key not in unique_map:
            unique_map[key] = c
            candidates_by_strain[c['strain']] += 1
            
    final_candidates = list(unique_map.values())

    print("-" * 30)
    print(f"Total Unique Candidates [N] (Top {TOP_K}): {len(final_candidates)}")
    print("Breakdown by Strain:")
    for strain in sorted(candidates_by_strain.keys()):
        print(f"  {strain}: {candidates_by_strain[strain]}")
    
    # Best Score per Strain
    best_by_strain = {}
    for c in final_candidates:
        s = c['strain']
        if s not in best_by_strain or c['tanimoto'] < best_by_strain[s]:
            best_by_strain[s] = c['tanimoto']
            
    print("Best Score by Strain:")
    for s, score in best_by_strain.items():
        print(f"  {s}: {score}")

    # Homology Analysis
    homology_cutoff = 1e-50
    passed_homology = sum(1 for c in final_candidates if c['evalue'] < homology_cutoff)
    total_final = len(final_candidates)
    percentage = (passed_homology / total_final * 100) if total_final > 0 else 0
    print(f"Homology < {homology_cutoff}: {passed_homology} / {total_final} ({percentage:.1f}%)")

if __name__ == "__main__":
    main()
