import os
import csv
from collections import Counter
import re

# Configuration
PREDICTIONS_DIR = "/media/data/dev/microbiomeML/paper_metabolism/ChemGIME/predictions"
OPTIMAL_THRESHOLD = 0.55
TARGET_DRUGS = ["budesonid", "salmeterol"] # Substrings to match in filename

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
                    # Check for EC. If not in CSV, we can try to infer from Name (weak)
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
    """Extracts strain name from path if present."""
    parts = path.split(os.sep)
    for p in parts:
        if "-all-Reactions" in p:
            return p.replace("-all-Reactions", "").replace("_", " ")
    return None

def main():
    print(f"Analyzing predictions in {PREDICTIONS_DIR}...")
    
    all_rows = []
    file_count = 0
    
    for root, dirs, files in os.walk(PREDICTIONS_DIR):
        strain_name = get_strain_from_path(root)
        if not strain_name:
            continue
            
        for file in files:
            if file.endswith("_results.csv") and any(d in file.lower() for d in TARGET_DRUGS):
                filepath = os.path.join(root, file)
                file_count += 1
                reactions = parse_csv_report(filepath)
                
                for r in reactions:
                    r['strain'] = strain_name
                    # r['source_file'] = file
                    all_rows.append(r)

    print(f"Total Files Processed: {file_count}")
    print(f"Total Rows Extracted: {len(all_rows)}")
    
    if not all_rows:
        print("No data found.")
        return

    # Filter by threshold
    candidates_below_threshold = [r for r in all_rows if r['tanimoto'] <= OPTIMAL_THRESHOLD]
    
    # Calculate Unique Candidates (Accession + Strain)
    # Using Set to count unique proteins per strain
    unique_set = set()
    candidates_by_strain = Counter()
    
    final_candidates = []
    
    for c in candidates_below_threshold:
        key = (c['strain'], c['accession'])
        if key not in unique_set:
            unique_set.add(key)
            candidates_by_strain[c['strain']] += 1
            final_candidates.append(c)

    print("-" * 30)
    print(f"Total Unique Candidates [N] (<= {OPTIMAL_THRESHOLD}): {len(final_candidates)}")
    print("Breakdown by Strain:")
    for strain in sorted(candidates_by_strain.keys()):
        print(f"  {strain}: {candidates_by_strain[strain]}")
    
    # Global metrics
    min_score = min(r['tanimoto'] for r in all_rows)
    print(f"Global Best Score (Min Tanimoto): {min_score}")
    
    min_score_threshold = min(r['tanimoto'] for r in final_candidates) if final_candidates else "N/A"
    print(f"Best Score in Candidates (<= {OPTIMAL_THRESHOLD}): {min_score_threshold}")

    # Homology Analysis on Final Candidates
    homology_cutoff = 1e-50 # Strict
    passed_homology = sum(1 for c in final_candidates if c['evalue'] < homology_cutoff)
    total_final = len(final_candidates)
    if total_final > 0:
        print(f"Homology < {homology_cutoff}: {passed_homology} / {total_final} ({passed_homology/total_final*100:.1f}%)")
    else:
        print(f"Homology < {homology_cutoff}: 0/0")

    # Dominant Enzyme Family/Class (simple heuristic based on name)
    # e.g. "oxidoreductase", "transferase", "hydrolase"
    if final_candidates:
        class_keywords = ["oxidoreductase", "transferase", "hydrolase", "lyase", "isomerase", "ligase", "translocase", "synthase", "dehydrogenase", "kinase"]
        class_counts = Counter()
        for c in final_candidates:
            name_lower = c['name'].lower()
            found = False
            for kw in class_keywords:
                if kw in name_lower:
                    class_counts[kw] += 1
                    found = True
                    # break # Count all matches? or just first? Usually one main class.
            if not found:
                 class_counts["other"] += 1
        
        print("Top Enzyme Classes:")
        for k, v in class_counts.most_common(5):
            print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
