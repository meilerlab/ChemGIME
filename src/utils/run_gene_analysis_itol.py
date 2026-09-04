
"""
Gene Similarity Analysis Script using BLAST for Local Execution.

This script recursively searches a top-level directory for subdirectories
containing HTML files. For each HTML file found, it creates a dedicated output
folder that mirrors the input structure and performs the following steps:

1.  Parses an HTML report's 'Top Similar Reactions' table to extract a list
    of enzymes and their chemical similarity (Tanimoto Distance).
2.  For EACH enzyme in the table, it fetches the corresponding protein sequence
    and its length from NCBI.
3.  Creates a local BLAST database from a directory of user-provided FNA files
    (this is done only once). It also records the length of each translated protein.
4.  Searches EACH fetched protein sequence against the local BLAST database
    using 'blastp' to find its best sequence-similar match.
5.  Merges the BLAST results (sequence similarity) with the Tanimoto data
    (chemical similarity) and the collected sequence lengths.
6.  Saves a final summary table (.csv) and a clustered heatmap (.png)
    into the dedicated output folder for that HTML file.

Example Input Structure:
/path/to/html/
├── experiment_A/
│   └── report1.html
│   └── report2.html
└── experiment_B/
    └── report3.html

Example Output Structure:
/path/to/outputs/
├── experiment_A/
│   ├── report1_heatmap.png
│   ├── report2_heatmap.png
│   └── analysis_data/
│       ├── report1_results.csv
│       ├── report1_query.fasta
│       └── ...
└── experiment_B/
    └── ...


Prerequisites:
- NCBI BLAST+ must be installed and accessible in the system's PATH.
  - macOS: `brew install blast`
  - Linux: `sudo apt-get install ncbi-blast+`
- Python libraries must be installed:
  - `pip install biopython beautifulsoup4 pandas matplotlib seaborn scipy`

Usage:
  python run_gene_analysis.py --html-dir /path/to/html --fna-dir /path/to/fna [--output-dir /path/to/outputs]

"""

import os
import sys
import subprocess
import argparse
import re
import shutil
import numpy as np
from bs4 import BeautifulSoup
from io import StringIO
from Bio import SeqIO, Entrez
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist

# NCBI requires a valid email for Entrez API access.
# Read from NCBI_EMAIL env-var to avoid hardcoding credentials.
Entrez.email = os.environ.get("NCBI_EMAIL", "")
if not Entrez.email:
    print(
        "⚠️  Warning: NCBI_EMAIL environment variable is not set.\n"
        "   NCBI may throttle or block your requests.\n"
        "   Set it with: export NCBI_EMAIL=your.real@email.com",
        file=sys.stderr,
    )

def check_blast_installed():
    """Checks if BLAST+ tools are installed and in the system's PATH."""
    if not shutil.which("makeblastdb") or not shutil.which("blastp"):
        print("Error: NCBI BLAST+ is not installed or not in your system's PATH.", file=sys.stderr)
        sys.exit(1)
    print("✅ NCBI BLAST+ installation found.")

def parse_single_html_report(html_filepath):
    """
    Parses a single HTML file to extract each enzyme from the 'Top Similar Reactions' table.
    MODIFICATION: Now searches all columns for the Accession ID to be more robust.
    """
    print(f"\n🔎 Parsing HTML file: '{os.path.basename(html_filepath)}'...")
    reactions_data = []
    accession_pattern = re.compile(r'WP_\d+\.\d+')

    with open(html_filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    soup = BeautifulSoup(content, 'html.parser')

    header = soup.find(lambda tag: tag.name in ['h1', 'h2', 'h3', 'h4'] and 'Top Similar Reactions' in tag.get_text())
    if not header:
        print(f"     ⚠️ Warning: Could not find the 'Top Similar Reactions' table. Skipping.")
        return pd.DataFrame()

    table = header.find_next('table')
    if not table:
        print(f"     ⚠️ Warning: Found the 'Top Similar Reactions' header but no subsequent table. Skipping.")
        return pd.DataFrame()

    for row in table.find_all('tr')[1:]:
        cols = row.find_all('td')
        if len(cols) < 7:
            continue
        
        try:
            accession_id = None
            # MODIFICATION: Search the entire row for the first matching accession ID.
            for cell in cols:
                # Check for an ID within a link first
                link = cell.find('a')
                if link and link.get('href'):
                    match = accession_pattern.search(link.get('href')) or accession_pattern.search(link.text)
                    if match:
                        accession_id = match.group(0)
                        break  # Found it, stop searching this row
                
                # If not in a link, check the plain text of the cell
                match = accession_pattern.search(cell.text)
                if match:
                    accession_id = match.group(0)
                    break # Found it, stop searching this row
            
            if not accession_id:
                # This warning will now only trigger if no ID is found anywhere in the row.
                print(f"     ⚠️ Warning: Could not find Accession ID in any column for row. Skipping. Row content: {[c.text.strip() for c in cols]}")
                continue

            rank = int(cols[0].text.strip())
            enzyme_name = cols[2].text.strip()
            ec_number = cols[3].text.strip()
            tanimoto_dist = float(cols[6].text.strip())

            reactions_data.append({
                "Accession_ID": accession_id,
                "Rank": rank,
                "Enzyme Name": enzyme_name,
                "EC Number": ec_number,
                "Tanimoto Distance": tanimoto_dist
            })
        except (ValueError, IndexError) as e:
            print(f"     ⚠️ Warning: Skipping row due to data error: {e}. Row content: {[c.text.strip() for c in cols]}")
            continue

    if not reactions_data:
        print("     ⚠️ No valid reaction data could be extracted.")
        return pd.DataFrame()

    df = pd.DataFrame(reactions_data)
    df = df.sort_values(by="Tanimoto Distance", ascending=True)
    df = df.drop_duplicates(subset=["Enzyme Name"], keep="first").reset_index(drop=True)
    
    print(f"     ✅ Found {len(df)} unique enzymes to analyze.")
    return df

# MODIFICATION: This function now also returns a dictionary of sequence lengths.
def fetch_ncbi_sequences(ids, output_filepath):
    """Fetches protein sequences from NCBI, saves them, and returns their lengths."""
    if not ids: return None, {}
    print(f"⬇️  Fetching {len(ids)} protein sequences from NCBI...")
    try:
        handle = Entrez.efetch(db="protein", id=ids, rettype="fasta", retmode="text")
        fasta_data = handle.read()
        handle.close()

        # Save to file
        with open(output_filepath, "w") as f:
            f.write(fasta_data)
        
        # MODIFICATION: Parse the data to get lengths
        seq_lengths = {}
        # The first part of the header up to the first space is the ID e.g. >WP_12345.1 description
        # Biopython's parser handles this correctly.
        for record in SeqIO.parse(StringIO(fasta_data), "fasta"):
             # The record.id from Entrez sometimes includes version info that matches the accession.
            seq_lengths[record.id] = len(record.seq)

        print(f"     ✅ Sequences saved to '{output_filepath}'.")
        return output_filepath, seq_lengths
    except Exception as e:
        print(f"     ❌ Error fetching from NCBI: {e}", file=sys.stderr)
        return None, {}

# MODIFICATION: This function now also returns a dictionary of translated sequence lengths.
def process_fna_directory(fna_dir):
    """Finds and translates nucleotide sequences from a directory of FNA files."""
    print(f"\n🔬 Processing local FNA files from '{fna_dir}'...")
    translated_fasta_file = "user_translated_proteins.fasta"
    translated_records = []
    
    for root, _, files_in_dir in os.walk(fna_dir):
        for file in files_in_dir:
            if file.lower().endswith((".fna", ".fasta", ".fa")):
                filepath = os.path.join(root, file)
                for record in SeqIO.parse(filepath, "fasta"):
                    try:
                        protein_seq = record.seq.translate(to_stop=True)
                        protein_record = SeqRecord(protein_seq, id=record.id, description=record.description)
                        translated_records.append(protein_record)
                    except Exception as e:
                        if "Partial codon" not in str(e):
                             print(f"       - ⚠️ Could not translate {record.id} in {file}: {e}", file=sys.stderr)

    if not translated_records:
        print("❌ No valid protein sequences could be translated from the FNA files.")
        return None, {}
        
    SeqIO.write(translated_records, translated_fasta_file, "fasta")
    
    # MODIFICATION: Create a dictionary of sequence lengths.
    user_seq_lengths = {rec.id: len(rec.seq) for rec in translated_records}
    
    print(f"✅ Translated {len(translated_records)} sequences and saved to '{translated_fasta_file}'.")
    return translated_fasta_file, user_seq_lengths


def create_blast_db(user_protein_file):
    if not user_protein_file: return None
    db_name = "local_protein_db"
    print(f"\n⚙️ Creating local BLAST database '{db_name}'...")
    command = ["makeblastdb", "-in", user_protein_file, "-dbtype", "prot", "-out", db_name]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print("✅ BLAST database created successfully.")
        return db_name
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create BLAST database: {e.stderr}", file=sys.stderr)
        return None

def run_blast_search(query_file, db_name, output_filepath):
    if not query_file or not db_name: return None
    print(f"🚀 Running BLAST search for {os.path.basename(query_file)}...")
    command = ["blastp", "-query", query_file, "-db", db_name, "-out", output_filepath, "-outfmt", "6 qseqid sseqid pident evalue"]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"     ✅ BLAST results saved to '{output_filepath}'.")
        return output_filepath
    except subprocess.CalledProcessError as e:
        print(f"     ❌ BLAST search failed: {e.stderr}", file=sys.stderr)
        return None

# MODIFICATION: Function now accepts sequence length dictionaries.
def merge_and_analyze(html_df, blast_output_file, user_protein_file, query_lengths, user_lengths):
    """
    Merges chemical and sequence similarity results, including sequence lengths.
    Finds the best BLAST hit for EACH enzyme.
    """
    if blast_output_file is None or not os.path.exists(blast_output_file):
        return pd.DataFrame()
    print("📊 Merging chemical and sequence similarity results...")
    blast_cols = ["Accession_ID", "Most Similar User Gene", "BLAST Similarity (%)", "E-Value"]
    try:
        blast_df = pd.read_csv(blast_output_file, sep='\t', header=None, names=blast_cols)
        if blast_df.empty: return pd.DataFrame()
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    
    # Find the best hit for EACH query accession ID.
    best_blast_hits = blast_df.sort_values(by=["E-Value", "BLAST Similarity (%)"], ascending=[True, False]).drop_duplicates("Accession_ID", keep="first")
    
    # Merge on the specific Accession_ID of each enzyme.
    merged_df = pd.merge(html_df, best_blast_hits, how="left", on="Accession_ID")
    if merged_df.empty: return pd.DataFrame()

    # MODIFICATION: Add sequence length columns
    merged_df['Query Seq Length'] = merged_df['Accession_ID'].map(query_lengths)
    merged_df['Hit Seq Length'] = merged_df['Most Similar User Gene'].map(user_lengths)

    user_annotations = {rec.id: rec.description for rec in SeqIO.parse(user_protein_file, "fasta")}
    merged_df["User Gene Annotation"] = merged_df["Most Similar User Gene"].apply(lambda x: user_annotations.get(x, "N/A"))
    print("     ✅ Analysis and merge complete.")
    return merged_df

# MODIFICATION: Heatmap generation now includes sequence lengths.
def generate_heatmap(df, output_filepath, context_filename):
    """
    Generates a clustered heatmap with dual heatmaps for Tanimoto and BLAST similarity.
    Includes sequence lengths in the annotations.
    """
    if df.empty or 'Tanimoto Distance' not in df.columns:
        print("     - Skipping heatmap: No data available.")
        return

    print("🎨 Generating clustered heatmap...")
    
    # --- 1. Data Preparation ---
    heatmap_cols = ['Tanimoto Distance', 'BLAST Similarity (%)', 'EC Number', 'Most Similar User Gene', 'Query Seq Length', 'Hit Seq Length']
    heatmap_df = df.set_index('Enzyme Name')[heatmap_cols].copy()
    heatmap_df['BLAST Similarity (%)'] = heatmap_df['BLAST Similarity (%)'].fillna(0)
    heatmap_df['Most Similar User Gene'] = heatmap_df['Most Similar User Gene'].fillna('N/A')
    # MODIFICATION: Fill NA for lengths with 0 or a placeholder.
    heatmap_df['Query Seq Length'] = heatmap_df['Query Seq Length'].fillna(0).astype(int)
    heatmap_df['Hit Seq Length'] = heatmap_df['Hit Seq Length'].fillna(0).astype(int)


    if len(heatmap_df) < 2:
        print("     - Skipping heatmap: Not enough data points (need at least 2) for clustering.")
        return

    # Normalize Tanimoto Distance for color mapping (lower is better, so invert)
    t_dist = heatmap_df['Tanimoto Distance']
    heatmap_df['Tanimoto (Norm.)'] = 1 - ((t_dist - t_dist.min()) / (t_dist.max() - t_dist.min()))
    
    # --- 2. Clustering ---
    row_linkage = linkage(pdist(heatmap_df[['Tanimoto Distance']]), method='average')

    # --- 3. Plot Layout ---
    fig = plt.figure(figsize=(24, max(8, len(heatmap_df) * 0.4)))
    # Gridspec: Dendrogram | EC Colors | Tanimoto Heatmap | BLAST Heatmap | Enzyme Names
    gs = gridspec.GridSpec(1, 5, width_ratios=[1.5, 0.2, 1, 3.5, 6], wspace=0.1)
    
    ax_dendro = fig.add_subplot(gs[0])
    ax_colors = fig.add_subplot(gs[1], sharey=ax_dendro)
    ax_heatmap_tanimoto = fig.add_subplot(gs[2], sharey=ax_dendro)
    ax_heatmap_blast = fig.add_subplot(gs[3], sharey=ax_dendro)
    ax_labels = fig.add_subplot(gs[4], sharey=ax_dendro)
    
    # --- 4. Plotting ---
    # Plot Dendrogram
    dendro_data = dendrogram(row_linkage, orientation='left', ax=ax_dendro, link_color_func=lambda k: 'black')
    ax_dendro.axis('off')
    
    # Reorder data based on clustering
    clustered_order = heatmap_df.index[dendro_data['leaves']]
    heatmap_df = heatmap_df.reindex(clustered_order)

    # Plot EC Class Color Bar
    heatmap_df['EC_Class'] = heatmap_df['EC Number'].str.split('.').str[0].fillna('N/A')
    ec_class_map = {'1': 'Oxidoreductases', '2': 'Transferases', '3': 'Hydrolases', 
                    '4': 'Lyases', '5': 'Isomerases', '6': 'Ligases', 'N/A': 'N/A'}
    heatmap_df['EC_Class_Name'] = heatmap_df['EC_Class'].map(ec_class_map).fillna('Other')
    unique_ec_names = sorted(heatmap_df['EC_Class_Name'].unique())
    ec_palette = sns.color_palette("husl", n_colors=len(unique_ec_names))
    ec_color_map = dict(zip(unique_ec_names, ec_palette))
    row_colors_list = heatmap_df['EC_Class_Name'].map(ec_color_map).tolist()
    row_colors_array = np.array(row_colors_list).reshape(-1, 1, 3) 
    ax_colors.imshow(row_colors_array, interpolation='nearest', aspect='auto')
    ax_colors.axis('off')

    # Plot Tanimoto Heatmap (Normalized)
    sns.heatmap(heatmap_df[['Tanimoto (Norm.)']], ax=ax_heatmap_tanimoto, cmap="viridis", cbar=False, annot=False)
    ax_heatmap_tanimoto.set_xticks([])
    ax_heatmap_tanimoto.set_yticks([])
    ax_heatmap_tanimoto.set_ylabel('')
    ax_heatmap_tanimoto.set_xlabel('Tanimoto\n(Norm.)', labelpad=10)

    # MODIFICATION: Create custom annotations for the BLAST heatmap including hit length
    blast_annotations = heatmap_df.apply(
        lambda row: f"{row['BLAST Similarity (%)']:.1f}% ({row['Most Similar User Gene']})\n{row['Hit Seq Length']} aa",
        axis=1
    ).to_numpy().reshape(-1, 1)

    # Plot BLAST Heatmap (Raw, with custom annotations)
    sns.heatmap(heatmap_df[['BLAST Similarity (%)']], ax=ax_heatmap_blast, cmap="magma", cbar=False, 
                annot=blast_annotations, fmt="", annot_kws={"size": 8})
    ax_heatmap_blast.set_xticks([])
    ax_heatmap_blast.set_yticks([])
    ax_heatmap_blast.set_ylabel('')
    ax_heatmap_blast.set_xlabel('BLAST % (HIT ID)\nSeq Length', labelpad=10)

    # MODIFICATION: Add horizontal enzyme names including query length
    ax_labels.axis('off')
    for i, (enzyme_name, query_len) in enumerate(zip(heatmap_df.index, heatmap_df['Query Seq Length'])):
        label_text = f"{enzyme_name} ({query_len} aa)"
        ax_labels.text(0.02, i + 0.5, label_text, ha='left', va='center', fontsize=10)

    # --- 5. Legend and Final Touches ---
    # Create a new axis for the legend area in the top right
    legend_area_ax = fig.add_axes([0.85, 0.75, 0.15, 0.2]) # [left, bottom, width, height]
    legend_area_ax.axis('off')
    
    # EC Class Legend
    handles = [plt.Rectangle((0,0), 1, 1, color=ec_color_map[name]) for name in unique_ec_names]
    leg = legend_area_ax.legend(handles, unique_ec_names, title='EC Class', loc='upper left', frameon=True)
    leg.get_frame().set_edgecolor('black')
    
    # Tanimoto Color Bar
    cbar_ax_tanimoto = fig.add_axes([0.85, 0.60, 0.15, 0.03]) # Position color bar below legend
    sm_tanimoto = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=1))
    cbar_tanimoto = fig.colorbar(sm_tanimoto, cax=cbar_ax_tanimoto, orientation='horizontal')
    cbar_tanimoto.set_label('Norm. Tanimoto Similarity', labelpad=5)
    
    # BLAST Color Bar
    cbar_ax_blast = fig.add_axes([0.85, 0.45, 0.15, 0.03]) # Position color bar below Tanimoto
    sm_blast = plt.cm.ScalarMappable(cmap="magma", norm=plt.Normalize(vmin=0, vmax=100))
    cbar_blast = fig.colorbar(sm_blast, cax=cbar_ax_blast, orientation='horizontal')
    cbar_blast.set_label('BLAST Similarity (%)', labelpad=5)

    fig.suptitle(f"Hierarchical Clustering by Chemical Similarity vs. Sequence Similarity\n(Source: {context_filename})", fontsize=16, y=0.98)
    plt.subplots_adjust(top=0.9, bottom=0.1, right=0.82)

    plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
    print(f"     ✅ Heatmap saved to '{output_filepath}'.")
    plt.close(fig)

def main():
    """Main execution function to set up and run the analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="Recursively find and process HTML reports, comparing genes using BLAST.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--html-dir", required=True, help="Top-level directory with HTML files.")
    parser.add_argument("--fna-dir", required=True, help="Directory with local FNA files.")
    parser.add_argument("--output-dir", required=False, help="Base directory for all outputs.")
    args = parser.parse_args()

    if not os.path.isdir(args.html_dir):
        sys.exit(f"Error: The specified HTML directory does not exist: {args.html_dir}")

    if args.output_dir is None:
        args.output_dir = os.path.basename(os.path.normpath(args.html_dir)) + "_results"
        print(f"--- No output directory specified. Defaulting to '{args.output_dir}' ---")

    check_blast_installed()
    # MODIFICATION: Capture sequence length dictionaries
    user_fasta_file, user_seq_lengths = process_fna_directory(args.fna_dir)
    if not user_fasta_file: sys.exit("Aborting: Could not process local FNA files.")
    local_db_name = create_blast_db(user_fasta_file)
    if not local_db_name: sys.exit("Aborting: Could not create local BLAST database.")

    html_files_found = 0
    for root, _, files in os.walk(args.html_dir):
        for filename in files:
            if not filename.lower().endswith(".html"):
                continue
            html_files_found += 1
            html_filepath = os.path.join(root, filename)
            print(f"\n\n{'='*20} Processing file: {html_filepath} {'='*20}")
            html_data_df = parse_single_html_report(html_filepath)
            if html_data_df.empty:
                print(f"--- No data parsed from {filename}. Skipping. ---")
                continue

            base_filename = os.path.splitext(filename)[0]
            relative_path = os.path.relpath(os.path.dirname(html_filepath), args.html_dir)
            group_output_dir = os.path.join(args.output_dir, relative_path)
            data_subdir = os.path.join(group_output_dir, 'analysis_data')
            os.makedirs(group_output_dir, exist_ok=True)
            os.makedirs(data_subdir, exist_ok=True)
            
            heatmap_output_path = os.path.join(group_output_dir, f"{base_filename}_blast-similarity_heatmap.png")
            csv_output_path = os.path.join(data_subdir, f"{base_filename}_results.csv")
            ncbi_fasta_file = os.path.join(data_subdir, f"{base_filename}_query.fasta")
            blast_result_file = os.path.join(data_subdir, f"{base_filename}_blast_results.tsv")

            unique_query_ids = list(html_data_df["Accession_ID"].unique())
            # MODIFICATION: Capture query sequence lengths
            _, query_seq_lengths = fetch_ncbi_sequences(unique_query_ids, ncbi_fasta_file)
            run_blast_search(ncbi_fasta_file, local_db_name, blast_result_file)
            # MODIFICATION: Pass length dictionaries to the analysis function
            final_df = merge_and_analyze(html_data_df, blast_result_file, user_fasta_file, query_seq_lengths, user_seq_lengths)

            if not final_df.empty:
                print(f"\n--- Results for {filename} ---")
                # MODIFICATION: Add new length columns to display and CSV output
                display_cols = ["Rank", "Enzyme Name", "Query Seq Length", "Accession_ID", "Most Similar User Gene", "Hit Seq Length", "BLAST Similarity (%)", "Tanimoto Distance", "E-Value", "User Gene Annotation"]
                display_cols = [col for col in display_cols if col in final_df.columns]
                with pd.option_context('display.max_rows', 20, 'display.max_columns', None, 'display.width', 1200):
                    print(final_df[display_cols])
                final_df.to_csv(csv_output_path, index=False)
                print(f"\n     ✅ Results table saved to '{csv_output_path}'")
                generate_heatmap(final_df, heatmap_output_path, filename)
            else:
                print(f"--- No significant matches found for {filename}. ---")

    if html_files_found == 0:
        print("\n--- Warning: No HTML files were found in the specified directory tree. ---")
    print("\n--- Script finished ---")

if __name__ == "__main__":
    main()