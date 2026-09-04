#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script provides a multi-faceted analysis of reaction similarity data,
transforming it from raw HTML reports into publication-quality visualizations
and structured data. It can be run from the command line to process entire
directories of HTML files iteratively.

The script performs the following steps for each input file:
1.  Loads and parses reaction similarity data from an HTML file.
2.  Preprocesses the data by handling duplicate entries and parsing hierarchical
    EC numbers.
3.  Generates and saves three distinct visualizations:
    a. An annotated heatmap clustering reactions by chemical similarity.
    b. A series of box plots grouping reactions by functional EC class.
    c. An integrated network graph showing relationships between reactions.
4.  Saves the processed data to a CSV file for further analysis.
"""

import argparse
import os
import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
import numpy as np

def parse_ec(ec_number):
    """
    Parses an EC number string into its four constituent parts.

    Args:
        ec_number (str): The EC number string (e.g., '1.2.3.4').

    Returns:
        list: A list containing the four parts of the EC number.
              Returns ['N/A', 'N/A', 'N/A', 'N/A'] for invalid input.
    """
    if pd.isna(ec_number) or ec_number in ('nan', ''):
        return ['N/A'] * 4
    parts = str(ec_number).split('.')
    return [parts[i] if i < len(parts) else 'N/A' for i in range(4)]

def process_html_file(html_path, output_dir, global_min_tanimoto, global_max_tanimoto):
    """
    Processes a single HTML file to extract reaction data, generate
    visualizations, and save the results.

    Args:
        html_path (str): The full path to the input HTML file.
        output_dir (str): The directory where results will be saved.
        global_min_tanimoto (float): The global minimum Tanimoto distance for axis scaling.
        global_max_tanimoto (float): The global maximum Tanimoto distance for axis scaling.
    """
    print(f"\nProcessing file: {html_path}")
    base_filename = os.path.splitext(os.path.basename(html_path))[0]

    # --- 1. Load and Preprocess the Data ---
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
    except FileNotFoundError:
        print(f"  [Error] File not found: {html_path}. Skipping.")
        return

    soup = BeautifulSoup(html_content, 'html.parser')
    table = soup.find('table')
    if not table:
        print(f"  [Warning] No table found in {html_path}. Skipping.")
        return

    header_row = table.find('tr')
    if not header_row or not header_row.find_all('th'):
        print(f"  [Warning] No table header (<th> tags) found in {html_path}. Skipping.")
        return
    columns = [th.text.strip() for th in header_row.find_all('th')]

    data = []
    for row in table.find_all('tr')[1:]:
        cols = row.find_all('td')
        if len(cols) == len(columns):
            data.append([col.text.strip() for col in cols])
        else:
            print(f"  [Warning] Skipping row with mismatched column count in {html_path}.")

    if not data:
        print(f"  [Warning] No data found in the table in {html_path}. Skipping.")
        return

    df_raw = pd.DataFrame(data, columns=columns)
    
    column_map = {
        'Rank': 'rank', 'Seed ID': 'seed_id', 'seed_id': 'seed_id',
        'MetaNetX ID': 'metanetx_id', 'Name': 'name', 'name': 'name',
        'Tanimoto Distance': 'tanimoto_distance', 'tanimoto distance': 'tanimoto_distance',
        'EC Number': 'ec_number', 'ec': 'ec_number', 'Pathway': 'pathway',
        'pathway': 'pathway', 'MetaCyc Title': 'metacyc_title',
        'metacyc title': 'metacyc_title'
    }
    df_raw.rename(columns=lambda c: column_map.get(c.strip(), c.strip()), inplace=True)

    required_cols = ['tanimoto_distance', 'ec_number', 'name', 'seed_id', 'pathway']
    if not all(col in df_raw.columns for col in required_cols):
        print(f"  [Error] Missing one or more required columns in {html_path}. Required: {required_cols}. Found: {list(df_raw.columns)}. Skipping.")
        return

    df_raw['tanimoto_distance'] = pd.to_numeric(df_raw['tanimoto_distance'], errors='coerce')
    df_raw.dropna(subset=['tanimoto_distance'], inplace=True)
    if df_raw.empty:
        print(f"  [Warning] No valid Tanimoto data after cleaning in {html_path}. Skipping visualization.")
        return

    grouping_cols = ['seed_id', 'name', 'tanimoto_distance', 'ec_number', 'pathway']
    df_unique = df_raw.groupby(grouping_cols).size().reset_index(name='Hit_Count')

    ec_df = df_unique['ec_number'].apply(lambda x: pd.Series(parse_ec(x)))
    ec_df.columns = ['EC_Class', 'EC_Subclass', 'EC_SubSubclass', 'EC_Serial']
    df_processed = pd.concat([df_unique, ec_df], axis=1)

    ec_class_map = {
        '1': 'Oxidoreductases', '2': 'Transferases', '3': 'Hydrolases',
        '4': 'Lyases', '5': 'Isomerases', '6': 'Ligases',
        '7': 'Translocases', 'N/A': 'Not Available'
    }
    df_processed['EC_Class_Name'] = df_processed['EC_Class'].map(ec_class_map).fillna('Other')
    df_processed['Label'] = df_processed['name'] + " (" + df_processed['seed_id'] + ")"

    csv_path = os.path.join(output_dir, f"{base_filename}_processed_data.csv")
    df_processed.to_csv(csv_path, index=False)
    print(f"  - Saved processed data to {csv_path}")

    # --- 2. Analysis A: Chemical Similarity View (Annotated Heatmap) ---
    if not df_processed.empty:
        try:
            tanimoto_distances = df_processed[['tanimoto_distance']]
            if tanimoto_distances.shape[0] > 1:
                row_linkage = linkage(tanimoto_distances, method='ward')
                scaler = MinMaxScaler()
                heatmap_data_scaled = pd.DataFrame(scaler.fit_transform(df_processed[['tanimoto_distance', 'Hit_Count']]),
                                                   columns=['Tanimoto Distance (Scaled)', 'Hit Count (Scaled)'],
                                                   index=df_processed['Label'])
                ec_class_names = df_processed.set_index('Label')['EC_Class_Name']
                unique_ec_classes = sorted(ec_class_names.unique())
                palette = sns.color_palette("husl", len(unique_ec_classes))
                ec_color_map = dict(zip(unique_ec_classes, palette))
                row_colors = ec_class_names.map(ec_color_map)
                g = sns.clustermap(heatmap_data_scaled,
                                   row_linkage=row_linkage, col_cluster=False,
                                   row_colors=row_colors, cmap="viridis_r",
                                   figsize=(12, max(10, 0.5 * len(df_processed))),
                                   dendrogram_ratio=(0.2, 0.02),
                                   cbar_kws={'label': 'Normalized Value'})
                g.fig.subplots_adjust(right=0.7)
                g.ax_cbar.set_position([1.1, .8, .08, .1])
                handles = [plt.Rectangle((0,0),1,1, color=ec_color_map[label]) for label in unique_ec_classes]
                g.ax_heatmap.legend(handles, unique_ec_classes, title='EC Class',
                                    bbox_to_anchor=(2, 1), loc='upper right', borderaxespad=0.)
                g.ax_heatmap.set_xlabel("Normalized Metrics")
                g.ax_heatmap.set_ylabel("Reaction (Name and Seed ID)")
                plt.suptitle('Hierarchical Clustering by Chemical Similarity', y=0.99, fontsize=16)
                heatmap_path = os.path.join(output_dir, f"{base_filename}_heatmap.png")
                plt.savefig(heatmap_path, bbox_inches='tight')
                plt.close()
                print(f"  - Saved heatmap to {heatmap_path}")
            else:
                print("  - Skipped heatmap: Not enough data points for clustering.")
        except Exception as e:
            print(f"  [Error] Could not generate heatmap: {e}")

    # --- 3. Analysis B: Functional Grouping View (Box Plots) ---
    if not df_processed.empty:
        try:
            ec_counts = df_processed['EC_Class_Name'].value_counts().reset_index()
            ec_counts.columns = ['EC_Class_Name', 'Count']
            df_plot_data = pd.merge(df_processed, ec_counts, on='EC_Class_Name')
            df_plot_data['Label_with_Count'] = df_plot_data['EC_Class_Name'] + " (n=" + df_plot_data['Count'].astype(str) + ")"
            df_plot_data['Size_Scaled'] = df_plot_data['Hit_Count'] * 25
            y_order = df_plot_data.groupby('Label_with_Count')['tanimoto_distance'].median().sort_values().index
            plt.figure(figsize=(14, 8))
            ax = sns.boxplot(x='tanimoto_distance', y='Label_with_Count', data=df_plot_data,
                             hue='Label_with_Count', palette='husl', order=y_order, legend=False, dodge=False)
            y_cat_map = {cat: i for i, cat in enumerate(y_order)}
            for cat in y_order:
                cat_data = df_plot_data[df_plot_data['Label_with_Count'] == cat]
                y_pos = y_cat_map[cat]
                y_jittered = np.random.normal(loc=y_pos, scale=0.1, size=len(cat_data))
                ax.scatter(x=cat_data['tanimoto_distance'], y=y_jittered,
                           s=cat_data['Size_Scaled'], color=".25", alpha=0.7, zorder=10)
            legend_hit_counts = sorted(df_plot_data['Hit_Count'].unique())
            if len(legend_hit_counts) > 3:
                legend_counts_for_display = [
                    legend_hit_counts[0],
                    legend_hit_counts[len(legend_hit_counts) // 2],
                    legend_hit_counts[-1]]
            else:
                legend_counts_for_display = legend_hit_counts
            legend_handles = [plt.scatter([], [], s=count * 25, color='.25', alpha=0.7, label=f'{count} hits') for count in legend_counts_for_display]
            ax.legend(handles=legend_handles, title="Hit Count", loc='lower right', labelspacing=2.0, frameon=True)
            plt.title('Distribution of Tanimoto Distances by EC Class', fontsize=16)
            plt.xlabel('Tanimoto Distance', fontsize=12)
            plt.ylabel('EC Class (Number of Unique Reactions)', fontsize=12)
            plt.grid(axis='x', linestyle='--', alpha=0.6)
            plt.tight_layout()
            # --- MODIFIED SECTION ---
            # Set x-axis limits using the pre-calculated global min and max
            # Add a small padding to prevent points from being on the edge.
            if global_min_tanimoto is not None and global_max_tanimoto is not None:
                plt.xlim(global_min_tanimoto - 0.02, global_max_tanimoto + 0.02)
            # --- END MODIFIED SECTION ---
            boxplot_path = os.path.join(output_dir, f"{base_filename}_boxplot.png")
            plt.savefig(boxplot_path, bbox_inches='tight')
            plt.close()
            print(f"  - Saved boxplot to {boxplot_path}")
        except Exception as e:
            print(f"  [Error] Could not generate boxplot: {e}")

def find_global_tanimoto_range(input_dir):
    """
    Scans all HTML files in a directory to find the global min and max
    Tanimoto distances. This is the first pass of a two-pass process.
    """
    print("--- Starting Pass 1: Scanning for Tanimoto distance range ---")
    global_min = float('inf')
    global_max = float('-inf')
    
    column_map = {
        'Tanimoto Distance': 'tanimoto_distance', 'tanimoto distance': 'tanimoto_distance'
    }

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.html'):
                html_path = os.path.join(root, file)
                try:
                    with open(html_path, 'r', encoding='utf-8') as f:
                        soup = BeautifulSoup(f.read(), 'html.parser')
                    table = soup.find('table')
                    if not table: continue
                    
                    header_row = table.find('tr')
                    if not header_row: continue
                    
                    columns = [th.text.strip() for th in header_row.find_all('th')]
                    # Standardize just the column we need for this pass
                    renamed_columns = [column_map.get(c, c) for c in columns]
                    
                    if 'tanimoto_distance' not in renamed_columns:
                        continue
                        
                    tanimoto_idx = renamed_columns.index('tanimoto_distance')
                    
                    distances = []
                    for row in table.find_all('tr')[1:]:
                        cols = row.find_all('td')
                        if len(cols) == len(columns):
                            distances.append(cols[tanimoto_idx].text.strip())
                    
                    if distances:
                        s = pd.to_numeric(pd.Series(distances), errors='coerce').dropna()
                        if not s.empty:
                            global_min = min(global_min, s.min())
                            global_max = max(global_max, s.max())
                except Exception as e:
                    print(f"  [Warning] Could not scan {html_path}: {e}")

    if global_min == float('inf') or global_max == float('-inf'):
        print("--- Pass 1 Warning: No valid Tanimoto distances found. Using default range. ---")
        return None, None

    print(f"--- Pass 1 Complete: Global Tanimoto range is [{global_min:.4f}, {global_max:.4f}] ---")
    return global_min, global_max

def main():
    """
    Main function to parse command-line arguments and orchestrate the
    processing of HTML files.
    """
    parser = argparse.ArgumentParser(
        description="Process reaction similarity HTML files to generate visualizations and data.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('input_dir', type=str, help="The path to the directory containing the input HTML files.")
    parser.add_argument('output_dir', type=str, help="The path to the directory where results will be saved.")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"[Error] Input directory not found: {args.input_dir}")
        return

    # --- Pass 1: Find global Tanimoto range ---
    global_min, global_max = find_global_tanimoto_range(args.input_dir)

    # --- Pass 2: Process each file and generate plots ---
    print("\n--- Starting Pass 2: Processing files and generating plots ---")
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith('.html'):
                html_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, args.input_dir)
                current_output_dir = os.path.join(args.output_dir, relative_path)
                os.makedirs(current_output_dir, exist_ok=True)
                process_html_file(html_path, current_output_dir, global_min, global_max)

    print("\nProcessing complete.")

if __name__ == '__main__':
    plt.switch_backend('Agg')
    main()
