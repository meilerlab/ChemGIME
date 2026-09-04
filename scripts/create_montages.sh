#!/bin/bash

#
# This script creates montages of BLAST similarity heatmaps.
#
# To avoid memory errors, it finds each organism-specific subdirectory
# and creates a separate montage for the images within it. This is more
# memory-efficient than creating one single, large montage.
#
# Usage:
#   ./create_montages.sh <path_to_base_analysis_directory>
#
# Example:
#   ./create_montages.sh ./plots
#
# Dependencies:
#   - ImageMagick (specifically the 'montage' command)
#

# --- Configuration ---
BASE_DIR="$1"
FILE_PATTERN="*_blast-similarity_heatmap.png"

# --- Script Body ---

# 1. Input Validation
if [ -z "$BASE_DIR" ]; then
    echo "Error: Please provide the path to the base analysis directory."
    echo "Usage: $0 <path_to_directory>"
    exit 1
fi

if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Directory '$BASE_DIR' not found."
    exit 1
fi

# 2. Setup Global Output Directory
MONTAGE_OUTPUT_DIR="${BASE_DIR}/montage_summary"
mkdir -p "$MONTAGE_OUTPUT_DIR"
echo "✅ Montages will be saved in: $MONTAGE_OUTPUT_DIR"
echo "---"

# 3. Process Each Subdirectory Individually
# Loop through each subdirectory in the base directory.
for organism_dir in "$BASE_DIR"/*/; do
    # Check if it's actually a directory
    if [ ! -d "$organism_dir" ]; then
        continue
    fi

    organism_name=$(basename "$organism_dir")
    echo "Processing directory: $organism_name"

    # Find all heatmap files within this specific directory.
    PLOT_FILES=($(find "$organism_dir" -maxdepth 1 -type f -name "$FILE_PATTERN"))

    if [ ${#PLOT_FILES[@]} -gt 0 ]; then
        # Define a unique output file for this organism's montage.
        MONTAGE_FILE="$MONTAGE_OUTPUT_DIR/${organism_name}_summary.png"
        echo "  👍 Found ${#PLOT_FILES[@]} heatmaps. Creating montage..."

        montage_args=()
        for file in "${PLOT_FILES[@]}"; do
            # Label is just the report name now, since the organism is in the filename.
            filename=$(basename "$file")
            temp_label=${filename#report_}
            report_name=${temp_label%_blast-similarity_heatmap.png}
            montage_args+=(-label "$report_name" "$file")
        done

        # Create the montage for the current organism.
        montage "${montage_args[@]}" \
            -resize 400x \
            -tile 4x \
            -geometry +10+10 \
            -title "$organism_name" \
            "$MONTAGE_FILE"

        if [ $? -eq 0 ] && [ -s "$MONTAGE_FILE" ]; then
            echo "  🎉 Successfully created: $MONTAGE_FILE"
        else
            echo "  ❌ Error: Montage command failed for '$organism_name'."
        fi
    else
        echo "  🤷 No heatmap files found in this directory. Skipping."
    fi
    echo "---"
done

echo "Script complete."