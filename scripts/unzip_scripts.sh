#!/bin/bash

input_dir="/gpfs/public/research/xy/yubowang/data_trans_1030/qwen_extract_title_data/"
output_dir="/gpfs/public/research/xy/yubowang/arxiv-llm/qwen_extract_title_data_1031/"

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# Find all zip files and extract them
find "$input_dir" -type f -name "*.zip" | while read -r file; do
    # Extract to output directory
    unzip -o "$file" -d "$output_dir"

    # Alternative using 7zip if installed
    # if command -v 7z >/dev/null 2>&1; then
    #     7z x "$file" -o"$output_dir" >/dev/null
    # else
    #     unzip -o "$file" -d "$output_dir"
    # fi
done

echo "Extraction complete. Files extracted to $output_dir"