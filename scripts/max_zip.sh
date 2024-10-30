#!/bin/bash


input_dir="/data/yubowang/arxiv-llm/qwen_extract_title_data"
output_dir="/data/yubowang/data_trans_1030/qwen_extract_title_data"

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# Find all json files and process them
find "$input_dir" -type f -name "*.json" | while read -r file; do
    # Get base filename without path and extension
    filename=$(basename "$file" .json)

    # Create zip with maximum compression
    zip -j -9 "$output_dir/${filename}.zip" "$file"

    # Alternative method using 7zip if installed (generally better compression)
    # if command -v 7z >/dev/null 2>&1; then
    #     7z a -tzip -mx=9 "$output_dir/${filename}.zip" "$file" >/dev/null
    # else
    #     zip -j -9 "$output_dir/${filename}.zip" "$file"
    # fi
done

echo "Compression complete. Files saved in $output_dir"