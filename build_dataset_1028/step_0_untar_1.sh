#!/bin/bash

input_dir="/data/yubowang/2410_arxiv/2410"
output_dir="/data/yubowang/2410_arxiv_latex"

# 创建输出目录（如果不存在）
mkdir -p "$output_dir"

# 检查输出目录是否创建成功
if [ ! -d "$output_dir" ]; then
    echo "Error: Failed to create output directory '$output_dir'"
    exit 1
fi

# 计数器
success_count=0
fail_count=0

# 遍历所有gz文件并解压
for gz_file in "$input_dir"/*.gz; do
    # 检查是否存在匹配的文件
    if [ ! -e "$gz_file" ]; then
        echo "No .gz files found in $input_dir"
        exit 0
    fi

    echo "Processing: $gz_file"

    # 获取不带.gz的文件名作为子目录名
    subdir="$output_dir/$(basename "$gz_file" .gz)"

    # 创建子目录
    mkdir -p "$subdir"

    # 检查子目录是否创建成功
    if [ ! -d "$subdir" ]; then
        echo "Error: Failed to create subdirectory '$subdir'"
        ((fail_count++))
        continue
    fi

    # 解压文件到对应子目录
    if tar -xzf "$gz_file" -C "$subdir"; then
        echo "Successfully extracted: $gz_file to $subdir"
        ((success_count++))
    else
        echo "Failed to extract: $gz_file"
        ((fail_count++))
    fi
done

# 打印统计信息
echo "Extraction complete!"
echo "Successfully extracted: $success_count files"
echo "Failed to extract: $fail_count files"