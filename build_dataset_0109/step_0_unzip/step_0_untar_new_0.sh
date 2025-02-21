#!/bin/bash

input_dir="/data/xueguang/arxiv-latex"
output_dir="/data/yubowang/2501_arxiv"

# 检查输入目录是否存在
if [ ! -d "$input_dir" ]; then
    echo "Error: Input directory '$input_dir' does not exist"
    exit 1
fi

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

# 遍历所有符合条件的tar文件并解压
for tar_file in "$input_dir"/arXiv_src_2412*.tar; do
    # 检查是否存在匹配的文件
    if [ ! -e "$tar_file" ]; then
        echo "No matching arXiv_src_2501*.tar files found in $input_dir"
        exit 0
    fi

    echo "Processing: $tar_file"

    # 获取文件名（不含路径和扩展名）
    filename=$(basename "$tar_file" .tar)

    target_dir="$output_dir"

    # 解压文件
    if tar -xf "$tar_file" -C "$target_dir"; then
        echo "Successfully extracted: $tar_file -> $target_dir"
        ((success_count++))
    else
        echo "Failed to extract: $tar_file"
        ((fail_count++))
    fi
done

# 打印统计信息
echo "Extraction complete!"
echo "Successfully extracted: $success_count files"
echo "Failed to extract: $fail_count files"