#!/bin/bash
find "/Users/MyDisk/2024/git/mm_data_process/benchmark_tasks" -type d -name "images" -exec rm -rf {} +

#!/bin/bash

input_dir="/Users/MyDisk/2024/git/mm_data_process/benchmark_tasks"
output_name="benchmark_tasks_$(date +%Y%m%d_%H%M%S).zip"

echo "开始压缩..."
# -9 最高压缩级别
# -r 递归处理目录
zip -9 -r "$output_name" "$input_dir"

echo "压缩完成：$output_name"
echo "压缩包大小: $(du -h "$output_name" | cut -f1)"