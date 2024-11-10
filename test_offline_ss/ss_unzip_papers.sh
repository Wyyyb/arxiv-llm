#!/bin/bash

# 设置目录路径
## DIR="/data/yubowang/offline_ss_abstract"
#DIR="/data/yubowang/offline_ss_papers"
## DIR="/data/yubowang/offline_ss_tldrs"

DIR="/gpfs/public/research/xy/yubowang/offline_ss_papers"

# 创建日志文件
LOG_FILE="$DIR/decompress_log.txt"
touch "$LOG_FILE"

echo "开始解压缩操作: $(date)" | tee -a "$LOG_FILE"

# 计数器
total_files=$(find "$DIR" -name "*.gz" | wc -l)
current=0

# 遍历所有.gz文件
find "$DIR" -name "*.gz" | while read file; do
    ((current++))

    # 获取不带.gz的文件名
    output_file="${file%.gz}"

    echo "[$current/$total_files] 正在解压: $file" | tee -a "$LOG_FILE"

    # 解压文件
    gzip -d -k "$file" 2>> "$LOG_FILE"

    if [ $? -eq 0 ]; then
        echo "成功解压: $file -> $output_file" | tee -a "$LOG_FILE"
    else
        echo "解压失败: $file" | tee -a "$LOG_FILE"
    fi
done

echo "解压完成: $(date)" | tee -a "$LOG_FILE"
echo "详细日志请查看: $LOG_FILE"

