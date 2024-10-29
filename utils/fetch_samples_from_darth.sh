#!/bin/bash

# 配置参数
REMOTE_USER="yubo"
REMOTE_HOST="darth.cs.uwaterloo.ca"
OUTPUT_PATH="../local_1028/samples_1028/"  # 本地输出路径

# 存储远程路径的文件
PATHS_FILE="remote_paths.txt"

# 确保输出根目录存在
mkdir -p "$OUTPUT_PATH"

# 读取路径列表并处理每个路径
while IFS= read -r remote_path; do
    # 获取目录名称
    dir_name=$(basename "$remote_path")
    local_dir="$OUTPUT_PATH/$dir_name"

    # 创建本地目录
    mkdir -p "$local_dir"

    # 使用rsync下载文件
    # -a: 归档模式，保持文件属性
    # -v: 详细输出
    # -z: 压缩传输
    # --progress: 显示进度
    rsync -avz --progress "$REMOTE_USER@$REMOTE_HOST:$remote_path/" "$local_dir/"

    if [ $? -eq 0 ]; then
        echo "Successfully downloaded files from $remote_path to $local_dir"
    else
        echo "Error downloading files from $remote_path"
    fi
done < "$PATHS_FILE"

