import json
import math
import os
import subprocess
from pathlib import Path


def split_jsonl_and_zip(input_path, output_path):
    # 创建临时目录
    tmp_dir = Path("tmp")
    tmp_dir.mkdir(exist_ok=True)

    # 读取输入文件
    with open(input_path) as f:
        lines = f.readlines()

    # 计算每个子文件应有的行数
    total_lines = len(lines)
    lines_per_file = math.ceil(total_lines / 50)

    # 确保输出目录存在
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # 拆分文件并逐个压缩
    for i in range(50):
        start = i * lines_per_file
        end = min((i + 1) * lines_per_file, total_lines)

        # 写入临时jsonl文件
        tmp_file = tmp_dir / f"train_data_1103-{i}.jsonl"
        with open(tmp_file, "w") as f:
            f.writelines(lines[start:end])

        # 压缩单个文件
        zip_name = f"train_data_1103-{i}.zip"
        subprocess.run(["zip", "-9", "-j",
                        str(Path(output_path) / zip_name),
                        str(tmp_file)])

        # 删除临时jsonl文件
        # tmp_file.unlink()

    # 删除临时目录
    # tmp_dir.rmdir()


# 使用示例
input_file_path = "/data/yubowang/arxiv-llm/local_1031/train_data_1103.jsonl"
output_file_path = "/data/yubowang/data_trans_1030/train_data_1103_split"
split_jsonl_and_zip(input_file_path, output_file_path)

