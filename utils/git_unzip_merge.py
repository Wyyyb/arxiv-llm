import json
import os
import subprocess
from pathlib import Path


def merge_zipped_jsonl(input_dir, output_file):
    # 创建临时目录用于解压文件
    tmp_dir = Path("tmp")
    tmp_dir.mkdir(exist_ok=True)

    # 确保输出目录存在
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 获取所有zip文件并排序
    zip_files = sorted(Path(input_dir).glob("*.zip"))

    # 用于存储所有jsonl内容的列表
    all_lines = []

    # 处理每个zip文件
    for zip_file in zip_files:
        # 解压到临时目录
        subprocess.run(["unzip", "-o", str(zip_file), "-d", str(tmp_dir)])

        # 读取解压的jsonl文件
        jsonl_files = list(tmp_dir.glob("*.jsonl"))
        if jsonl_files:
            with open(jsonl_files[0]) as f:
                all_lines.extend(f.readlines())

        # 清理临时解压的文件
        for jsonl_file in jsonl_files:
            jsonl_file.unlink()

    # 写入合并后的文件
    with open(output_file, "w") as f:
        f.writelines(all_lines)

    # 删除临时目录
    tmp_dir.rmdir()


# 使用示例
input_dir = "/gpfs/public/research/xy/yubowang/data_trans_1030/train_data_1103_split"
output_file = "../local/training_data/train_data_1103.jsonl"
merge_zipped_jsonl(input_dir, output_file)