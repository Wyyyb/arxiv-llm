import json
import os
from math import ceil


def split_json_file(input_json_path, output_folder):
    # 创建输出文件夹(如果不存在)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 读取JSON文件
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 计算每个子文件应包含的item数量
    total_items = len(data)
    items_per_file = ceil(total_items / 10)

    # 获取原始文件名(不含扩展名)
    base_filename = os.path.splitext(os.path.basename(input_json_path))[0]

    # 将字典转换为列表
    items = list(data.items())

    # 分割并保存文件
    for i in range(10):
        start_idx = i * items_per_file
        end_idx = min((i + 1) * items_per_file, total_items)

        # 创建子字典
        sub_dict = dict(items[start_idx:end_idx])

        # 构建输出文件路径
        output_filename = f"{base_filename}_{i + 1}.json"
        output_path = os.path.join(output_folder, output_filename)

        # 保存子文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sub_dict, f, ensure_ascii=False, indent=4)

    return f"Successfully split {input_json_path} into 10 files in {output_folder}"


split_json_file("/data/yubowang/arxiv-llm/local_1031/offline_query_ss_1110.json",
                "/data/yubowang/data_trans_1030/ss_data_query_1110/")

