import os
from pathlib import Path
import json
from tqdm import tqdm
import base64


def write_html_files(json_path, output_folder):
    """
    从JSON文件中读取HTML内容并写入到指定文件夹中

    Args:
        json_path (str): JSON文件路径
        output_folder (str): 输出HTML文件的文件夹路径
    """
    # 确保输出文件夹存在
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # 读取JSON文件
    try:
        with open(json_path, 'r') as f:
            html_contents = json.load(f)
    except Exception as e:
        print(f"Error reading JSON file: {str(e)}")
        return

    # 遍历JSON中的每个条目并写入HTML文件
    for filename, content in tqdm(html_contents.items()):
        try:
            file_path = output_path / filename
            # 将base64编码的内容解码回二进制
            binary_content = base64.b64decode(content)
            with open(file_path, 'wb') as f:
                f.write(binary_content)
        except Exception as e:
            print(f"Error writing {filename}: {str(e)}")


# 使用示例
json_file_path = "/data/yubowang/htmls_json_1218/htmls_upload.json"
output_html_dir = "/data/yubowang/htmls_1218/"

write_html_files(json_file_path, output_html_dir)

