import os
from pathlib import Path
import json
import base64
from tqdm import tqdm


def read_html_files_raw(folder_path):
    """
    读取指定文件夹下所有HTML文件的原始内容

    Args:
        folder_path (str): 文件夹路径

    Returns:
        dict: 以文件名为key，文件base64编码内容为value的字典
    """
    html_contents = {}

    folder = Path(folder_path)

    for html_file in tqdm(folder.rglob('*.html')):
        try:
            with open(html_file, 'rb') as f:
                # 将二进制内容转换为base64编码的字符串
                content = base64.b64encode(f.read()).decode('utf-8')
                html_contents[html_file.name] = content

        except Exception as e:
            print(f"Error reading {html_file}: {str(e)}")

    return html_contents


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


# 示例使用
html_dir_path = "/gpfs/public/research/public/htmls"
json_output_path = "/gpfs/public/research/public/htmls_json/htmls_upload.json"

# 收集HTML文件
result = read_html_files_raw(html_dir_path)

# 保存为JSON
with open(json_output_path, "w") as fo:
    fo.write(json.dumps(result))

