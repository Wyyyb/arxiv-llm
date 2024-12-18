import os
from pathlib import Path
import json
from tqdm import tqdm


def read_html_files_raw(folder_path):
    """
    读取指定文件夹下所有HTML文件的原始内容

    Args:
        folder_path (str): 文件夹路径

    Returns:
        dict: 以文件名为key，文件原始内容为value的字典
    """
    html_contents = {}

    folder = Path(folder_path)

    for html_file in tqdm(folder.rglob('*.html')):
        try:
            with open(html_file, 'r', encoding='utf-8') as f:
                html_contents[html_file.name] = f.read()

        except Exception as e:
            print(f"Error reading {html_file}: {str(e)}")

    return html_contents


html_dir_path = "/gpfs/public/research/public/htmls"

result = read_html_files_raw(html_dir_path)

with open("/gpfs/public/research/public/htmls_json/htmls_upload.json", "w") as fo:
    fo.write(json.dumps(result))

