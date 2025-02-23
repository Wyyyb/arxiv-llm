import requests
import os
import gzip
import json
from tqdm import tqdm
from datetime import datetime


def download_papers(api_key, output_dir="/data/yubowang/offline_ss_papers"):
    """
    下载 Semantic Scholar 论文摘要到本地

    参数:
    api_key: Semantic Scholar API密钥
    output_dir: 输出目录，默认为'downloaded_papers'
    """

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # API配置
    headers = {
        'x-api-key': api_key
    }
    dataset_url = "https://api.semanticscholar.org/datasets/v1/release/latest/dataset/papers"

    try:
        # 获取数据集元数据
        print("正在获取数据集元数据...")
        response = requests.get(dataset_url, headers=headers)
        response.raise_for_status()
        dataset_metadata = response.json()

        # 获取文件URL列表
        file_urls = dataset_metadata.get('files', [])
        if not file_urls:
            print("未找到可下载的文件")
            return

        print(f"找到 {len(file_urls)} 个文件待下载")

        # 下载每个文件
        for i, url in enumerate(file_urls, 1):
            # 生成输出文件名
            filename = f"papers_part_{i}.json.gz"
            output_path = os.path.join(output_dir, filename)
            if os.path.exists(output_path):
                print("skip", filename)
                continue

            print(f"\n正在下载文件 {i}/{len(file_urls)}: {filename}")

            try:
                # 使用流式下载
                response = requests.get(url, stream=True)
                response.raise_for_status()

                # 获取文件大小
                total_size = int(response.headers.get('content-length', 0))

                # 使用进度条下载
                with open(output_path, 'wb') as f:
                    with tqdm(total=total_size, unit='iB', unit_scale=True, desc="下载进度") as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))

                print(f"成功下载: {filename}")

                # 可选：解压并验证文件
                try:
                    with gzip.open(output_path, 'rb') as f:
                        # 读取前几行验证文件完整性
                        for _ in range(5):
                            json.loads(f.readline())
                    print(f"文件验证成功: {filename}")
                except Exception as e:
                    print(f"警告：文件可能损坏 {filename}: {str(e)}")

            except Exception as e:
                print(f"下载文件 {filename} 时出错: {str(e)}")
                continue

        print("\n下载完成！")
        print(f"文件保存在: {os.path.abspath(output_dir)}")

    except requests.exceptions.RequestException as e:
        print(f"获取数据集信息时出错: {str(e)}")
    except Exception as e:
        print(f"发生错误: {str(e)}")


# 使用示例
if __name__ == "__main__":
    # API_KEY = "xPw99ZZQlprx8uLPejCY8SM6H5HM8eA8jhoXaZ82"
    API_KEY = "vgPRBYMNV9asiaTwN5o5b7mH2f0HnOVM9yN0MWp6"
    download_papers(API_KEY, "/data/yubowang/arxiv-llm/offline_ss_paper")


