from huggingface_hub import HfApi, create_repo, upload_folder, snapshot_download
import os
from tqdm import tqdm


def upload_model_to_hf_hub(repo_id, local_dir, hf_token):
    """
    使用 HTTP API 上传模型到 Hugging Face Hub

    参数：
    - repo_id: str, Hugging Face 仓库 ID
    - local_dir: str, 本地模型目录路径
    - hf_token: str, Hugging Face API token
    """
    # 检查目录是否存在
    if not os.path.exists(local_dir):
        raise FileNotFoundError(f"路径 {local_dir} 不存在，请检查路径！")

    # 创建或获取 Hugging Face 仓库
    api = HfApi()
    create_repo(repo_id=repo_id, token=hf_token, exist_ok=True)

    # 上传目录内容
    upload_folder(
        folder_path=local_dir,
        path_in_repo=".",  # 上传到仓库根目录
        repo_id=repo_id,
        token=hf_token
    )
    print(f"模型已成功上传到 Hugging Face Hub: https://huggingface.co/{repo_id}")


def upload():
    # 用户变量
    repo_id = "ubowang/scholar-copilot-7B-ckpt2000-1130"
    local_dir = "/gpfs/public/research/xy/yubowang/arxiv-llm/model_output/v1127_multi_cite/checkpoint-2000/"
    hf_token = "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"

    # 调用函数
    upload_model_to_hf_hub(repo_id, local_dir, hf_token)


def download_model_from_hf_hub(repo_id, local_dir, hf_token):
    """
    从 Hugging Face Hub 下载模型到本地目录，并显示下载进度

    参数：
    - repo_id: str, Hugging Face 仓库 ID
    - local_dir: str, 本地保存目录路径
    - hf_token: str, Hugging Face API token
    """
    print(f"开始从 {repo_id} 下载模型到 {local_dir}")

    # 确保目标目录存在
    os.makedirs(local_dir, exist_ok=True)

    # 下载仓库内容
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            token=hf_token,
            local_dir_use_symlinks=False,  # 不使用符号链接
            tqdm_class=tqdm  # 使用tqdm显示进度
        )
        print(f"模型已成功下载到: {local_dir}")
    except Exception as e:
        print(f"下载过程中发生错误: {str(e)}")


def download():
    repo_id = "ubowang/scholar-copilot-7B-ckpt2000-1130"
    local_dir = "/data/yubowang/arxiv-llm/model_output/v1127_multi_cite/checkpoint-2000"
    hf_token = "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"  # 替换为你的token

    # 下载模型
    download_model_from_hf_hub(repo_id, local_dir, hf_token)


# upload()
download()
