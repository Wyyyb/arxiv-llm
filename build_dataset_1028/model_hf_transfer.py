from huggingface_hub import HfApi, create_repo, upload_folder
import os


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


def main():
    # 用户变量
    repo_id = "ubowang/scholar-copilot-7B-ckpt2000-1130"
    local_dir = "/gpfs/public/research/xy/yubowang/arxiv-llm/model_output/v1127_multi_cite/checkpoint-2000/"
    hf_token = "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"

    # 调用函数
    upload_model_to_hf_hub(repo_id, local_dir, hf_token)


main()

