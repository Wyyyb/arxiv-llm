from huggingface_hub import HfApi
import os


def upload_model_to_hf(
        local_path="/gpfs/public/research/xy/yubowang/arxiv-llm/model_output/v1127_multi_cite/checkpoint-2000/",
        repo_name="scholar-copilot-7B-ckpt2000-1130",
        token=None
):
    """
    上传本地模型到 Hugging Face Hub，如果仓库存在则覆盖内容

    Args:
        local_path: 本地模型路径
        repo_name: HF上的仓库名称
        token: HF API token
    """
    if token is None:
        raise ValueError("Please provide your Hugging Face API token")

    # 初始化API
    api = HfApi()

    try:
        # 检查仓库是否存在
        try:
            repo_info = api.repo_info(
                repo_id=repo_name,
                token=token
            )
            print(f"Repository {repo_name} already exists, will override content")
        except Exception:
            # 仓库不存在，创建新的私有仓库
            api.create_repo(
                repo_id=repo_name,
                private=True,
                token=token
            )
            print(f"Created new repository: {repo_name}")

        # 上传模型文件，设置 repo_type="model" 来指定这是一个模型仓库
        api.upload_folder(
            folder_path=local_path,
            repo_id=repo_name,
            repo_type="model",
            token=token,
            ignore_patterns=["*.pyc", ".git*", ".DS_Store"],  # 忽略一些不必要的文件
        )

        print(f"Successfully uploaded model to: https://huggingface.co/{repo_name}")

    except Exception as e:
        print(f"Error occurred during upload: {str(e)}")


upload_model_to_hf(token="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")


