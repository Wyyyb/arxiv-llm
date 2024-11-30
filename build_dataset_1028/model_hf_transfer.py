from huggingface_hub import HfApi, create_repo, repo_exists
from pathlib import Path
import os
import time


def upload_model_to_hf(
        input_model_dir_path: str,
        repo_name: str,
        hf_token: str,
        commit_message: str = "Upload model",
        organization: str = None,
        max_retries: int = 5
) -> str:
    """
    将本地模型上传到 Hugging Face Hub

    Args:
        input_model_dir_path: 本地模型目录的路径
        repo_name: 要创建的仓库名称
        hf_token: Hugging Face 的访问令牌
        commit_message: 提交信息
        organization: 组织名称（可选）
        max_retries: 最大重试次数

    Returns:
        str: 模型在HF上的URL
    """
    try:
        # 初始化API
        api = HfApi()

        # 设置完整的仓库名称
        if organization:
            full_repo_name = f"{organization}/{repo_name}"
        else:
            full_repo_name = repo_name

        # 创建私有仓库
        print(f"Creating repository: {full_repo_name}")
        create_repo(
            repo_id=full_repo_name,
            token=hf_token,
            private=True,
            repo_type="model",
            exist_ok=True
        )

        # 等待并验证仓库创建成功
        for i in range(max_retries):
            if repo_exists(repo_id=full_repo_name, token=hf_token):
                print("Repository created successfully!")
                break
            if i < max_retries - 1:
                print(f"Waiting for repository to be ready... (attempt {i + 1}/{max_retries})")
                time.sleep(5)
        else:
            raise Exception("Repository creation could not be verified after maximum retries")

        # 获取本地模型目录中的所有文件
        path = Path(input_model_dir_path)
        files = [str(x) for x in path.glob("**/*") if x.is_file()]

        # 上传所有文件
        for file in files:
            # 计算相对路径
            relative_path = os.path.relpath(file, input_model_dir_path)

            # 重试上传
            for attempt in range(max_retries):
                try:
                    print(f"Uploading: {relative_path} (attempt {attempt + 1}/{max_retries})")
                    api.upload_file(
                        path_or_fileobj=file,
                        path_in_repo=relative_path,
                        repo_id=full_repo_name,
                        token=hf_token,
                        commit_message=commit_message
                    )
                    print(f"Successfully uploaded: {relative_path}")
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    print(f"Upload failed, retrying... Error: {str(e)}")
                    time.sleep(5)

        # 返回模型URL
        return f"https://huggingface.co/{full_repo_name}"

    except Exception as e:
        print(f"Error uploading model: {str(e)}")
        raise


# 使用示例：
if __name__ == "__main__":
    HF_TOKEN = "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"  # 最好从环境变量获取

    # 调用函数
    model_url = upload_model_to_hf(
        input_model_dir_path="/gpfs/public/research/xy/yubowang/arxiv-llm/model_output/v1127_multi_cite/checkpoint-2000/",
        repo_name="scholar-copilot-7B-ckpt2000-1130",
        hf_token=HF_TOKEN,
        commit_message="Initial model upload",
        organization=None  # 如果要上传到组织下，填写组织名称
    )

    print(f"Model uploaded successfully! URL: {model_url}")