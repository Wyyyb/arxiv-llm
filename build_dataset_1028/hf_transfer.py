from datasets import Dataset
import json
from huggingface_hub import HfApi, login
from tqdm import tqdm


def upload_jsonl_to_hf(jsonl_path, repo_id, token=None, private=True):
    """
    将jsonl文件上传到Hugging Face

    Args:
        jsonl_path: jsonl文件的路径
        repo_id: Hugging Face上的仓库ID，格式为 "username/repo_name"
        token: Hugging Face的访问令牌，如果为None则需要之前已经登录
        private: 是否创建私有仓库，默认为True
    """
    # 如果提供了token就登录
    if token:
        login(token)

    # 读取jsonl文件
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading JSONL"):
            data.append(json.loads(line))

    # 转换为Dataset格式
    dataset = Dataset.from_list(data)

    # 上传到Hugging Face
    dataset.push_to_hub(
        repo_id,
        private=private,
        token=token
    )

    print(f"Successfully uploaded to {repo_id}")

    return dataset


token = "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"
upload_jsonl_to_hf(
    jsonl_path="/data/yubowang/arxiv-llm/local_1125/train_data_1125.jsonl",
    repo_id="ubowang/cite-llm",
    token=token
)

# 方式2：之前已经用huggingface-cli login登录过
# upload_jsonl_to_hf(
#     jsonl_path="path/to/your/file.jsonl",
#     repo_id="your_username/your_repo_name"
# )
