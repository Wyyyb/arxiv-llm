from datasets import Dataset
from huggingface_hub import HfApi, login
from datasets import load_dataset
import os
import json
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


def download_hf_to_jsonl(repo_id, output_path, token=None):
    """
    从Hugging Face下载数据集并保存为JSONL文件

    Args:
        repo_id: Hugging Face上的仓库ID，格式为 "username/repo_name"
        output_path: 输出的JSONL文件路径
        token: Hugging Face的访问令牌，如果访问私有仓库则需要提供
    """
    # 从Hugging Face加载数据集
    dataset = load_dataset(repo_id, token=token)

    # 如果数据集有多个分片（如train, test等），通常会返回DatasetDict
    # 我们需要确定使用哪个分片
    if hasattr(dataset, 'keys'):
        # 默认使用第一个分片
        split_name = list(dataset.keys())[0]
        dataset = dataset[split_name]

    # 将数据集保存为JSONL文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in tqdm(dataset, desc="Writing JSONL"):
            # 将每条数据转换为JSON字符串并写入文件
            json_str = json.dumps(item, ensure_ascii=False)
            f.write(json_str + '\n')

    print(f"Successfully downloaded and saved to {output_path}")
    print(f"Total records: {len(dataset)}")

    return dataset


hf_token = "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"
# upload_jsonl_to_hf(
#     jsonl_path="/data/yubowang/arxiv-llm/local_1125/train_data_1125.jsonl",
#     repo_id="ubowang/cite-llm-single-cite-train",
#     token=hf_token
# )
#
# upload_jsonl_to_hf(
#     jsonl_path="/data/yubowang/arxiv-llm/local_1125/eval_data_1125.jsonl",
#     repo_id="ubowang/cite-llm-single-cite-eval",
#     token=hf_token
# )

upload_jsonl_to_hf(
    jsonl_path="/data/yubowang/arxiv-llm/corpus_data/corpus_data_1124.jsonl",
    repo_id="ubowang/cite-llm-corpus_data_1124",
    token=hf_token
)

os.makedirs("../local_1125", exist_ok=True)
download_hf_to_jsonl(repo_id="ubowang/cite-llm-single-cite-train",
                     output_path="../local_1125/train_data_1125.jsonl",
                     token=hf_token)

download_hf_to_jsonl(repo_id="ubowang/cite-llm-single-cite-eval",
                     output_path="../local_1125/eval_data_1125.jsonl",
                     token=hf_token)
