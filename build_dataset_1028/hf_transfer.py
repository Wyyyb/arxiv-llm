from datasets import Dataset
from huggingface_hub import HfApi, login
from datasets import load_dataset
import os
import json
from tqdm import tqdm

from datasets import Dataset, DatasetDict
import json
from huggingface_hub import HfApi, login
from tqdm import tqdm


def upload_jsonl_files_to_hf(train_path, eval_path, repo_id, token=None, private=True):
    """
    将训练集和验证集的jsonl文件上传到Hugging Face

    Args:
        train_path: 训练集jsonl文件的路径
        eval_path: 验证集jsonl文件的路径
        repo_id: Hugging Face上的仓库ID，格式为 "username/repo_name"
        token: Hugging Face的访问令牌，如果为None则需要之前已经登录
        private: 是否创建私有仓库，默认为True
    """
    # 如果提供了token就登录
    if token:
        login(token)

    # 读取训练集
    train_data = []
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading training data"):
            train_data.append(json.loads(line))

    # 读取验证集
    eval_data = []
    with open(eval_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading eval data"):
            eval_data.append(json.loads(line))

    # 转换为Dataset格式
    dataset_dict = DatasetDict({
        'train': Dataset.from_list(train_data),
        'validation': Dataset.from_list(eval_data)
    })

    # 上传到Hugging Face
    dataset_dict.push_to_hub(
        repo_id,
        private=private,
        token=token
    )

    print(f"Successfully uploaded to {repo_id}")
    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(eval_data)}")

    return dataset_dict


def download_and_save_jsonl(repo_id, output_dir, token=None):
    """
    从Hugging Face下载数据集并保存为jsonl文件

    Args:
        repo_id: Hugging Face上的仓库ID，格式为 "username/repo_name"
        output_dir: 输出目录路径
        token: Hugging Face的访问令牌，如果为None则需要之前已经登录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 下载数据集
    dataset = load_dataset(repo_id, token=token)

    # 保存训练集
    train_path = os.path.join(output_dir, 'train_data.jsonl')
    with open(train_path, 'w', encoding='utf-8') as f:
        for item in tqdm(dataset['train'], desc="Saving training data"):
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # 保存验证集
    eval_path = os.path.join(output_dir, 'eval_data.jsonl')
    with open(eval_path, 'w', encoding='utf-8') as f:
        for item in tqdm(dataset['validation'], desc="Saving validation data"):
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Successfully downloaded and saved files to {output_dir}")
    print(f"Training set size: {len(dataset['train'])}")
    print(f"Validation set size: {len(dataset['validation'])}")

    return {
        'train_path': train_path,
        'eval_path': eval_path,
        'train_size': len(dataset['train']),
        'eval_size': len(dataset['validation'])
    }


hf_token = "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"
upload_jsonl_files_to_hf(
    train_path="/data/yubowang/arxiv-llm/local_1125/train_data_1125.jsonl",
    eval_path="/data/yubowang/arxiv-llm/local_1125/eval_data_1125.jsonl",
    repo_id="ubowang/cite-llm-single-cite-1127",
    token=hf_token
)

