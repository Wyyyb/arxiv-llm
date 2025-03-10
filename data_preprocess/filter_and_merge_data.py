from transformers import AutoTokenizer
import json
import os
from tqdm import tqdm
import time
import torch


def is_within_token_limit(input_string, max_tokens=15000):
    # 加载Qwen2.5-7B的tokenizer
    tokenizer = AutoTokenizer.from_pretrained("/gpfs/public/research/xy/yubowang/models/Qwen2.5-7B")

    # 对输入字符串进行编码
    encoded = tokenizer.encode(input_string, add_special_tokens=True)

    # 获取token数量
    token_count = len(encoded)

    # 判断token数是否在限制范围内
    return token_count <= max_tokens


def batch_process_gpu(texts, tokenizer, max_tokens=15000, batch_size=64):
    results = []
    start_time = time.time()

    # 将处理移动到GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encoded = tokenizer(batch, add_special_tokens=True, padding=True, truncation=True, return_tensors="pt")

        # 将编码后的数据移到GPU
        encoded = {k: v.to(device) for k, v in encoded.items()}

        token_counts = encoded.attention_mask.sum(dim=1)
        batch_results = (token_counts <= max_tokens).cpu().tolist()
        results.extend(batch_results)

    end_time = time.time()
    return results, end_time - start_time


def process_data(input_dir="../data_bk/", output_dir="../data/"):
    os.makedirs(output_dir, exist_ok=True)
    train_data = []
    val_data = []
    exceed_number = 0
    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        print("Processing", file)
        if not file.endswith(".jsonl"):
            continue
        if file.startswith("train"):
            with open(file_path, "r") as fi:
                for line in tqdm(fi.readlines()):
                    curr = json.loads(line)
                    if is_within_token_limit(curr["paper"]):
                        train_data.append(curr)
                    else:
                        exceed_number += 1
        elif file.startswith("val"):
            with open(file_path, "r") as fi:
                for line in tqdm(fi.readlines()):
                    curr = json.loads(line)
                    if is_within_token_limit(curr["paper"]):
                        val_data.append(curr)
                    else:
                        exceed_number += 1
    train_output_path = os.path.join(output_dir, "train_data_1019.jsonl")
    val_output_path = os.path.join(output_dir, "val_data_1019.jsonl")
    print("exceed num", exceed_number)
    print("training data num", len(train_data))
    print("val data num", len(val_data))
    with open(train_output_path, "w") as fo:
        for each in train_data:
            fo.write(json.dumps(each))
            fo.write("\n")
    with open(val_output_path, "w") as fo:
        for each in val_data:
            fo.write(json.dumps(each))
            fo.write("\n")


process_data()













