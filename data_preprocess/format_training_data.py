import json
import os
import random
from tqdm import tqdm
import re


def remove_citations(latex_string):
    patterns = [
        r'(~?\s*\\cite(?:\s*\[.*?\])?\s*{[^}]+})',
        r'(~?\s*\\cites(?:\s*\[.*?\])?\s*{[^}]+})',
        r'(~?\s*\\citet(?:\s*\[.*?\])?\s*{[^}]+})',
        r'(~?\s*\\citep(?:\s*\[.*?\])?\s*{[^}]+})',
        r'(~?\s*\\citealp\s*{[^}]+})',
        r'(~?\s*\\citeauthor\s*{[^}]+})',
        r'(~?\s*\\citeyear\s*{[^}]+})',
        r'(~?\s*\\citealt\s*{[^}]+})',
        r'(~?\s*\\parencite\s*{[^}]+})',
        r'(~?\s*\\textcite\s*{[^}]+})',
        r'(~?\s*\\autocite\s*{[^}]+})'
    ]

    # 合并所有模式为一个正则表达式，使用 | 分隔
    combined_pattern = '|'.join(patterns)

    # 使用正则表达式替换所有匹配的引用为空字符串
    # 添加 re.DOTALL 和 re.MULTILINE 标志以匹配跨行的引用
    cleaned_latex = re.sub(combined_pattern, '', latex_string, flags=re.DOTALL | re.MULTILINE)

    return cleaned_latex


def format_single_file(file_path):
    res = []
    with open(file_path, "r") as fi:
        info = json.load(fi)
        for k, v in info.items():
            if not v["satisfied_data"]:
                continue
            data = v["data"]
            paper = "<|paper_start|>" + data["paper"].strip() + "<|paper_end|>"
            references = data["targets"]
            ori_targets = []
            ori_targets_idx = []
            index = 0
            for key, value in references.items():
                value = remove_citations(value)
                if key not in paper:
                    print("wrong format of data", file_path, data["arxiv_id"], key)
                    return None
                citation_str = "<|cite_start|>" + "(Reference: " + value.strip() + ")<|cite_end|>"
                paper = paper.replace(key, citation_str)
                ori_targets.append("<|reference_start|>" + value.strip() + "<|reference_end|>")
                ori_targets_idx.append(index)
                index += 1
            random.shuffle(ori_targets_idx)
            ori_targets_id = ori_targets_idx[:4]
            targets_idx = sorted(ori_targets_id)
            targets = []
            for each_index in targets_idx:
                targets.append(ori_targets[each_index])
            paper = remove_citations(paper)
            data["paper"] = paper
            data["targets"] = targets
            data["targets_idx"] = targets_idx
            res.append(data)
    return res


def format_data(input_dir):
    train_data = []
    val_data = []
    os.makedirs("../data_1018", exist_ok=True)
    train_data_path = "../data_1018/train_data_1016_$.jsonl"
    val_data_path = "../data_1018/val_data_1016.jsonl"
    for each_file in tqdm(os.listdir(input_dir)):
        if not each_file.endswith(".json"):
            continue
        print("Processing", each_file)
        file_path = os.path.join(input_dir, each_file)
        seg_1 = each_file.split("_")[0]
        if seg_1 in ["2408", "2409"]:
            val_data += format_single_file(file_path)
        else:
            train_data += format_single_file(file_path)
    train_data_num = len(train_data)
    val_data_num = len(val_data)
    print("Training data number", train_data_num)
    print("Validation data number", val_data_num)
    with open(val_data_path, "w") as fo:
        for each in val_data:
            fo.write(json.dumps(each))
            fo.write("\n")
        # fo.write(json.dumps(val_data))
    batch_size = 10000
    i = 0
    while i < train_data_num:
        end = min(i + batch_size, train_data_num)
        curr_batch = train_data[i: end]
        curr_path = train_data_path.replace("$", str(i // batch_size))
        i += batch_size
        with open(curr_path, "w") as fo:
            for each in curr_batch:
                fo.write(json.dumps(each))
                fo.write("\n")
            # fo.write(json.dumps(curr_batch))


if __name__ == "__main__":
    format_data("../local/arxiv_base")
    # format_data("/home/yubo/cite_llm/local/arxiv_base/")














