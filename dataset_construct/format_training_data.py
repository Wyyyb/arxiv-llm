import os
import re
from transformers import AutoTokenizer
import json
import time
import torch
import random
from tqdm import tqdm


def transfer_single(item, tokenizer):
    arxiv_id = item['arxiv_id']
    paper = item['paper']
    targets = item['targets']

    # 找出所有cite tokens的位置
    token_positions = []
    for token in targets.keys():
        start = paper.find(token)
        if start != -1:
            token_positions.append((start, token))

    # 按位置排序
    token_positions.sort()

    # 如果cite tokens少于4个，直接返回空列表
    if len(token_positions) < 4:
        return []

    # 构建parts列表
    parts = []
    last_end = 0
    for start, token in token_positions:
        if start > last_end:
            parts.append(('text', paper[last_end:start]))
        parts.append(('cite', token))
        last_end = start + len(token)
    if last_end < len(paper):
        parts.append(('text', paper[last_end:]))

    # 计算token数量
    all_texts = [part[1] for part in parts if part[0] == 'text']
    all_citations = [targets[part[1]] for part in parts if part[0] == 'cite']

    text_token_nums = batch_compute_tokens(tokenizer, all_texts) if all_texts else []
    citation_token_nums = batch_compute_tokens(tokenizer, all_citations) if all_citations else []

    # 重建parts列表，加入token数量信息
    text_idx = 0
    citation_idx = 0
    parts_with_tokens = []
    for part_type, content in parts:
        if part_type == 'text':
            token_num = text_token_nums[text_idx]
            text_idx += 1
        else:  # cite
            token_num = citation_token_nums[citation_idx]
            citation_idx += 1
        parts_with_tokens.append((part_type, content, token_num))

    # 构建segments
    segments = []
    segment_id = 0
    i = 0
    while i < len(parts_with_tokens):
        current_segment = {
            'parts': [],
            'total_tokens': 0,
            'cite_count': 0,
            'last_cite_idx': -1
        }

        while i < len(parts_with_tokens):
            part_type, content, token_num = parts_with_tokens[i]

            # 如果单个token就超过限制，直接跳过这个token
            if token_num > 16000:
                i += 1
                continue

            new_total = current_segment['total_tokens'] + token_num

            # 如果是cite token，更新last_cite_idx
            if part_type == 'cite':
                # 检查与上一个cite token之间的距离
                if current_segment['cite_count'] > 0:
                    tokens_between = sum(p[2] for p in parts_with_tokens[current_segment['last_cite_idx'] + 1:i])
                    if tokens_between > 16000:
                        # 如果之前的segment合格，在上一个cite token后切分
                        if current_segment['cite_count'] >= 4:
                            segment_paper = ''.join(part[1] for part in current_segment['parts'])
                            segment_targets = {part[1]: targets[part[1]]
                                               for part in current_segment['parts']
                                               if part[0] == 'cite'}
                            segments.append({
                                'arxiv_id': f"{arxiv_id}-{segment_id}",
                                'paper': segment_paper,
                                'targets': segment_targets
                            })
                            segment_id += 1
                            i = current_segment['last_cite_idx'] + 1
                            break
                        else:
                            # 放弃当前segment，从下一个位置重新开始
                            i = current_segment['last_cite_idx'] + 1
                            break

            # 检查是否超过token限制
            if new_total > 16000:
                # 回退到上一个cite token
                if current_segment['cite_count'] >= 4:
                    # 截断到上一个cite token的位置
                    current_segment['parts'] = current_segment['parts'][:-(i - current_segment['last_cite_idx'] - 1)]
                    segment_paper = ''.join(part[1] for part in current_segment['parts'])
                    segment_targets = {part[1]: targets[part[1]]
                                       for part in current_segment['parts']
                                       if part[0] == 'cite'}
                    segments.append({
                        'arxiv_id': f"{arxiv_id}-{segment_id}",
                        'paper': segment_paper,
                        'targets': segment_targets
                    })
                    segment_id += 1
                    i = current_segment['last_cite_idx'] + 1
                else:
                    # 如果cite tokens不够4个，放弃当前segment
                    if current_segment['last_cite_idx'] >= 0:
                        i = current_segment['last_cite_idx'] + 1
                    else:
                        # 如果连一个cite token都没有，从下一个位置重新开始
                        i += 1
                break

            # 添加当前部分到segment
            current_segment['parts'].append((part_type, content))
            current_segment['total_tokens'] = new_total
            if part_type == 'cite':
                current_segment['cite_count'] += 1
                current_segment['last_cite_idx'] = i

            i += 1

        # 处理最后一个segment
        if i >= len(parts_with_tokens) and current_segment['cite_count'] >= 4:
            segment_paper = ''.join(part[1] for part in current_segment['parts'])
            segment_targets = {part[1]: targets[part[1]]
                               for part in current_segment['parts']
                               if part[0] == 'cite'}
            segments.append({
                'arxiv_id': f"{arxiv_id}-{segment_id}",
                'paper': segment_paper,
                'targets': segment_targets
            })

    return segments


def batch_compute_tokens(tokenizer, text_list):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("device", device)
    encoded = tokenizer(text_list, add_special_tokens=True, padding=True,
                        truncation=True, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}
    token_num_list = encoded['input_ids'].ne(tokenizer.pad_token_id).sum(dim=1).cpu().tolist()
    return token_num_list


def split_by_cite_tokens(text):
    # Define patterns for both types of tokens
    multi_cite_pattern = r'<\|multi_cite_token\$\d+\$\|>'
    cite_pattern = r'<\|cite_token\$\d+\$\|>'

    # Combine patterns
    combined_pattern = f"({multi_cite_pattern}|{cite_pattern})"

    # Split the text and keep the delimiters
    segments = re.split(f'({combined_pattern})', text)

    return segments


def format_data(input_dir, output_path):
    # tokenizer = AutoTokenizer.from_pretrained("/gpfs/public/research/xy/yubowang/models/Qwen2.5-7B")
    # tokenizer = AutoTokenizer.from_pretrained("../local/Qwen2.5-7B")
    tokenizer = AutoTokenizer.from_pretrained("/data/yubowang/models/qwen2.5-1.5b/")
    special_tokens = ['<|paper_start|>', '<|paper_end|>', '<|cite_start|>', '<|cite_end|>',
                      '<|reference_start|>', '<|reference_end|>']
    res_train = []
    res_val = []
    tokenizer.add_tokens(special_tokens)
    for file in os.listdir(input_dir):
        if not file.endswith(".json"):
            continue
        print("processing file...", file)
        file_path = os.path.join(input_dir, file)
        curr = process_single_base(tokenizer, file_path)
        if file.startswith("2409_"):
            for each in curr:
                curr_res = post_format_training(each, tokenizer)
                if not curr_res:
                    continue
                res_val.append(curr_res)
        else:
            for each in curr:
                curr_res = post_format_training(each, tokenizer)
                if not curr_res:
                    continue
                res_train.append(curr_res)
    print("original val data num:", len(res_val))
    res_val = check_res_token_num(res_val, tokenizer)
    print("final val data num:", len(res_val))
    save_res(res_val, output_path.replace("train_data", "val_data"))
    print("original train data num:", len(res_train))
    res_train = check_res_token_num(res_train, tokenizer)
    print("final train data num:", len(res_train))
    save_res(res_train, output_path)


def check_res_token_num(res, tokenizer):
    total_token = 0
    text_list = []
    id_list = []
    batch_size = 256
    wrong_data_id = []
    for each in res:
        id_list.append(each["arxiv_id"])
        text_list.append(each["paper"])
    for i in tqdm(range(0, len(text_list), batch_size)):
        batch = text_list[i:i + batch_size]
        token_num_list = batch_compute_tokens(tokenizer, batch)
        for j, token_num in enumerate(token_num_list):
            if token_num > 16100:
                curr_id = id_list[j + i]
                wrong_data_id.append(curr_id)
            else:
                total_token += token_num
    result = []
    for i, each in enumerate(res):
        if i in wrong_data_id:
            print("wrong_data_id", i)
            print("wrong data", each["arxiv_id"])
            continue
        result.append(each)
    print("total token number:", total_token)
    print("average token number:", total_token // len(result))
    return result


def save_res(res_data, output_path):
    print("saving res...")
    with open(output_path, "w") as fo:
        for each in tqdm(res_data):
            fo.write(json.dumps(each))
            fo.write("\n")


def process_single_base(tokenizer, base_path):
    result = []
    count = 0
    with open(base_path, 'r') as fi:
        ori_data = json.load(fi)
        for k, v in tqdm(ori_data.items()):
            each = v["data"]
            if not v["satisfied_data"]:
                continue
            # check_res = post_format_training(each, tokenizer)
            # if not check_res:
            #     print(each["arxiv_id"])
            #     count += 1
            #     print("count", count)
            curr = transfer_single(each, tokenizer)
            s = 1
            for seg in curr:
                if not seg["paper"] or seg["paper"] == "":
                    continue
                result.append(seg)
    return result


def post_format_training(item, tokenizer):
    paper = item["paper"]
    targets = item["targets"]
    new_targets = []
    for k, v in targets.items():
        if k not in paper:
            print("error: k not in paper", k, paper)
            return None
        abstract = v.replace("<|reference_start|>", "").replace("<|reference_end|>", "")
        abstract = "<|cite_start|>(Reference: " + abstract + ")<|cite_end|>"
        paper = paper.replace(k, abstract)
        # token_num = batch_compute_tokens(tokenizer, [paper])[0]
        # # print("checking token_num: ", token_num)
        # if token_num > 16010:
        #     print("token num exceed 16k", item)
        #     return None
    ori_keys = list(targets.keys())
    new_keys = ori_keys
    random.shuffle(new_keys)
    selected_keys = new_keys[:4]
    targets_id = []
    for each in selected_keys:
        ori_index = ori_keys.index(each)
        targets_id.append(ori_index)
        new_targets.append(targets[each])
    new_item = {"arxiv_id": item["arxiv_id"], "paper": paper, "targets": new_targets, "targets_id": targets_id}
    return new_item


if __name__ == '__main__':
    # format_data("../local/arxiv_base_1025_sample/", "../local/training_data_1025_sample/train_data.jsonl")
    format_data("../local/arxiv_base_1025/", "../local/training_data_1025/train_data_1025.jsonl")
