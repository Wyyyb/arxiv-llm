import json
import os
import random
from tqdm import tqdm
import time
from transformers import AutoTokenizer
import torch


def transfer_single(file_dir):
    step_7_info_path = os.path.join(file_dir, "step_7_1124_info.json")
    if os.path.exists(step_7_info_path):
        with open(step_7_info_path, "r") as fi:
            step_7_info_list = json.load(fi)
    else:
        return []
    if step_7_info_list is None:
        return []
    step_8_info_list = []
    for step_7_info in step_7_info_list:
        if len(step_7_info["bib_info_map"]) < 4:
            continue
        cite_token_list = []
        cite_content_list = []
        reference_list = []
        for cite_token, cite_content in step_7_info["bib_info_map"].items():
            cite_token_list.append(cite_token)
            cite_content_list.append(cite_content)
            reference = trans_to_reference(cite_content)
            reference_list.append(reference)
        selected_id = [_ for _ in range(len(cite_token_list))]
        random.shuffle(selected_id)
        selected_id = selected_id[:4]
        # print("selected_id", selected_id)
        selected_id = sorted(selected_id)
        # print("targets_idx", selected_id)
        paper = step_7_info["paper"]
        arxiv_id = step_7_info["arxiv_id"]
        targets = []
        for i, cite_token in enumerate(cite_token_list):
            if cite_token not in paper:
                print("Error: Cannot find", cite_token, "in paper", arxiv_id)
                continue
            paper = paper.replace(cite_token, cite_content_list[i])
            if i in selected_id:
                targets.append(reference_list[i])
        step_8_info = {"arxiv_id": arxiv_id, "paper": paper, "targets": targets, "targets_idx": selected_id}
        step_8_info_list.append(step_8_info)
        # print("step_8_info", step_8_info)
        # time.sleep(10)
    return step_8_info_list


def trans_to_reference(cite_content):
    res = cite_content.replace(" <|cite_start|> (Reference: ", "<|reference_start|> ")
    res = res.replace(") <|cite_end|>", " <|reference_end|>")
    return res


def batch_compute_tokens(tokenizer, text_list):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("device", device)
    encoded = tokenizer(text_list, add_special_tokens=True, padding=True,
                        truncation=True, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}
    token_num_list = encoded['input_ids'].ne(tokenizer.pad_token_id).sum(dim=1).cpu().tolist()
    return token_num_list


def check_train_data_token_num(tokenizer, train_data):
    res_data = []
    batch_size = 256
    i = 0
    exceed_count = 0
    while i < len(train_data):
        end = min(len(train_data), i + batch_size)
        curr_batch = []
        for j in range(i, end):
            curr_batch.append(train_data[j]["paper"])
        token_num_list = batch_compute_tokens(tokenizer, curr_batch)
        for k in range(len(token_num_list)):
            if token_num_list[k] > 16100:
                print("exceed 16100 length\n", "count:", exceed_count)
                exceed_count += 1
                continue
            res_data.append(train_data[i + k])
        i = end
    return res_data


def run_on_darth_server(input_dir):
    tokenizer = AutoTokenizer.from_pretrained("/data/yubowang/models/qwen2.5-1.5b/")
    special_tokens = ['<|paper_start|>', '<|paper_end|>', '<|cite_start|>', '<|cite_end|>',
                      '<|reference_start|>', '<|reference_end|>']
    tokenizer.add_tokens(special_tokens)
    train_data = []
    for sub_dir in os.listdir(input_dir):
        print("Processing", sub_dir)
        if os.path.isdir(os.path.join(input_dir, sub_dir)):
            for paper_dir in tqdm(os.listdir(os.path.join(input_dir, sub_dir))):
                if not paper_dir.startswith(sub_dir):
                    print("skip", paper_dir)
                    continue
                paper_dir_path = os.path.join(input_dir, sub_dir, paper_dir)
                curr_res = transfer_single(paper_dir_path)
                if not curr_res:
                    continue
                for each in curr_res:
                    train_data.append(each)
    print("ori train data number", len(train_data))
    train_data = check_train_data_token_num(tokenizer, train_data)
    print("train data number", len(train_data))
    os.makedirs("../local_1125", exist_ok=True)
    train_data_path = "../local_1125/train_data_1125.jsonl"
    with open(train_data_path, "w") as fo:
        for each in train_data:
            fo.write(json.dumps(each))
            fo.write("\n")


if __name__ == "__main__":
    run_on_darth_server("/data/yubowang/arxiv_plain_latex_data_1028")


