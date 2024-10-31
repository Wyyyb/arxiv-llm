import os
import re
from transformers import AutoTokenizer
import json
import time
import torch
import random
from tqdm import tqdm


def batch_compute_tokens(tokenizer, text_list):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("device", device)
    encoded = tokenizer(text_list, add_special_tokens=True, padding=True,
                        truncation=True, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}
    token_num_list = encoded['input_ids'].ne(tokenizer.pad_token_id).sum(dim=1).cpu().tolist()
    return token_num_list


def split_text_with_citations(text):
    """
    Split text into segments based on citation patterns <|cite_X|> and <|multi_cite_X_Y|>.

    Args:
        text (str): Input text containing citations

    Returns:
        list: List of text segments and citations
    """
    import re

    # Define citation pattern
    pattern = r'<\|(?:cite_\d+|multi_cite_\d+_\d+)\|>'

    # Find all matches and their positions
    matches = []
    for match in re.finditer(pattern, text):
        matches.append((match.start(), match.end(), match.group()))

    # Split text into segments
    result = []
    last_end = 0

    for start, end, citation in matches:
        # Add text before citation if it exists
        if start > last_end:
            segment = text[last_end:start].strip()
            if segment:
                result.append(segment)

        # Add citation
        result.append(citation)
        last_end = end

    # Add remaining text if it exists
    if last_end < len(text):
        segment = text[last_end:].strip()
        if segment:
            result.append(segment)

    return result


def format_abs(abstract):
    res = f" <|cite_start|> (Reference: {abstract}) <|cite_end|>"
    return res


def single_process_data(paper_dir_path, tokenizer, use_multi_cite=False):
    step_5_data_path = os.path.join(paper_dir_path, "step_5_info.json")
    if not os.path.exists(step_5_data_path):
        return None
    with open(step_5_data_path, "r") as fi:
        step_5_data = json.load(fi)
    # first format and filter invalid citation tokens
    arxiv_id = step_5_data.get('arxiv_id')
    title = step_5_data.get('title', None)
    abstract = step_5_data.get('abstract', None)
    intro = step_5_data.get('full_intro', None)
    bib_info = step_5_data.get('bib_info', None)
    if not (title and abstract and intro and bib_info):
        return None
    title = title.strip()
    abstract = abstract.strip().replace("<|reference_start|>", "").replace("<|reference_end|>", "")
    paper = "<|paper_start|> " + f"Title: {title}\nAbstract: {abstract}\n" + intro + " <|paper_end|>"
    segments = split_text_with_citations(paper)
    valid_cite_tokens_map = {}
    invalid_cite_tokens = []
    existing_multi_index = []
    for seg in segments:
        if seg.startswith("<|cite_") and seg.endswith("|>"):
            if seg not in bib_info:
                invalid_cite_tokens.append(seg)
                continue
            info = bib_info[seg]
            abstract = info["abstract"]
            if abstract and len(abstract) > 10:
                valid_cite_tokens_map[seg] = format_abs(abstract)
            else:
                invalid_cite_tokens.append(seg)
        if use_multi_cite:
            if seg.startswith("<|multi_cite_") and seg.endswith("|>"):
                if seg not in bib_info:
                    invalid_cite_tokens.append(seg)
                    continue
                info = bib_info[seg]
                abstract = info["abstract"]
                if abstract and len(abstract) > 10:
                    valid_cite_tokens_map[seg] = format_abs(abstract)
                else:
                    invalid_cite_tokens.append(seg)
        else:  # do not use multi cite for initial setting
            if seg.startswith("<|multi_cite_") and seg.endswith("|>"):
                if seg not in bib_info:
                    invalid_cite_tokens.append(seg)
                    continue
                info = bib_info[seg]
                abstract = info["abstract"]
                if abstract and len(abstract) > 10:
                    multi_cite_index = seg.split("_")[2]
                    if multi_cite_index not in existing_multi_index:
                        valid_cite_tokens_map[seg] = format_abs(abstract)
                    else:
                        invalid_cite_tokens.append(seg)
                else:
                    invalid_cite_tokens.append(seg)

    # remove invalid cite tokens
    for each in invalid_cite_tokens:
        paper = paper.replace(each, "")
    # check if paper with cite abstract will exceed 16K limits
    final_paper = paper
    for k, v in valid_cite_tokens_map.items():
        final_paper = final_paper.replace(k, v)
    token_num = batch_compute_tokens(tokenizer, [final_paper])[0]
    if token_num <= 16000:
        result = {"arxiv_id": arxiv_id, "paper": paper, "bib_info_map": valid_cite_tokens_map}
        return [result]
    # second time split the paper without invalid cite tokens
    segments = split_text_with_citations(paper)
    result = []
    part_id = 0
    curr_segs = []
    curr_length = 0
    next_seg_id = 0
    while curr_length < 16000:
        if next_seg_id >= len(segments):
            break
        if segments[next_seg_id].startswith("<|multi_cite_") or segments[next_seg_id].startswith("<|cite_"):
            next_seg = valid_cite_tokens_map[segments[next_seg_id]]
        else:
            next_seg = segments[next_seg_id]
        next_seg_token_num = batch_compute_tokens(tokenizer, [next_seg])[0]
        if curr_length + next_seg_token_num < 16000:
            curr_length += next_seg_token_num
            curr_segs.append(segments[next_seg_id])
            next_seg_id += 1
        else:
            result = save_curr_res_to_result(tokenizer, curr_segs, result, valid_cite_tokens_map, part_id, arxiv_id)
            part_id += 1
            curr_segs = []
            curr_length = 0
    if check_curr_segs(curr_segs):
        result = save_curr_res_to_result(tokenizer, curr_segs, result, valid_cite_tokens_map, part_id, arxiv_id)
        part_id += 1
    return result


def check_token_limits(tokenizer, paper):
    token_num = batch_compute_tokens(tokenizer, [paper])[0]
    return token_num <= 16000


def save_curr_res_to_result(tokenizer, curr_segs, result, valid_cite_tokens_map, part_id, arxiv_id):
    paper = "".join(curr_segs)
    if not check_token_limits(tokenizer, paper):
        print("split before has error, wrong paper token number computed")
        return result
    else:
        print("successfully save a piece of data", arxiv_id, part_id)
    bib_info_map = {}
    for each in curr_segs:
        if each.startswith("<|multi_cite_") or each.startswith("<|cite_"):
            bib_info_map[each] = valid_cite_tokens_map[each]
    new_arxiv_id = arxiv_id + "-" + str(part_id)
    curr_res = {"arxiv_id": new_arxiv_id, "paper": paper, "bib_info_map": bib_info_map}
    result.append(curr_res)
    return result


def check_curr_segs(curr_segs):
    cite_token_num = 0
    for each in curr_segs:
        if each.startswith("<|multi_cite_") or each.startswith("<|cite_"):
            cite_token_num += 1
    return cite_token_num >= 4


def compute_result_token_number(tokenizer, result_item):
    paper = result_item["paper"]
    for k, v in result_item["bib_info_map"].items():
        paper = paper.replace(k, v)
    token_number = batch_compute_tokens(tokenizer, [paper])[0]
    return token_number


def run_on_darth_server(input_dir):
    total_token_num = 0.0
    valid_data_num = 0.0
    tokenizer = AutoTokenizer.from_pretrained("/data/yubowang/models/qwen2.5-1.5b/")
    special_tokens = ['<|paper_start|>', '<|paper_end|>', '<|cite_start|>', '<|cite_end|>',
                      '<|reference_start|>', '<|reference_end|>']
    tokenizer.add_tokens(special_tokens)
    for sub_dir in os.listdir(input_dir):
        print("Processing", sub_dir)
        if os.path.isdir(os.path.join(input_dir, sub_dir)):
            for paper_dir in tqdm(os.listdir(os.path.join(input_dir, sub_dir))):
                if not paper_dir.startswith(sub_dir):
                    print("skip", paper_dir)
                    continue
                paper_dir_path = os.path.join(input_dir, sub_dir, paper_dir)
                result = single_process_data(paper_dir_path, tokenizer, False)
                if result is None:
                    continue
                with open(os.path.join(paper_dir_path, "step_7_info.json"), "w") as fo:
                    fo.write(json.dumps(result))
                for each in result:
                    total_token_num += compute_result_token_number(tokenizer, each)
                    valid_data_num += 1
        print("total_token_num", total_token_num)
        print("valid_data_num", valid_data_num)
        print("average token number per data", total_token_num / valid_data_num)


if __name__ == "__main__":
    run_on_darth_server("/data/yubowang/arxiv_plain_latex_data_1028")







