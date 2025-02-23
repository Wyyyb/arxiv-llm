import re
import os
import json
from tqdm import tqdm


invalid_step_2_count = 0


def extract_citations(text):
    # Dictionary to store citations
    cite_dict = {}

    # Counter for all citations
    counter = 1

    # Patterns for different citation commands
    cite_patterns = [
        r'(?:~|\s)*\\cite\{([^}]+)\}',
        r'(?:~|\s)*\\citep\{([^}]+)\}',
        r'(?:~|\s)*\\citet\{([^}]+)\}',
        r'(?:~|\s)*\\citeyear\{([^}]+)\}',
        r'(?:~|\s)*\\citeauthor\{([^}]+)\}',
        r'(?:~|\s)*\\citealt\{([^}]+)\}',
        r'(?:~|\s)*\\citealp\{([^}]+)\}',
        r'(?:~|\s)*\\citenum\{([^}]+)\}'
    ]

    def replace_match(match):
        nonlocal counter
        keys = match.group(1).split(',')

        if len(keys) == 1:
            # Single citation
            placeholder = f"<|cite_{counter}|>"
            cite_dict[placeholder] = keys[0].strip()
            counter += 1
            return placeholder
        else:
            # Multiple citations
            placeholders = []
            for i, key in enumerate(keys, 1):
                placeholder = f"<|multi_cite_{counter}_{i}|>"
                cite_dict[placeholder] = key.strip()
                placeholders.append(placeholder)
            counter += 1
            return ''.join(placeholders)

    # Process text for each citation pattern
    processed_text = text
    for pattern in cite_patterns:
        processed_text = re.sub(pattern, replace_match, processed_text)

    return processed_text, cite_dict


def extract_bibitem_key(bibitem):
    import re

    # 首先处理掉%注释和换行的情况
    bibitem = re.sub(r'%\s*\n\s*', '', bibitem)

    # 处理多余的空格
    bibitem = re.sub(r'\s+', ' ', bibitem)

    # 支持多种bibitem格式的pattern，包括处理复杂的可选参数
    patterns = [
        # 更宽松的bibitem匹配模式，允许方括号内包含任意字符（包括嵌套的花括号）
        r'\\bibitem\s*(?:\[(?:[^\[\]]|\{(?:[^{}]|\{[^{}]*\})*\})*\])?\s*\{([^}]+)\}',
        r'\\bibitemdeclare\s*\{[^}]*\}\s*\{([^}]+)\}',  # bibitemdeclare格式
        r'\\bibitemstart\s*\{([^}]+)\}'  # bibitemstart格式
    ]

    # 尝试每种pattern
    for pattern in patterns:
        match = re.search(pattern, bibitem)
        if match:
            key = match.group(1).strip()
            return key
    key, title = extract_bib_item(bibitem)
    if key and title:
        return key
    if not bibitem.startswith("\\bibitem{} "):
        print("----------Failed to extract bibitem key:", bibitem)
    return None


def extract_bib_item(bib_item):
    # Pattern to match the citation key
    key_pattern = r'@\w+\s*\{([^,]+),'

    # Pattern to match the title
    # Handles both title = {...} and title = "..." formats
    # Also handles multi-line titles
    title_pattern = r'title\s*=\s*[{\"]([^}\"]+)[}\"]'

    # Extract key
    key_match = re.search(key_pattern, bib_item)
    key = key_match.group(1).strip() if key_match else None

    # Extract title
    title_match = re.search(title_pattern, bib_item, re.IGNORECASE)
    title = title_match.group(1).strip() if title_match else None

    return key, title


def collect_bib_info(paper_dir_path):
    # print("paper_dir_path", paper_dir_path)
    global invalid_step_2_count
    if os.path.exists(os.path.join(paper_dir_path, "bib_failed_items.json")):
        with open(os.path.join(paper_dir_path, "bib_failed_items.json"), "r") as fi:
            prev_bib_failed_items = json.load(fi)
    bib_failed_items = []
    step_2_res_path = os.path.join(paper_dir_path, "parsed_parts_25_step_3_result_0109.json")
    step_3_res_path = os.path.join(paper_dir_path, "collect_bib_info_step_4_info_0223.json")
    if not os.path.exists(step_2_res_path):
        # print("step 2 file not found", step_2_res_path)
        invalid_step_2_count += 1
        return []
    # if os.path.exists(step_3_res_path):
    #     # print("step 3 file found, skip it")
    #     with open(os.path.join(paper_dir_path, "bib_failed_items.json"), "r") as fi:
    #         bib_failed_items = json.load(fi)
    #     return bib_failed_items
    with open(step_2_res_path, "r") as fi:
        curr = json.load(fi)
    arxiv_id = curr["arxiv_id"]
    # print("Processing", arxiv_id)
    if curr["intro"] is None or curr["intro"] == "":
        invalid_step_2_count += 1
        return []
    intro = "<$begin_of_introduction$>\n" + curr["intro"] + "\n<$end_of_introduction$>\n"
    related_work = curr["related_work"]
    other_tex = curr["other_tex"]
    if related_work and related_work != "":
        intro = intro + "\n<$begin_of_related_work$>\n" + related_work + "\n<$end_of_related_work$>\n"
    if other_tex and other_tex != "":
        intro = intro + "\n<$begin_of_other_tex$>\n" + other_tex + "\n<$end_of_other_tex$>\n"
    intro, citations = extract_citations(intro)
    cited_keys_in_intro = []
    for k, v in citations.items():
        if v not in cited_keys_in_intro:
            cited_keys_in_intro.append(v)
    bib_info = {}
    for each in curr["bib_items"]:
        citation_key, title = extract_bib_item(each)
        if citation_key not in cited_keys_in_intro:
            # print("citation_key", citation_key)
            # print("cited_keys_in_intro", cited_keys_in_intro)
            continue
        if title is None:
            continue
        bib_info[citation_key] = [each, 0]

    for each in curr["bbl_items"]:
        citation_key = extract_bibitem_key(each)
        if not citation_key:
            continue
        if citation_key not in cited_keys_in_intro:
            # print("citation_key", citation_key)
            # print("cited_keys_in_intro", cited_keys_in_intro)
            continue
        if citation_key in bib_info:
            continue
        bib_info[citation_key] = [each, 1]
        item = [arxiv_id, citation_key, each]
        if item not in prev_bib_failed_items:
            bib_failed_items.append(item)
    if not bib_info:
        return []

    step_3_info = {"intro": intro,
                   "bib_info": bib_info, "citation_map": citations}
    with open(os.path.join(paper_dir_path, "bib_failed_items_0223.json"), "w") as fo:
        fo.write(json.dumps(bib_failed_items, indent=2))
    with open(step_3_res_path, "w") as fo:
        fo.write(json.dumps(step_3_info, indent=2))
    return bib_failed_items


def run_on_darth_server(input_dir, output_failed_item_path):
    global invalid_step_2_count
    failed_items = []
    for sub_dir in os.listdir(input_dir):
        print("Processing", sub_dir)
        if os.path.isdir(os.path.join(input_dir, sub_dir)):
            for paper_dir in tqdm(os.listdir(os.path.join(input_dir, sub_dir))):
                if not paper_dir.startswith(sub_dir):
                    print("skip", paper_dir)
                    continue
                curr = collect_bib_info(os.path.join(input_dir, sub_dir, paper_dir))
                failed_items += curr
    i = 0
    batch_size = 1000000
    print("failed item number: ", len(failed_items))
    print("invalid_step_2_count", invalid_step_2_count)
    while i < len(failed_items):
        if i + batch_size > len(failed_items):
            end = len(failed_items)
        else:
            end = i + batch_size
        curr_batch = failed_items[i: end]
        with open(os.path.join(output_failed_item_path, f"failed_items_batch_{str(i // batch_size)}.json"), "w") as fo:
            fo.write(json.dumps(curr_batch))
        i += batch_size


if __name__ == "__main__":
    input_darth_dir = "/data/yubowang/arxiv_plain_latex_data_1028"
    output_darth_dir = "/data/yubowang/arxiv-llm/qwen_extract_title_data_0223"
    run_on_darth_server(input_darth_dir, output_darth_dir)






