import json
import os
from tqdm import tqdm
import re


valid_data_num = 0


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


def integrate_single(data_dir_path, semantic_data, metadata, meta_id_map, qwen_data):
    step_3_info_path = os.path.join(data_dir_path, "step_3_info.json")
    step_5_info_path = os.path.join(data_dir_path, "step_5_info.json")
    if not os.path.exists(step_3_info_path):
        return None, semantic_data
    with open(step_3_info_path, "r") as fi:
        step_3_info = json.load(fi)
    if "bib_info" not in step_3_info or not step_3_info["bib_info"]:
        return None, semantic_data
    if "citation_map" not in step_3_info or not step_3_info["citation_map"]:
        return None, semantic_data
    if "full_intro" not in step_3_info or not step_3_info["full_intro"]:
        return None, semantic_data
    arxiv_id = data_dir_path.split("/")[-1]
    title = meta_id_map[arxiv_id]["title"]
    abstract = meta_id_map[arxiv_id]["abstract"]
    bib_info = {}
    valid_cite_count = 0
    for cite_token, citation_key in step_3_info["citation_map"]:
        ori_bib_item = step_3_info["bib_info"][citation_key]
        if citation_key in step_3_info["bib_info"] and step_3_info["bib_info"][citation_key][1] == 0:
            bibitem = step_3_info["bib_info"][citation_key][0]
            key, cite_title = extract_bib_item(bibitem)
        else:
            cite_title = find_title_from_qwen(qwen_data, arxiv_id, citation_key)
        cite_abstract = find_abs_from_metadata(metadata, cite_title)
        if not cite_title:
            message = "title extraction failed: " + str(arxiv_id) + "-" + str(citation_key)
        elif not cite_abstract:
            message = "paper not in arxiv: " + str(cite_title)
            if cite_title not in semantic_data:
                semantic_data[cite_title] = None
            elif cite_title in semantic_data and semantic_data[cite_title] is not None:
                cite_abstract = semantic_data[cite_title]
                message = "success, find paper in semantic data: " + str(cite_title)
                valid_cite_count += 1
        else:
            message = "success"
            valid_cite_count += 1
        curr = {"citation_key": citation_key, "title": cite_title, "abstract": cite_abstract,
                "message": message, "ori_bib_text": ori_bib_item}
        bib_info[cite_token] = curr
    step_5_data = {"arxiv_id": arxiv_id, "title": title, "abstract": abstract,
                   "full_intro": step_3_info["full_intro"], "bib_info": bib_info}
    global valid_data_num
    if valid_cite_count >= 4:
        valid_data_num += 1
    with open(step_5_info_path, "w") as fo:
        fo.write(json.dumps(step_5_data))
    return step_5_data, semantic_data


def find_abs_from_metadata(metadata, title):
    title = clean_title(title)
    if title is None or title not in metadata:
        return None
    return metadata[title]


def find_title_from_qwen(qwen_data, arxiv_id, citation_key):
    if arxiv_id not in qwen_data:
        print("arxiv_id not in qwen_data", arxiv_id)
        return None
    if citation_key not in qwen_data[arxiv_id]:
        print("citation_key not in qwen_data[arxiv_id]", citation_key, arxiv_id)
        return None
    return qwen_data[arxiv_id][citation_key]


def clean_title(title):
    replacements = {
        r'\textit{': '',
        r'\textbf{': '',
        r'\emph{': '',
        r'\em{': '',
        r'\it{': '',
        r'\bf{': '',
        r'\sc{': '',
        r'\sf{': '',
        r'\rm{': '',
        r'\tiny{': '',
        r'\scriptsize{': '',
        r'\footnotesize{': '',
        r'\small{': '',
        r'\normalsize{': '',
        r'\large{': '',
        r'\Large{': '',
        r'\LARGE{': '',
        r'\huge{': '',
        r'\Huge{': '',
        r'\uppercase{': '',
        r'\lowercase{': '',
        '}': '',
        r'\\': ' ',
        r'\&': '&',
        r'\%': '%',
        r'\$': '$',
        r'\#': '#',
        r'\_': '_',
        r'\{': '{',
        r'\}': '}',
        r'\~': '~',
        r'\^': '^',
        r'``': '"',
        r"''": '"'
    }
    cleaned = title

    # Replace LaTeX commands
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)

    # Remove extra whitespace
    cleaned = ' '.join(cleaned.split())

    # Remove remaining special characters but keep basic punctuation
    cleaned = re.sub(r'[^\w\s\-.,;:!?&()\[\]"\'/$]', '', cleaned)
    return cleaned.strip().lower()


def load_metadata(metadata_path):
    metadata = {}
    meta_id_map = {}
    with open(metadata_path, 'r') as fi:
        for line in fi.readlines():
            curr = json.loads(line)
            if "abstract" not in curr or "title" not in curr:
                print("Error loading metadata, no abstract found:\n", curr)
            abstract = curr.get('abstract').repalce("<|reference_start|>", "").replace("<|reference_end|>", "")
            cleaned_title = clean_title(curr["title"])
            if cleaned_title not in metadata:
                metadata[cleaned_title] = abstract
            if curr["arxiv_id"] not in meta_id_map:
                meta_id_map[curr["arxiv_id"]] = curr
    return metadata, meta_id_map


def load_qwen_data(qwen_data_path):
    qwen_data = {}
    print("loading qwen data")
    for file in tqdm(os.listdir(qwen_data_path)):
        if not file.endswith(".json"):
            continue
        with open(os.path.join(qwen_data_path, file), "r") as fi:
            curr_batch = json.load(fi)
            for each in curr_batch:
                arxiv_id = each[0]
                citation_key = each[1]
                title = each[3]
                if arxiv_id not in qwen_data:
                    qwen_data[arxiv_id] = {}
                if citation_key not in qwen_data[arxiv_id]:
                    qwen_data[arxiv_id][citation_key] = title
    print("load qwen data number: ", len(qwen_data))
    print("example: ", qwen_data[1234])
    return qwen_data


def step_5_integrate(input_dir, output_path, metadata_path, qwen_data_path, semantic_data_path):
    semantic_data = {}
    if os.path.exists(semantic_data_path):
        with open(semantic_data_path, "r") as fi:
            semantic_data = json.load(fi)
    step_5_full_data = []
    metadata, meta_id_map = load_metadata(metadata_path)
    qwen_data = load_qwen_data(qwen_data_path)
    global valid_data_num
    for sub_dir in os.listdir(input_dir):
        print("Processing", sub_dir)
        if os.path.isdir(os.path.join(input_dir, sub_dir)):
            for paper_dir in tqdm(os.listdir(os.path.join(input_dir, sub_dir))):
                if not paper_dir.startswith(sub_dir):
                    print("skip", paper_dir)
                    continue
                paper_dir_path = os.path.join(input_dir, sub_dir, paper_dir)
                curr_step_5_info, semantic_data = integrate_single(paper_dir_path, semantic_data,
                                                                   metadata, meta_id_map, qwen_data)
                step_5_full_data.append(curr_step_5_info)
        print("total valid number: ", valid_data_num)
    print("total semantic data number: ", len(semantic_data))
    with open(semantic_data_path, "w") as fo:
        fo.write(json.dumps(semantic_data))
    with open(output_path, "w") as fo:
        for each in step_5_full_data:
            fo.write(json.dumps(each))
            fo.write("\n")


def run_on_darth():
    os.makedirs("../local_1031/", exist_ok=True)
    input_dir = "/data/yubowang/arxiv_plain_latex_data_1028"
    step_5_output_path = "../local_1031/step_5_integration_1031.jsonl"
    metadata_path = "../corpus_data/meta_data_1022.jsonl"
    qwen_data_path = "../local_1031/qwen_extract_title_data_1031"
    semantic_data_path = "../local_1031/semantic_data_1031.json"
    step_5_integrate(input_dir, step_5_output_path, metadata_path, qwen_data_path, semantic_data_path)


if __name__ == '__main__':
    run_on_darth()

