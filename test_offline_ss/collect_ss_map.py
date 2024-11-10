import json
import os
from tqdm import tqdm


def load_papers(paper_dir_path):
    papers = {}
    for file in tqdm(os.listdir(paper_dir_path)):
        if not file.endswith('.json'):
            continue
        file_path = os.path.join(paper_dir_path, file)
        with open(file_path, "r") as fi:
            for line in fi.readlines():
                curr = json.loads(line)
                title = curr["title"]
                corpusid = curr["corpusid"]
                if corpusid not in papers:
                    papers[corpusid] = title
    print("number of papers data", len(papers))
    return papers


def load_abs(abs_dir_path):
    papers = {}
    for file in tqdm(os.listdir(abs_dir_path)):
        if not file.endswith('.json'):
            continue
        file_path = os.path.join(abs_dir_path, file)
        with open(file_path, "r") as fi:
            for line in fi.readlines():
                curr = json.loads(line)
                abstract = curr["abstract"]
                corpusid = curr["corpusid"]
                if corpusid not in papers:
                    papers[corpusid] = abstract
    print("number of abs data", len(papers))
    return papers


def load_tldr(tldr_dir_path):
    papers = {}
    for file in tqdm(os.listdir(tldr_dir_path)):
        if not file.endswith('.json'):
            continue
        file_path = os.path.join(tldr_dir_path, file)
        with open(file_path, "r") as fi:
            for line in fi.readlines():
                curr = json.loads(line)
                tldr = curr["text"]
                corpusid = curr["corpusid"]
                if corpusid not in papers:
                    papers[corpusid] = tldr
    print("number of tldr data", len(papers))
    return papers


def load_data(output_path, paper_dir_path, abs_dir_path):
    papers = load_papers(paper_dir_path)
    abstracts = load_abs(abs_dir_path)
    ss_data = []
    for k, v in tqdm(papers.items()):
        if k not in abstracts:
            # abstract = None
            continue
        else:
            abstract = abstracts[k]
        ss_data.append([k, v, abstract])
    print("number of ss_data", len(ss_data))
    with open(output_path, "w") as fo:
        for each in ss_data:
            fo.write(json.dumps(each) + "\n")


def collect():
    os.makedirs("/gpfs/public/research/xy/yubowang/ss_offline_data", exist_ok=True)
    output_path = "/gpfs/public/research/xy/yubowang/ss_offline_data/ss_offline_data_1109.jsonl"
    paper_dir_path = "/gpfs/public/research/xy/yubowang/offline_ss_paper/"
    abs_dir_path = "/gpfs/public/research/xy/yubowang/offline_ss_abstract/"
    # tldr_dir_path = "/data/yubowang/offline_ss_tldrs/"
    load_data(output_path, paper_dir_path, abs_dir_path)


collect()

