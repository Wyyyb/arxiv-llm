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
    return papers


def load_data(output_path, paper_dir_path, abs_dir_path, tldr_dir_path):
    papers = load_papers(paper_dir_path)
    abstracts = load_abs(abs_dir_path)
    tldrs = load_tldr(tldr_dir_path)
    ss_data = {}
    for k, v in tqdm(papers.items()):
        if v in ss_data:
            continue
        if k not in abstracts:
            abstract = None
        else:
            abstract = abstracts[k]
        if k not in tldrs:
            tldr = None
        else:
            tldr = tldrs[k]
        ss_data[v] = {"id": k, "abstract": abstract, "tldr": tldr}
    with open(output_path, "w") as fo:
        fo.write(json.dumps(ss_data))


def collect():
    output_path = "/data/yubowang/ss_offline_data/ss_offline_data_1109.json"
    paper_dir_path = "/data/yubowang/offline_ss_papers/"
    abs_dir_path = "/data/yubowang/offline_ss_abstract/"
    tldr_dir_path = "/data/yubowang/offline_ss_tldrs/"
    load_data(output_path, paper_dir_path, abs_dir_path, tldr_dir_path)


collect()

