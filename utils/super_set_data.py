import json
from tqdm import tqdm


with open("../scholar_copilot_data_1215/bibtex_info_1202.jsonl", "r") as fi:
    bibtex_info = {}
    for line in tqdm(fi.readlines()):
        curr = json.loads(line)
        paper_id = curr['id']
        bibtex_info[paper_id] = curr

with open("../scholar_copilot_data_1215/corpus_data_arxiv_1129.jsonl", "r") as fi:
    corpus_data = []
    for line in tqdm(fi.readlines()):
        curr = json.loads(line)
        paper_id = curr["paper_id"]
        if paper_id not in bibtex_info:
            print("Could not find paper", paper_id)
            continue
        bibtex = bibtex_info[paper_id]["bibtex"]
        # title = bibtex_info[paper_id]["title"]
        citation_key = bibtex_info[paper_id]["citation_key"]
        curr["bibtex"] = bibtex
        curr["citation_key"] = citation_key
        corpus_data.append(curr)
    print("len(corpus_data)", len(corpus_data))

with open("../scholar_copilot_data_1215/corpus_data_arxiv_1215.jsonl", "w") as fo:
    for each in tqdm(corpus_data):
        fo.write(json.dumps(each) + "\n")


