import json
import os
from tqdm import tqdm


def generate_corpus(file_path):
    output_path = "../data/meta_data_1020.jsonl"
    res = []
    with open(file_path, "r") as fi:
        for line in tqdm(fi.readlines()):
            curr = json.loads(line)
            docs_id = curr["id"]
            title = curr["title"]
            abstract = curr["abstract"].strip()
            res.append({"docs_id": docs_id, "title": title, "abstract": abstract})
    with open(output_path, "w") as fo:
        for each in res:
            fo.write(json.dumps(each))
            fo.write("\n")


if __name__ == '__main__':
    meta_path = "/Users/yubowang/Downloads/arxiv-metadata-oai-snapshot.json"
    generate_corpus(meta_path)

