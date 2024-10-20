import json
import os
from tqdm import tqdm


def check_cs(categories):
    cat = categories.lower()
    cat = cat.replace("physics", "")
    if "cs." in cat:
        return True
    else:
        return False


def generate_corpus(file_path):
    output_path = "../corpus_data/meta_data_1020.jsonl"
    res = []
    with open(file_path, "r") as fi:
        for line in tqdm(fi.readlines()):
            curr = json.loads(line)
            if not check_cs(curr["categories"]):
                continue
            docs_id = curr["id"]
            title = curr["title"]
            abstract = "<|reference_start|>" + curr["abstract"].strip() + "<|reference_end|>"
            res.append({"docs_id": docs_id, "title": title, "abstract": abstract})
    print("length of res", len(res))
    with open(output_path, "w") as fo:
        for each in res:
            fo.write(json.dumps(each))
            fo.write("\n")
    with open(output_path.replace(".jsonl", "_sample.jsonl"), "w") as fo:
        for each in res[:100]:
            fo.write(json.dumps(each))
            fo.write("\n")


if __name__ == '__main__':
    meta_path = "/Users/yubowang/Downloads/arxiv-metadata-oai-snapshot.json"
    generate_corpus(meta_path)

