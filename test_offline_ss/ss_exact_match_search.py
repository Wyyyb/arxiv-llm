import json
import os
from tqdm import tqdm
import copy
import re


def normalize_title(text: str) -> str:
    """
    将文本转小写，去除特殊字符和多余空白
    """
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def load_query(dir_path):
    query_data = []
    file_path_list = []
    for file in os.listdir(dir_path):
        if not file.endswith("json"):
            continue
        file_path = os.path.join(dir_path, file)
        with open(file_path, "r") as fi:
            curr = json.load(fi)
            query_data.append(curr)
            file_path_list.append(file_path)
    return query_data, file_path_list


def load_corpus(corpus_path):
    corpus_map = {}
    corpus_data = {}
    print("Loading corpus ...")
    with open(corpus_path, "r") as fi:
        # 使用 tqdm 包装文件对象
        for line in tqdm(fi):
            curr = json.loads(line)
            corpus_id, title, abstract = curr
            key = normalize_title(title)
            corpus_data[key] = str(corpus_id)
            corpus_map[str(corpus_id)] = abstract
    return corpus_data, corpus_map


def exact_match_search(corpus_data, query):
    query = normalize_title(query)
    if query not in corpus_data:
        return None
    return corpus_data[query]


def main():
    corpus_path = "/gpfs/public/research/xy/yubowang/ss_offline_data/ss_offline_data_1109.jsonl"
    query_dir_path = "/gpfs/public/research/xy/yubowang/arxiv-llm/local_1111/ss_data_query_1111_exact/"
    query_data_list, query_data_path_list = load_query(query_dir_path)
    corpus_data, corpus_map = load_corpus(corpus_path)

    for i, query_data in enumerate(query_data_list):
        # if not query_data_path_list[i].endswith("0.json") and not query_data_path_list[i].endswith("1.json"):
        #     continue
        print("querying", query_data_path_list[i])
        success_count = 0
        fail_count = 0
        res = copy.deepcopy(query_data)
        for k, v in tqdm(query_data.items()):
            if v is not None and "abstract" in v:
                res[k] = v
                success_count += 1
                continue
            query = k
            result = exact_match_search(corpus_data, query)
            if result:
                paper_id = result
                if paper_id not in corpus_map:
                    print("error in corpus map, not found id:", paper_id)
                    res[k] = v
                    continue
                res[k] = {"paper_id": paper_id, "abstract": corpus_map[paper_id],
                          "source": "exact match from offline ss"}
                success_count += 1
                if success_count % 10000 == 0:
                    print("success count", success_count)
                    print("fail count", fail_count)
                    print("query: ", query)
                    print("result: ", res[k])
                    with open(query_data_path_list[i], "w") as fo:
                        fo.write(json.dumps(res))
            else:
                res[k] = None
                fail_count += 1
                continue
        print("success count", success_count)
        print("fail count", fail_count)
        with open(query_data_path_list[i], "w") as fo:
            fo.write(json.dumps(res))


main()


