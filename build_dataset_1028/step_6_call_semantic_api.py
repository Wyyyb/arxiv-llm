from dataset_construct.semantic_paper_api_1022 import *
import json
import os
from tqdm import tqdm


def call_semantic_api(input_path, output_path):
    api_key = "xPw99ZZQlprx8uLPejCY8SM6H5HM8eA8jhoXaZ82"
    to_search_data = load_semantic_data(input_path)
    res_data = {}
    if os.path.exists(output_path):
        with open(output_path, "r") as fi:
            res_data = json.load(fi)
    for each in tqdm(to_search_data):
        if each in res_data:
            print("already found, skip it")
            continue
        curr_res = get_paper_info(each, api_key)
        res_data[each] = curr_res
    if len(res_data) % 10 == 0:
        with open(output_path, "w") as fo:
            fo.write(json.dumps(res_data))


def load_semantic_data(data_path):
    to_search_data = []
    with open(data_path, "r") as fi:
        data = json.load(fi)
    for title, abstract in data.items():
        to_search_data.append(title.strip())
    return to_search_data


if __name__ == "__main__":
    call_semantic_api("../local_1031/semantic_data_1031.json", "../local_1031/semantic_data_1101.json")



