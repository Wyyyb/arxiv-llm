import json
import os


def load_api_result_data(data_path):
    with open(data_path, "r") as fi:
        text = fi.read()
        text = text.replace(", \"A personal view on systems medicine and the emergence of proacti", "}")
        api_res = json.loads(text)
    print("api_result_data number", len(api_res))
    return api_res


def load_exact_match_res(data_dir_path):
    data = {}
    for file in os.listdir(data_dir_path):
        if not file.endswith(".json"):
            continue
        file_path = os.path.join(data_dir_path, file)
        with open(file_path, "r") as fi:
            curr = json.load(fi)
        for k, v in curr.items():
            k = k.strip()
            data[k] = v
    print("exact_match_res data number", len(data))
    return data


def merge_data(api_res, exact_match_res):
    success_count = 0
    for k, v in exact_match_res.items():
        if k not in api_res:
            # print("key not found in api_res", k)
            continue
        if v is not None and "abstract" in v:
            success_count += 1
            continue
        elif v is not None:
            print("abstract not found in exact match res", v)

        if api_res[k] is not None:
            exact_match_res[k] = api_res[k]
            success_count += 1
    return exact_match_res


def split_dict(data, split_num=5):
    items = list(data.items())
    total = len(data)
    each_num = total // split_num + 1
    res = []
    i = 0
    while i < total:
        end = min(i + each_num, total)
        curr = items[i: end]
        res.append(dict(curr))
    return res


def main():
    api_res_path = "/data/yubowang/arxiv-llm/local_1031/semantic_data_1101.json"
    exact_match_dir_path = "/data/yubowang/data_trans_1030/ss_data_query_1111_exact/"
    output_ss_data_dir_path = "/data/yubowang/arxiv-llm/local_darth_1014/"
    os.makedirs(output_ss_data_dir_path, exist_ok=True)
    api_res = load_api_result_data(api_res_path)
    exact_match_res = load_exact_match_res(exact_match_dir_path)
    res_data = merge_data(api_res, exact_match_res)
    data_list = split_dict(res_data, 5)
    for i in range(5):
        file_path = os.path.join(output_ss_data_dir_path, f"ss_result_data_1114_{i}.json")
        with open(file_path, "w") as fo:
            fo.write(json.dumps(data_list[i]))


main()
