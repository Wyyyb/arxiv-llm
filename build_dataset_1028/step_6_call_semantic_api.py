import requests
import time
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


def get_paper_info(title, api_key):
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search/match"
    params = {
        "query": title,
        "fields": "title,url,abstract"
    }

    headers = {
        "x-api-key": api_key
    }

    try:
        start = time.time()
        response = requests.get(base_url, params=params, headers=headers)
        response.raise_for_status()
        cost = float(time.time() - start)
        print("requesting semantic scholar api cost:", cost)
        if cost < 1:
            print("will sleeping...", 1 - cost)
            time.sleep(1-cost)
        data = response.json()

        if data:
            data = data["data"][0]
            return {
                "paperId": data.get("paperId", None),
                "title": data.get("title", None),
                "abstract": data.get("abstract", None),
                "url": data.get("url", None),
                "matchScore": data.get("matchScore", None)
            }
        else:
            return None

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print("Title match not found")
        else:
            print(f"HTTP error occurred: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return None


if __name__ == "__main__":
    call_semantic_api("../local_1031/semantic_data_1031.json", "../local_1031/semantic_data_1101.json")



