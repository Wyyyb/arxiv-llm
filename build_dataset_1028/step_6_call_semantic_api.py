import requests
import time
import json
import os
from tqdm import tqdm


def call_semantic_api(input_path, output_path):
    api_key = "xPw99ZZQlprx8uLPejCY8SM6H5HM8eA8jhoXaZ82"
    to_search_data = load_semantic_data(input_path)
    res_data = {}
    success_count = 0
    fail_count = 0
    low_score_count = 0
    total_count = 0
    if os.path.exists(output_path):
        with open(output_path, "r") as fi:
            res_data = json.load(fi)
    for each in tqdm(to_search_data):
        if each not in res_data or "message" in res_data[each]:
            curr_res = get_paper_info(each, api_key)
        else:
            curr_res = res_data[each]
        if curr_res is None:
            fail_count += 1
        elif "matchScore" in curr_res and float(curr_res["matchScore"]) < 30:
            low_score_count += 1
        elif "matchScore" in curr_res and float(curr_res["matchScore"]) >= 30:
            success_count += 1
        else:
            print("unsupported curr res:", curr_res)
        total_count += 1
        res_data[each] = curr_res
        if len(res_data) % 10 == 0:
            print(f"statistic: \ntotal_count: {total_count}\nsuccess_count: "
                  f"{success_count}\nlow_score_count: {low_score_count}\nfailed_count: {fail_count}")
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
        if cost < 2:
            print("will sleeping...", 2 - cost)
            time.sleep(2-cost)
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
            time.sleep(2)
            return None
        else:
            print(f"HTTP error occurred: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    time.sleep(2)
    return {"message": "client 429 error, try later"}


if __name__ == "__main__":
    call_semantic_api("../local_1031/semantic_data_1031.json", "../local_1031/semantic_data_1101.json")



