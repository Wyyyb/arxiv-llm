import requests
import time
import json
import os
from tqdm import tqdm


def call_semantic_api(input_path, output_path, api_key):
    to_search_data = load_semantic_data(input_path)
    success_count = 0
    fail_count = 0
    low_score_count = 0
    api_total_count = 0
    if os.path.exists(output_path):
        with open(output_path, "r") as fi:
            res_data = json.load(fi)
    else:
        res_data = to_search_data
    for k, v in tqdm(res_data.items()):
        if isinstance(v, dict) and "abstract" in v:
            continue
        if isinstance(v, dict) and "message" in v and v["message"] == "Title match not found":
            continue
        curr_res = get_paper_info(k, api_key)
        if "matchScore" in curr_res and float(curr_res["matchScore"]) < 30:
            low_score_count += 1
        elif "matchScore" in curr_res and float(curr_res["matchScore"]) >= 30:
            success_count += 1
        elif "message" in curr_res:
            fail_count += 1
        res_data[k] = curr_res
        api_total_count += 1

        if api_total_count % 1000 == 0:
            print(f"statistic: \napi_total_count: {api_total_count}\nsuccess_count: "
                  f"{success_count}\nlow_score_count: {low_score_count}\nfailed_count: {fail_count}")
            with open(output_path, "w") as fo:
                fo.write(json.dumps(res_data))


def load_semantic_data(data_path):
    to_search_data = []
    with open(data_path, "r") as fi:
        data = json.load(fi)
    # for title, abstract in data.items():
    #     to_search_data.append(title.strip())
    # return to_search_data
    return data


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
            # time.sleep(2-cost)
        time.sleep(1.2)
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
            time.sleep(1.2)
            return {"message": "Title match not found"}
        else:
            print(f"HTTP error occurred: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    time.sleep(1.2)
    return None


if __name__ == "__main__":
    API_KEY = "vgPRBYMNV9asiaTwN5o5b7mH2f0HnOVM9yN0MWp6"
    index = 2
    input_file_path = f"/data/yubowang/arxiv-llm/local_darth_1014/ss_result_data_1114_{str(index)}.json"
    output_file_path = input_file_path.replace(".json", "_output.json")
    call_semantic_api(input_file_path, output_file_path, API_KEY)



