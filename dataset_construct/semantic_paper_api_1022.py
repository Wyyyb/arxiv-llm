import requests
import time


def get_papers_info_batch(titles, api_key):
    base_url = "https://api.semanticscholar.org/graph/v1/paper/batch"
    batch_size = 10
    results = {}

    for i in range(0, len(titles), batch_size):
        batch_titles = titles[i:i + batch_size]

        params = {
            "query": batch_titles,  # 直接传入标题列表
            "fields": "paperId,title,url,abstract"
        }

        headers = {
            "x-api-key": api_key
        }

        start = time.time()
        try:
            response = requests.post(base_url, json=params, headers=headers)
            response.raise_for_status()
            cost = float(time.time() - start)
            print("requesting semantic scholar api cost:", cost)

            # Rate limiting - batch endpoint allows 1 RPS
            if cost < 1:
                sleep_time = 1 - cost
                print("will sleeping...", sleep_time)
                time.sleep(sleep_time)

            data = response.json()

            # 将结果与原始标题对应
            for idx, paper in enumerate(data):
                if paper:
                    results[batch_titles[idx]] = {
                        "paperId": paper.get("paperId", None),
                        "title": paper.get("title", None),
                        "abstract": paper.get("abstract", None),
                        "url": paper.get("url", None)
                    }
                else:
                    results[batch_titles[idx]] = None

        except requests.exceptions.HTTPError as e:
            print(f"HTTP error occurred: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(1)

    return results


# Example usage:
# titles = ["title1", "title2", "title3", ..., "title10"]
# results = get_papers_info_batch(titles, api_key)
# 返回结果形式: {"title1": {...}, "title2": {...}, ...}

def get_paper_info(title, api_key):
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search/match"
    params = {
        "query": title,
        "fields": "title,url,abstract"
    }

    headers = {
        "x-api-key": api_key
    }
    start = time.time()
    try:
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
        else:
            print(f"HTTP error occurred: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    time.sleep(2)
    return None


def get_papers_info_bulk(titles, api_key):
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"

    # Process titles in batches of 10
    batch_size = 10
    results = []

    for i in range(0, len(titles), batch_size):
        batch_titles = titles[i:i + batch_size]

        params = {
            "query": " OR ".join([f'"{title}"' for title in batch_titles]),
            "fields": "paperId,title,url,abstract,matchScore",
            "limit": batch_size
        }

        headers = {
            "x-api-key": api_key
        }

        start = time.time()
        try:
            response = requests.get(base_url, params=params, headers=headers)
            response.raise_for_status()
            cost = float(time.time() - start)
            print("requesting semantic scholar api cost:", cost)

            # Rate limiting - bulk endpoint allows 5 RPS
            if cost < 0.2:  # 1/5 second
                sleep_time = 0.2 - cost
                print("will sleeping...", sleep_time)
                time.sleep(sleep_time)

            data = response.json()

            if data and "data" in data:
                for paper in data["data"]:
                    paper_info = {
                        "paperId": paper.get("paperId", None),
                        "title": paper.get("title", None),
                        "abstract": paper.get("abstract", None),
                        "url": paper.get("url", None),
                        "matchScore": paper.get("matchScore", None)
                    }
                    results.append(paper_info)

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print("Title match not found")
            else:
                print(f"HTTP error occurred: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(0.2)

    return results


# Example usage:
# titles = ["title1", "title2", "title3", ..., "title10"]
# results = get_papers_info_bulk(titles, api_key)

API_KEY = "xPw99ZZQlprx8uLPejCY8SM6H5HM8eA8jhoXaZ82"
# 使用示例
paper_title_1 = "{DjiNN} and tonic: {DNN} as a service and its implications for future warehouse scale computers. "
paper_title_2 = "MMLU-Pro"
paper_title_3 = "MMMU"
paper_title_4 = "Augmenting Blackbox Large language model with Medical Textbooks"
paper_title_5 = "MMMU-Pro"
titles = [paper_title_1, paper_title_2, paper_title_3, paper_title_4, paper_title_5]

res_list = get_papers_info_bulk(titles, API_KEY)

for info in res_list:
    print(f"Paper ID: {info['paperId']}")
    print(f"Title: {info['title']}")
    print(f"Abstract: {info['abstract']}")
    print(f"MatchScore: {info['matchScore']}")


# info = get_paper_info(paper_title, API_KEY)
#
# if info:
#     print(f"Paper ID: {info['paperId']}")
#     print(f"Title: {info['title']}")
#     print(f"Abstract: {info['abstract']}")
# else:
#     print("Failed to retrieve paper information")

