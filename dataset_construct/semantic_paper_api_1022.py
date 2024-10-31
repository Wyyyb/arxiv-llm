import requests
import time


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
    time.sleep(2)
    return None


API_KEY = "xPw99ZZQlprx8uLPejCY8SM6H5HM8eA8jhoXaZ82"
# 使用示例
paper_title = "{DjiNN} and tonic: {DNN} as a service and its implications for future warehouse scale computers. "
info = get_paper_info(paper_title, API_KEY)

if info:
    print(f"Paper ID: {info['paperId']}")
    print(f"Title: {info['title']}")
    print(f"Abstract: {info['abstract']}")
else:
    print("Failed to retrieve paper information")

