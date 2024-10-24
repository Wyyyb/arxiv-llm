import requests
import time


def get_paper_info(title):
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search/match"
    params = {
        "query": title,
        "fields": "title,url,abstract"
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # 如果请求失败，这将引发一个异常

        data = response.json()

        # 检查是否有匹配的论文
        if data:
            return {
                "paperId": data.get("paperId"),
                "title": data.get("title"),
                "abstract": data.get("abstract"),
                "url": data.get("url"),
                "matchScore": data.get("matchScore")
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


# 使用示例
paper_title = "Human pose estimation with iterative error feedback."
paper_info = get_paper_info(paper_title)

if paper_info:
    print(f"Paper ID: {paper_info['paperId']}")
    print(f"Title: {paper_info['title']}")
    print(f"Abstract: {paper_info['abstract']}")
else:
    print("Failed to retrieve paper information")

