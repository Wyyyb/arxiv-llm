import requests
import time


def search_paper(title, api_key):
    """
    搜索单个论文标题，返回最相近的论文信息

    Args:
        title (str): 论文标题
        api_key (str): Semantic Scholar API密钥

    Returns:
        dict: 论文信息，包含标题、摘要、URL等
    """
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search/match"
    params = {
        "query": title,
        "fields": "title,url,abstract,year,authors,venue"  # 增加了一些可能有用的字段
    }

    headers = {
        "x-api-key": api_key
    }

    try:
        response = requests.get(base_url, params=params, headers=headers)
        # response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        if data and data["data"]:
            paper = data["data"][0]  # 获取最匹配的结果
            result = {
                "标题": paper.get("title"),
                "摘要": paper.get("abstract"),
                "链接": paper.get("url"),
                "匹配度": paper.get("matchScore"),
                "年份": paper.get("year"),
                "发表场所": paper.get("venue"),
                "作者": [author.get("name") for author in paper.get("authors", [])]
            }
            return result
        else:
            return {"错误": "未找到匹配的论文"}

    except requests.exceptions.HTTPError as e:
        return {"错误": f"HTTP错误: {e}"}
    except Exception as e:
        return {"错误": f"发生错误: {e}"}


def main():
    # 替换为你的API密钥
    API_KEY = "xPw99ZZQlprx8uLPejCY8SM6H5HM8eA8jhoXaZ82"

    # 获取用户输入的论文标题
    title = "Augmenting Black-box LLMs with Medical Textbooks for Biomedical Question Answering (Published in Findings of EMNLP 2024)"

    # 搜索论文
    result = search_paper(title, API_KEY)

    # 打印结果
    print("\n搜索结果:")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()


