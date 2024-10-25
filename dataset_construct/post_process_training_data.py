from extract_by_patterns import *
from semantic_paper_api_1022 import *
import os
import json


def load_semantic_meta_data(semantic_meta_data_path):
    res = []
    title_map = {}
    if os.path.exists(semantic_meta_data_path):
        with open(semantic_meta_data_path, "r") as fi:
            for line in fi.readlines():
                curr = json.loads(line)
                if curr["bias"] not in title_map:
                    title_map[curr["bias"]] = curr
                    res.append(curr)
    return res, title_map


def update_semantic_meta_data(semantic_meta_data, semantic_meta_data_path):
    with open(semantic_meta_data_path, "w") as fo:
        for each in semantic_meta_data:
            fo.write(json.dumps(each))
            fo.write("\n")


def post_process(curr_paper, meta_data, title_map, semantic_scholar_cache_path, semantic_scholar_cache):
    semantic_meta_data_path = "../corpus_data/semantic_meta_data_1022.jsonl"
    arxiv_id = curr_paper["arxiv_id"]
    if "intro" not in curr_paper and "related_work" not in curr_paper:
        curr_paper["satisfied_data"] = False
        return curr_paper
    intro_parts = [
        generate_title_latex(curr_paper.get("title", None)),
        generate_intro_latex(curr_paper.get("intro", None)),
        generate_related_work_latex(curr_paper.get("related_work", None))]
    intro = "\n".join(intro_parts)

    citation_list = extract_citation_keys(intro)
    index = 0
    targets = {}
    for cite_key, cite_value, is_multi in citation_list:
        # print("cite_value", cite_value)
        if "bib" not in curr_paper or cite_key not in curr_paper["bib"]:
            intro = intro.replace(cite_value, "")
            continue
        bib_item = curr_paper["bib"][cite_key]
        if "title" in bib_item and bib_item["title"] != "Unknown":
            ori_title = bib_item["title"]
            title = bib_item["title"]
            title = clean_title(title)
            title = title.lower()
        else:
            intro = intro.replace(cite_value, "")
            continue
        if title in title_map:
            cite_arxiv_id = title_map[title][0]
            abstract = meta_data[cite_arxiv_id]["abstract"]
        else:
            semantic_scholar_res = request_semantic(ori_title, semantic_scholar_cache_path,
                                                    semantic_scholar_cache)
            if not semantic_scholar_res:
                intro = intro.replace(cite_value, "")
                continue
            else:
                print("arxiv id", arxiv_id)
                print("semantic scholar newly recall title:", title)
                intro = intro.replace(cite_value, "")
                abstract = semantic_scholar_res["abstract"]

        # 当出现multi_cite的时候，只保留第一个合格的citation
        if cite_value not in intro:
            continue
        if is_multi:
            cite_token = f"<|multi_cite_token${str(index)}$|>"
        else:
            cite_token = f"<|cite_token${str(index)}$|>"

        intro = intro.replace(cite_value, cite_token, 1)
        targets[cite_token] = abstract
        index += 1

    while "  " in intro:
        intro = intro.replace("  ", "")

    paper = intro
    data = {"arxiv_id": arxiv_id, "paper": paper, "targets": targets}
    curr_paper["data"] = data
    curr_paper["satisfied_data"] = index >= 4
    return curr_paper


def request_semantic(title, semantic_scholar_cache_path, semantic_scholar_cache):
    return None
    SC_API_KEY = "xPw99ZZQlprx8uLPejCY8SM6H5HM8eA8jhoXaZ82"
    if title in semantic_scholar_cache:
        paper_info = semantic_scholar_cache[title]
    else:
        paper_info = get_paper_info(title, SC_API_KEY)
        semantic_scholar_cache[title] = paper_info
    if len(semantic_scholar_cache) % 100 == 0:
        with open(semantic_scholar_cache_path, "w") as fo:
            fo.write(json.dumps(semantic_scholar_cache))
    if paper_info is None:
        return None
    if "matchScore" not in paper_info or paper_info["matchScore"] is None:
        return None
    if paper_info["matchScore"] < 30:
        print("score less than 30", title, paper_info["title"])
        return None
    if paper_info["abstract"] is None or len(paper_info["abstract"]) < 50:
        return None
    else:
        abstract = paper_info["abstract"]
        abstract = abstract.replace("\n", " ").strip()
        return {"docs_id": paper_info["paperId"],
                "paperId": paper_info["paperId"],
                "abstract": "<|reference_start|>" + abstract + "<|reference_end|>",
                "title": clean_title((paper_info["title"])),
                "matchScore": paper_info["matchScore"],
                "url": paper_info["url"],
                "bias": title}


def generate_intro_latex(content):
    if not content:
        content = ""
    latex_code = r"\section{Introduction}" + "\n"
    latex_code += content.strip() + "\n"
    return latex_code


def generate_related_work_latex(content):
    if not content:
        content = ""
    latex_code = r"\section{Related Work}" + "\n"
    latex_code += content.strip() + "\n"
    return latex_code


def generate_title_latex(title):
    if not title:
        title = ""
    latex_code = r"\title{" + title.strip() + "}\n"
    latex_code += r"\maketitle" + "\n\n"
    return latex_code




