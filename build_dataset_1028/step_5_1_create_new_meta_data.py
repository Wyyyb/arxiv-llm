import json
import os
from tqdm import tqdm


def load_ori_metadata(file_path="../corpus_data/meta_data_1022.jsonl"):
    res = []
    with open(file_path, "r") as fi:
        for line in fi:
            curr = json.loads(line)
            abstract = curr["abstract"]
            title = curr["title"]
            abstract = abstract.replace("<|reference_start|>", "").replace("<|reference_end|>", "")
            curr["abstract"] = f"<|reference_start|>{title}: {abstract}<|reference_end|>"
            curr["source"] = "arxiv"
            curr["meta_title"] = title
            curr.pop("docs_id")
            curr["paper_id"] = curr["arxiv_id"]
            res.append(curr)
    print("arxiv meta number:", len(res))
    return res


def load_ss_res_data(ss_res_dir="../local_darth_1014/"):
    res = []
    exact_count = 0
    api_count = 0
    low_score_count = 0
    v_none_count = 0
    title_not_fount_count = 0
    for file in os.listdir(ss_res_dir):
        if not file.endswith("output.json"):
            continue
        file_path = os.path.join(ss_res_dir, file)
        with open(file_path, "r") as fi:
            data = json.load(fi)
            for k, v in tqdm(data.items()):
                curr = {}
                if v is None:
                    v_none_count += 1
                    continue
                if "abstract" not in v:
                    if v["message"] != 'Title match not found':
                        print("invalid message", v)
                    title_not_fount_count += 1
                    continue
                elif v["abstract"] is None:
                    abstract = ""
                else:
                    abstract = v["abstract"].strip()
                if "matchScore" in v and v["matchScore"] < 30:
                    # print("***********matchScore less than 30:\n", k, v)
                    low_score_count += 1
                    continue
                if "source" in v and v["source"] == "exact match from offline ss":
                    exact_count += 1
                    curr["source"] = v["source"]
                else:
                    api_count += 1
                    curr["source"] = "ss api result"
                title = k
                if "paperId" in v:
                    paper_id = v["paperId"]
                elif "paper_id" in v:
                    paper_id = v["paper_id"]
                else:
                    print("***********paper id not found", v)
                if "title" not in v:
                    meta_title = title
                else:
                    meta_title = v["title"]
                curr["meta_title"] = meta_title
                curr["title"] = title
                curr["paper_id"] = paper_id
                curr["abstract"] = f"<|reference_start|>{meta_title}: {abstract}<|reference_end|>"
                res.append(curr)
    print("api_count", api_count)
    print("exact_count", exact_count)
    print("low score count", low_score_count)
    print("title_not_fount_count", title_not_fount_count)
    print("v_none_count", v_none_count)
    return res


def identify(metadata):
    res = []
    corpus_id_count = 0
    exist_paper_id = []
    print("identifying")
    for each in tqdm(metadata):
        paper_id = each["paper_id"]
        if paper_id in exist_paper_id:
            continue
        corpus_id_count += 1
        if each["source"] == "arxiv":
            corpus_id = f"arxiv-{str(corpus_id_count)}"
        else:
            corpus_id = f"ss-{str(corpus_id_count)}"
        curr = {"corpus_id": corpus_id, "paper_id": paper_id, "title": each["meta_title"],
                "abstract": each["abstract"], "source": each["source"]}
        res.append(curr)
        exist_paper_id.append(paper_id)
    return res


def main():
    ori_metadata = load_ori_metadata()
    ss_metadata = load_ss_res_data()
    metadata = ori_metadata + ss_metadata
    corpus_data = identify(metadata)
    print("1123 version metadata number", len(metadata))
    print("1124 version corpus data number", len(corpus_data))
    with open("../corpus_data/metadata_1123.jsonl", "w") as fo:
        for each in metadata:
            fo.write(json.dumps(each) + "\n")
    with open("../corpus_data/corpus_data_1124.jsonl", "w") as fo:
        for each in corpus_data:
            fo.write(json.dumps(each) + "\n")


main()




