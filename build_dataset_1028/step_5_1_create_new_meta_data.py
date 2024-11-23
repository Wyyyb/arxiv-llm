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
            res.append(curr)
    print("arxiv meta number:", len(res))
    return res


def load_ss_res_data(ss_res_dir="../local_darth_1014/"):
    res = []
    exact_count = 0
    api_count = 0
    low_score_count = 0
    for file in os.listdir(ss_res_dir):
        if not file.endswith("output.json"):
            continue
        file_path = os.path.join(ss_res_dir, file)
        with open(file_path, "r") as fi:
            data = json.load(fi)
            for k, v in tqdm(data.items()):
                curr = {}
                if v is None:
                    continue
                if "abstract" not in v or v["abstract"] is None:
                    continue
                abstract = v["abstract"].strip()
                if "matchScore" in v and v["matchScore"] < 30:
                    print("***********matchScore less than 30:\n", k, v)
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
    return res


def main():
    ori_metadata = load_ori_metadata()
    ss_metadata = load_ss_res_data()
    metadata = ori_metadata + ss_metadata
    with open("../corpus_data/metadata_1123.jsonl", "w") as fo:
        for each in metadata:
            fo.write(json.dumps(each) + "\n")


main()




