import json
import os


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
            res.append(curr)
    print("arxiv meta number:", len(res))
    return res


def load_ss_res_data(ss_res_dir="../local_darth_1014/"):
    res = []
    exact_count = 0
    api_count = 0
    for file in os.listdir(ss_res_dir):
        if not file.endswith("output.json"):
            continue
        file_path = os.path.join(ss_res_dir, file)
        with open(file_path, "r") as fi:
            data = json.load(fi)
            for k, v in data.items():
                curr = {}
                if v is None:
                    continue
                if "abstract" not in v or v["abstract"] is None:
                    continue
                abstract = v["abstract"].strip()
                if "matchScore" in v and v["matchScore"] < 50:
                    print("***********matchScore less than 50:\n", k, v)
                    continue
                if "source" in v and v["source"] == "exact match from offline ss":
                    exact_count += 1
                    curr["source"] = v["source"]
                else:
                    api_count += 1
                    curr["source"] = "ss api result"
                title = k
                if "paperId" not in v:
                    print("***********paper id not found", v)
                paper_id = v["paperId"]
                curr["title"] = title
                curr["paper_id"] = paper_id
                curr["abstract"] = f"<|reference_start|>{title}: {abstract}<|reference_end|>"
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




