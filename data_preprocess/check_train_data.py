import json
import os


def check(input_dir):
    for file in os.listdir(input_dir):
        wrong_num = 0
        if not file.endswith("jsonl"):
            continue
        file_path = os.path.join(input_dir, file)
        data = []
        with open(file_path, "r") as fi:
            for line in fi.readlines():
                data.append(json.loads(line))
        for each in data:
            if len(each["targets"]) != 4 or len(each["targets_idx"]) != 4:
                print(file, each["arxiv_id"])
            paper = each["paper"]
            if paper.count("<|cite_end|>") < 4 or paper.count("<|cite_start|>") < 4:
                print("cite num", file, each["arxiv_id"], paper.count("<|cite_start|>"), paper.count("<|cite_end|>"))
                wrong_num += 1
            for each_idx in each["targets_idx"]:
                if each_idx >= paper.count("<|cite_end|>") or paper.count("<|cite_end|>") != paper.count("<|cite_start|>"):
                    print(each["arxiv_id"], each_idx, paper.count("<|cite_end|>"))
        # print("cite num wrong", file, wrong_num)


# check("../local/arxiv_base/")
check("../data/")














