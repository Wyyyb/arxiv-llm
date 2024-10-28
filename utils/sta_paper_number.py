import json
import os


def count_paper_number(input_dir_path):
    count = 0
    for sub_dir in os.listdir(input_dir_path):
        curr_path = os.path.join(input_dir_path, sub_dir)
        if not os.path.isdir(curr_path):
            continue
        for each in os.listdir(curr_path):
            if not each.startswith(sub_dir):
                continue
            count += 1
    print(input_dir_path, "has", count, "papers.")


def count_meta_data_paper(input_path):
    res_data = []
    total = 0
    num_2409 = 0
    num_2410 = 0
    with open(input_path, "r") as fi:
        for line in fi.readlines():
            total += 1
            curr = json.loads(line)
            # if curr["arxiv_id"].startswith("2409"):
            #     num_2409 += 1
            #     continue
            if curr["arxiv_id"].startswith("2410"):
                num_2410 += 1
                continue
            res_data.append(curr)
    print(input_path, "has", len(res_data), "papers.")
    print("total papers:", total)
    print("2409 papers:", num_2409)
    print("2410 papers:", num_2410)
    # with open("../corpus_data/arxiv_meta_data_1028.jsonl", "w") as fo:
    #     for each in res_data:
    #         fo.write(json.dumps(each) + "\n")


# count_paper_number("/data/yubowang/arxiv-latex-filtered_1014")
count_paper_number("/data/yubowang/arxiv_plain_latex_data_1028")
# count_meta_data_paper("../corpus_data/meta_data_1022.jsonl")




