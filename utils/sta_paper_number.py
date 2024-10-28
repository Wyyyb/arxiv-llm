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


count_paper_number("/data/yubowang/arxiv-latex-filtered_1014")
count_paper_number("/data/yubowang/arxiv_plain_latex_data_1028")





