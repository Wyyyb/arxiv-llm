from extract_and_parse_bib import *
from post_process_training_data import *
import json
from tqdm import tqdm


sta_record = {"main tex not found": 0,
              "less than four citations found": 0,
              "intro and related work not found": 0,
              "arxiv id not found in metadata": 0,
              "bibtex not found": 0,
              "satisfied data number": 0,
              "total paper number": 0}


def single_paper_process(paper_dir):
    global sta_record
    main_tex = find_main_tex_file(paper_dir)
    if not main_tex:
        # print("main text not found in", paper_dir)
        sta_record["main tex not found"] += 1
        return None
    content = ""
    for each_main_tex in main_tex:
        curr_content = read_file_safely(each_main_tex)
        curr_content = handle_input_commands(curr_content, paper_dir)
        content = content + curr_content
    content = re.sub(r'\n{3,}', '\n\n', content)
    content = remove_comments(content)
    content = remove_figures_and_tables(content)
    title = extract_title(content)
    intro = extract_intro(content)
    related_work = extract_related_work(content)
    bib_entries = find_and_extract_bib(paper_dir, content)

    return {
        'title': title,
        'intro': intro,
        'related_work': related_work,
        'bib': bib_entries
    }


def remove_comments(content):
    # Remove comments that start with % and continue to the end of the line
    return re.sub(r'%.*$', '', content, flags=re.MULTILINE)


def remove_figures_and_tables(content):
    # 移除图片
    content = re.sub(r'\\begin{figure}.*?\\end{figure}', '', content, flags=re.DOTALL)
    content = re.sub(r'\\includegraphics(\[.*?\])?\{.*?\}', '', content)

    # 移除表格
    content = re.sub(r'\\begin{table}.*?\\end{table}', '', content, flags=re.DOTALL)
    content = re.sub(r'\\begin{tabular}.*?\\end{tabular}', '', content, flags=re.DOTALL)

    # 移除其他可能的浮动体
    content = re.sub(r'\\begin{wrapfigure}.*?\\end{wrapfigure}', '', content, flags=re.DOTALL)
    content = re.sub(r'\\begin{algorithm}.*?\\end{algorithm}', '', content, flags=re.DOTALL)

    # 移除图表标题
    content = re.sub(r'\\caption{.*?}', '', content)

    # 移除多余的空行
    content = re.sub(r'\n\s*\n', '\n\n', content)

    return content.strip()


def build_arxiv_base(input_dir):
    arxiv_base = defaultdict(dict)
    for paper_dir in tqdm(os.listdir(input_dir)):
        # if "1608." not in paper_dir:
        #     continue
        full_path = os.path.join(input_dir, paper_dir)
        if os.path.isdir(full_path):
            try:
                paper_data = single_paper_process(full_path)
            except Exception as e:
                print("Error processing", e, full_path)
                paper_data = None
            if paper_data:
                arxiv_base[paper_dir] = paper_data
    return dict(arxiv_base)


def load_meta_data(meta_data_path):
    meta_data = {}
    with open(meta_data_path, "r") as fi:
        for line in tqdm(fi.readlines()):
            curr = json.loads(line)
            meta_data[curr["docs_id"]] = curr
    print("meta data amount: ", len(meta_data))
    return meta_data


def construct(latex_dir, output_dir, failed_record_path, sta_file_path, semantic_scholar_cache_path):
    global sta_record
    if os.path.exists(semantic_scholar_cache_path):
        with open(semantic_scholar_cache_path, "r") as fi:
            semantic_scholar_cache = json.load(fi)
    else:
        semantic_scholar_cache = {}
    arxiv_base_output_dir = output_dir
    meta_data = load_meta_data("../corpus_data/meta_data_1022.jsonl")
    title_map = {}
    for k, v in meta_data.items():
        title = v["title"]
        title = clean_title(title)
        title = title.lower()
        if title not in title_map:
            title_map[title] = [k]
        else:
            title_map[title].append(k)
    failed_record = {}

    def add_message(arxiv_id, m):
        if arxiv_id not in failed_record:
            failed_record[arxiv_id] = [m]
        else:
            failed_record[arxiv_id].append(m)

    count = 0
    unqualified_count = 0
    for sub_dir in os.listdir(latex_dir):
        curr_dir = os.path.join(latex_dir, sub_dir)
        if not os.path.isdir(curr_dir):
            # print("not a dir", curr_dir)
            continue
        print("building", curr_dir)
        temp_arxiv_base = build_arxiv_base(curr_dir)
        curr_arxiv_base = {}
        print("Processing", curr_dir)
        for k, v in tqdm(temp_arxiv_base.items()):
            curr_paper = {}
            if k not in meta_data:
                add_message(k, "arxiv id not found in metadata")
                sta_record["arxiv id not found in metadata"] += 1
            curr_meta = meta_data[k]
            if not v["title"] or v["title"] == "":
                add_message(k, "title not in arxiv base")
                curr_paper["title"] = curr_meta["title"]
            else:
                curr_paper["title"] = v["title"]
            if not v["intro"] or v["intro"] == "":
                add_message(k, "intro not in arxiv base")
            else:
                curr_paper["intro"] = v["intro"]
            if not v["related_work"] or v["related_work"] == "":
                # add_message(k, "related_work not in arxiv base")
                pass
            else:
                curr_paper["related_work"] = v["related_work"]
            if not v["bib"]:
                add_message(k, "bib not in arxiv base")
                sta_record["bibtex not found"] += 1
            else:
                curr_paper["bib"] = v["bib"]
            # curr_paper["abstract"] = v["abstract"]
            curr_paper["abstract"] = curr_meta["abstract"]
            curr_paper["arxiv_id"] = k
            curr_paper = post_process(curr_paper, meta_data, title_map, semantic_scholar_cache_path,
                                      semantic_scholar_cache)
            if "intro" not in curr_paper and "related_work" not in curr_paper:
                sta_record["intro and related work not found"] += 1
                continue
            curr_arxiv_base[k] = curr_paper
            if curr_paper["satisfied_data"]:
                count += 1
                sta_record["satisfied data number"] += 1
            else:
                unqualified_count += 1
                sta_record["less than four citations found"] += 1
        if not curr_arxiv_base:
            continue
        length = len(temp_arxiv_base)
        sta_record["total paper number"] += length
        output_path = os.path.join(arxiv_base_output_dir, sub_dir + f"_{str(length)}_{str(count)}_arxiv_base.json")
        with open(output_path, "w") as fo:
            fo.write(json.dumps(curr_arxiv_base, indent=2))
        with open(failed_record_path, "w") as fo:
            fo.write(json.dumps(failed_record, indent=2))
        with open(sta_file_path, "w") as fo:
            fo.write(json.dumps(sta_record, indent=2))
    print("qualified count", count)
    print("unqualified count", unqualified_count)


def main():
    # latex_dir = "/Users/MyDisk/2024/git/cite_rag_bk/local/"
    latex_dir = "/Users/MyDisk/2024/git/arxiv-llm/local/latex_sample_1024/"
    output_dir = "../local/arxiv_base_1024_sample"
    # latex_dir = "/data/yubowang/arxiv-latex-filtered_1014"
    # output_dir = "../local/arxiv_base_1024"
    os.makedirs("../local/", exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    construct(latex_dir, output_dir, failed_record_path="../local/failed_record_1024.json",
              sta_file_path="../local/global_sta_record.json",
              semantic_scholar_cache_path="../local/semantic_scholar_cache.json")


main()

