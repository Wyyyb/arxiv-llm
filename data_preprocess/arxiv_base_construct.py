import os
import re
from collections import defaultdict
from collections import OrderedDict
import json
import chardet


def find_main_tex_file(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.tex'):
                file_path = os.path.join(root, file)
                content = read_file_safely(file_path)
                if '\\begin{document}' in content:
                    return os.path.join(root, file)
    return None


def extract_title(content):
    match = re.search(r'\\title{(.+?)}', content, re.DOTALL)
    if not match:
        match = re.search(r'\\title\[.+?\]{(.+?)}', content, re.DOTALL)
    if match:
        title = match.group(1)
        # Remove newlines and extra spaces
        title = re.sub(r'\s+', ' ', title).strip()
        return title
    return None


def handle_input_commands(content, base_dir, depth=0, max_depth=10):
    def replace_input(match):
        if depth >= max_depth:
            # print(f"Warning: Maximum recursion depth ({max_depth}) reached. Stopping recursion.")
            return match.group(0)

        input_file = match.group(1)
        input_path = os.path.join(base_dir, input_file)
        if not input_path.endswith('.tex'):
            input_path += '.tex'

        if os.path.exists(input_path):
            file_content = read_file_safely(input_path)
            # with open(input_path, 'r', encoding='utf-8') as f:
            #     file_content = f.read()
            # Recursively handle inputs in the included file
            return handle_input_commands(file_content, os.path.dirname(input_path), depth + 1, max_depth)
        else:
            # print(f"Warning: Input file not found: {input_path}")
            return match.group(0)

    content = re.sub(r'\\input{(.+?)}', replace_input, content)
    return content


def remove_comments(content):
    # Remove comments that start with % and continue to the end of the line
    return re.sub(r'%.*$', '', content, flags=re.MULTILINE)


def extract_intro(content, base_dir):
    # 定义可能的介绍部分标记模式
    intro_patterns = [
        r'\\section{Introduction}',
        r'\\section{INTRODUCTION}',
        r'\\section\*{Introduction}',
        r'\\section\*{INTRODUCTION}',
        r'\\subsection{Introduction}',
        r'\\subsection{INTRODUCTION}',
        r'\\subsection\*{Introduction}',
        r'\\subsection\*{INTRODUCTION}',
        r'\\chapter{Introduction}',
        r'\\chapter{INTRODUCTION}',
        r'\\chapter\*{Introduction}',
        r'\\chapter\*{INTRODUCTION}',
        r'1\.\s+Introduction',
        r'1\.\s+INTRODUCTION',
        r'I\.\s+Introduction',
        r'I\.\s+INTRODUCTION',
        r'\\noindent\s*\\textbf{Introduction}',
        r'\\noindent\s*\\textbf{INTRODUCTION}'
    ]

    # 组合所有模式
    combined_pattern = '|'.join(f'({pattern})' for pattern in intro_patterns)

    # 搜索介绍部分
    intro_match = re.search(f'({combined_pattern})(.+?)\\n(\\\\section|\\\\chapter|\\d+\\.|II\\.|\\\\begin{{)', content,
                            re.DOTALL | re.IGNORECASE)

    if intro_match:
        # 获取匹配的标题和内容
        intro_title = intro_match.group(1)
        intro_content = intro_match.group(len(intro_patterns) + 2).strip()

        if intro_content:
            # print(f"Found introduction with pattern: {intro_title}")
            return intro_content

    # print("Introduction not found")
    return None


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


def extract_bib_entries(content):
    # Regular expression to match BibTeX entries
    bib_pattern = r'@\w+\s*\{[^@]*\}'

    # Find all matches in the content
    bib_entries = re.findall(bib_pattern, content, re.DOTALL)

    # Clean up each entry (remove leading/trailing whitespace)
    bib_entries = [entry.strip() for entry in bib_entries]

    return bib_entries


def extract_bibitem_entries(content):
    formatted_entries = []

    # 分割内容为单独的 bibitem 条目
    bibitem_pattern = r'\\bibitem(?:\[([^\]]+)\])?\{([^}]+)\}((?:(?!\\bibitem).|\n)*)'
    bibitem_entries = re.findall(bibitem_pattern, content, re.DOTALL)

    for entry in bibitem_entries:
        citation, key, details = entry

        # 检查是否符合包含 \newblock 的样式
        newblock_pattern = r'([^\n]+)\n\\newblock ([^\n]+)\n\\newblock (.+)'
        newblock_match = re.match(newblock_pattern, details.strip(), re.DOTALL)

        if newblock_match:
            # 处理包含 \newblock 的条目
            authors, title, journal_info = newblock_match.groups()

            # 清理和格式化信息
            authors = authors.strip().rstrip('.')
            title = title.strip()
            journal_info = re.sub(r'\s+', ' ', journal_info).strip()

            # 提取期刊名称
            journal_match = re.search(r'\\em ([^}]+)', journal_info)
            journal = journal_match.group(1) if journal_match else ""

            # 提取年份
            year_match = re.search(r'\((\d{4})\)', authors)
            year = year_match.group(1) if year_match else ""

        else:
            # 处理不包含 \newblock 的条目
            details = re.sub(r'\s+', ' ', details).strip()

            # 提取作者和标题
            author_title_match = re.match(r'(.+?), ``(.+?),''', details)
            if author_title_match:
                authors = author_title_match.group(1)
                title = author_title_match.group(2)
            else:
                authors = "Unknown"
                title = "Unknown"

            # 提取期刊/arxiv 和年份
            journal_match = re.search(r'\\emph{([^}]+)}', details)
            journal = journal_match.group(1) if journal_match else "Unknown"

            year_match = re.search(r'(\d{4})\.?$', details)
            year = year_match.group(1) if year_match else "Unknown"

        # 格式化为 BibTeX 样式的条目
        formatted_entry = f"@article{{{key},\n"
        formatted_entry += f"  author = {{{authors}}},\n"
        formatted_entry += f"  title = {{{title}}},\n"
        formatted_entry += f"  journal = {{{journal}}},\n"
        formatted_entry += f"  year = {{{year}}},\n"
        formatted_entry += f"  note = {{{details.strip()}}}\n"
        formatted_entry += "}"
        formatted_entries.append(formatted_entry)

    return formatted_entries


def read_file_safely(file_path):
    try:
        # 首先尝试 UTF-8
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # 如果 UTF-8 失败，使用 chardet 检测编码
        with open(file_path, 'rb') as f:
            raw_data = f.read()
        detected = chardet.detect(raw_data)
        encoding = detected['encoding']

        # 使用检测到的编码重新读取
        try:
            return raw_data.decode(encoding)
        except:
            # 如果仍然失败，使用 'replace' 错误处理
            return raw_data.decode(encoding, errors='replace')


def find_and_extract_bib(directory, main_tex_content):
    res = []
    # 首先尝试在主 .tex 文件中查找引用
    bib_entries = extract_bib_entries(main_tex_content)
    if bib_entries:
        res += bib_entries
    bib_entries = extract_bibitem_entries(main_tex_content)
    if bib_entries:
        res += bib_entries

    # 如果主文件中没有找到，则查找 .bib 文件
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith('.bib'):
                content = read_file_safely(file_path)
                res += extract_bib_entries(content)
            elif file.endswith('.bbl') or file.endswith('.bblx'):
                content = read_file_safely(file_path)
                res += extract_bibitem_entries(content)
    bib_info_map = {}
    for each in res:
        curr = parse_bibtex(each)
        if not curr or "key" not in curr:
            continue
        bib_info_map[curr["key"]] = curr
    return bib_info_map


def parse_bibtex(bibtex_str):
    result = {}

    type_key_pattern = r'@(\w+)\{([^,]+),'
    match = re.match(type_key_pattern, bibtex_str)
    if match:
        result['type'] = match.group(1)
        result['key'] = match.group(2)

    field_pattern = r'(\w+)\s*=\s*\{([^}]+)\}'
    fields = re.findall(field_pattern, bibtex_str)

    for field, value in fields:
        result[field.lower()] = value.strip()

    return result


def process_paper(paper_dir):
    main_tex = find_main_tex_file(paper_dir)
    if not main_tex:
        return None
    content = read_file_safely(main_tex)
    # with open(main_tex, 'r', encoding='utf-8') as f:
    #     content = f.read()
    content = handle_input_commands(content, paper_dir)
    content = remove_comments(content)
    content = remove_figures_and_tables(content)
    title = extract_title(content)
    intro = extract_intro(content, paper_dir)
    bib_entries = find_and_extract_bib(paper_dir, content)

    return {
        'title': title,
        'intro': intro,
        'bib': bib_entries
    }


def build_arxiv_base(input_dir):
    arxiv_base = defaultdict(dict)
    for paper_dir in os.listdir(input_dir):
        # if paper_dir != "2409.19922":
        #     continue
        full_path = os.path.join(input_dir, paper_dir)
        if os.path.isdir(full_path):
            try:
                # print("Building", full_path)
                paper_data = process_paper(full_path)
            except Exception as e:
                print("Error processing", e, full_path)
                paper_data = None
            # paper_data = process_paper(full_path)
            if paper_data:
                arxiv_base[paper_dir] = paper_data
    return dict(arxiv_base)


def load_meta_data(meta_data_dir="meta_data"):
    arxiv_ids = {}
    for file in os.listdir(meta_data_dir):
        if not file.endswith('.json'):
            continue
        file_path = os.path.join(meta_data_dir, file)
        print("loading metadata", file_path)
        with open(file_path, "r") as fi:
            for line in fi.readlines():
                curr = json.loads(line)
                if "physics" not in curr["categories"] and "/" not in curr["id"]:
                    arxiv_ids[curr["id"]] = {"arxiv_id": curr["id"], "update_date": curr["update_date"],
                                             "abstract": curr["abstract"], "title": curr["title"]}
                elif "cs/" in curr["id"] and "physics" not in curr["id"]:
                    arxiv_ids[curr["id"]] = {"arxiv_id": curr["id"], "update_date": curr["update_date"],
                                             "abstract": curr["abstract"], "title": curr["title"]}
    print("metadata number:", len(arxiv_ids))
    return arxiv_ids


def extract_citation_keys(latex_text):
    patterns = [
        r'(~?\\cite{([^}]+)})',
        r'(~?\\citet{([^}]+)})',
        r'(~?\\citep{([^}]+)})',
        r'(~?\\cite\[([^\]]+)\]{([^}]+)})',
        r'(~?\\citealp{([^}]+)})',
        r'(~?\\citeauthor{([^}]+)})',
        r'(~?\\citeyear{([^}]+)})',
        r'(~?\\citealt{([^}]+)})',
        r'(~?\\citep\[([^\]]+)\]{([^}]+)})',
        r'(~?\\citet\[([^\]]+)\]{([^}]+)})',
        r'(~?\\parencite{([^}]+)})',
        r'(~?\\textcite{([^}]+)})',
        r'(~?\\autocite{([^}]+)})'
    ]

    citations_dict = OrderedDict()

    combined_pattern = '|'.join(f'(?:{pattern})' for pattern in patterns)

    matches = re.finditer(combined_pattern, latex_text)

    for match in matches:
        full_citation = match.group(0)  # 完整的引用字符串
        # 提取键（可能包含多个键）
        groups = match.groups()
        keys = next((group for group in reversed(groups) if group is not None), '')

        # 检查是否是多篇文章引用
        is_multi_citation = ',' in keys

        for key in keys.split(','):
            key = key.strip()
            if key not in citations_dict:
                citations_dict[key] = (full_citation, is_multi_citation)
            elif len(full_citation) > len(citations_dict[key][0]):
                # 保留更长的引用，但保持原有的多篇引用标记
                citations_dict[key] = (full_citation, citations_dict[key][1] or is_multi_citation)

    return citations_dict


def clean_title(title):
    pattern = r'[\s\x00-\x20\x7f\xa0]'
    title = re.sub(pattern, ' ', title)
    title = title.replace("{", "").replace("}", "").replace(".", "").replace("\n", "")
    while "  " in title:
        title = title.replace("  ", " ")
    return title.lower()


def post_process(curr_paper, meta_data, title_map):
    # print("curr_paper", curr_paper)
    arxiv_id = curr_paper["arxiv_id"]
    if "intro" not in curr_paper:
        curr_paper["satisfied_data"] = False
        return curr_paper
    intro = curr_paper["intro"]
    citation_dicts = extract_citation_keys(intro)
    index = 0
    targets = {}
    for cite_key, (cite_value, is_multi) in citation_dicts.items():
        if "bib" not in curr_paper or cite_key not in curr_paper["bib"]:
            intro = intro.replace(cite_value, "")
            continue
        bib_item = curr_paper["bib"][cite_key]
        if "title" in bib_item and bib_item["title"] != "Unknown":
            title = bib_item["title"]
            title = clean_title(title)
        else:
            intro = intro.replace(cite_value, "")
            continue
        if title not in title_map:
            print("arxiv id", arxiv_id)
            print("unmatched title:", title)
            intro = intro.replace(cite_value, "")
            continue
        cite_arxiv_id = title_map[title][0]
        # print("abs metadata", meta_data[cite_arxiv_id])
        abstract = meta_data[cite_arxiv_id]["abstract"]
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
    # paper = "<|paper_start|>" + intro + "<|paper_end|>"
    paper = intro
    data = {"arxiv_id": arxiv_id, "paper": paper, "targets": targets}
    curr_paper["data"] = data
    curr_paper["satisfied_data"] = index >= 4
    return curr_paper


def main():
    # latex_dir = "/Users/MyDisk/2024/git/cite_rag_bk/local/"
    latex_dir = "/data/yubowang/arxiv-latex-filtered_1014"
    # latex_dir = "../local/"
    os.makedirs("../local/", exist_ok=True)
    arxiv_base_output_dir = "../local/arxiv_base/"
    os.makedirs(arxiv_base_output_dir, exist_ok=True)
    failed_record_path = "../local/failed_record.json"
    meta_data = load_meta_data()
    title_map = {}
    for k, v in meta_data.items():
        title = v["title"]
        title = clean_title(title)
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

    for sub_dir in os.listdir(latex_dir):
        count = 0
        curr_dir = os.path.join(latex_dir, sub_dir)
        if not os.path.isdir(curr_dir):
            print("not a dir", curr_dir)
            continue
        temp_arxiv_base = build_arxiv_base(curr_dir)
        curr_arxiv_base = {}
        print("Processing", curr_dir)
        for k, v in temp_arxiv_base.items():
            curr_paper = {}
            if k not in meta_data:
                add_message(k, "arxiv id not found in metadata")
                continue
            curr_meta = meta_data[k]
            if not v["title"] or v["title"] == "":
                add_message(k, "title not in arxiv base")
            else:
                curr_paper["title"] = v["title"]
            if not v["intro"] or v["intro"] == "":
                add_message(k, "intro not in arxiv base")
            else:
                curr_paper["intro"] = v["intro"]
            if not v["bib"]:
                add_message(k, "bib not in arxiv base")
            else:
                curr_paper["bib"] = v["bib"]
            curr_paper["abstract"] = curr_meta["abstract"]
            if curr_meta["title"] != v["title"] and curr_meta["title"] != "":
                curr_paper["bias"] = curr_meta["title"]
            curr_paper["update_date"] = curr_meta["update_date"]
            curr_paper["arxiv_id"] = k
            curr_paper = post_process(curr_paper, meta_data, title_map)
            curr_arxiv_base[k] = curr_paper
            if curr_paper["satisfied_data"]:
                count += 1
        if not curr_arxiv_base:
            continue
        length = len(curr_arxiv_base)
        output_path = os.path.join(arxiv_base_output_dir, sub_dir + f"_{str(length)}_{str(count)}_arxiv_base.json")
        with open(output_path, "w") as fo:
            fo.write(json.dumps(curr_arxiv_base, indent=2))
        with open(failed_record_path, "w") as fo:
            fo.write(json.dumps(failed_record, indent=2))


if __name__ == "__main__":
    main()

