import re
import json
import os
from tqdm import tqdm


def remove_preamble(tex_content):
    """
    移除LaTeX文档中\begin{document}之前的内容

    Args:
        tex_content (str): 完整的LaTeX文档内容

    Returns:
        str: 从\begin{document}开始的文档内容。如果未找到\begin{document}，返回原始内容
    """
    # 查找\begin{document}的位置
    begin_doc_pos = tex_content.find(r'\begin{document}')

    if begin_doc_pos != -1:
        # 找到\begin{document}，返回从这里开始的内容
        return tex_content[begin_doc_pos:]
    else:
        # 未找到\begin{document}，返回原始内容
        return tex_content


def extract_by_patterns(pattern_type):
    INTRO_PATTERN = [
        re.compile(pattern, re.DOTALL | re.IGNORECASE) for pattern in [
            # 单独的 Introduction 模式
            r'\\section{Introduction}',
            r'\\section\*{Introduction}',
            r'\\subsection{Introduction}',
            r'\\subsection\*{Introduction}',
            r'\\chapter{Introduction}',
            r'\\chapter\*{Introduction}',
            r'1\.\s+Introduction',
            r'I\.\s+Introduction',
            r'\\noindent\s*\\textbf{Introduction}',

            # Background and Introduction 模式
            r'\\section{Background and Introduction}',
            r'\\section\*{Background and Introduction}',
            r'\\chapter{Background and Introduction}',
            r'\\chapter\*{Background and Introduction}',
            r'1\.\s+Background and Introduction',
            r'I\.\s+Background and Introduction',
            r'\\noindent\s*\\textbf{Background and Introduction}',

            # Background & Introduction 模式
            r'\\section{Background & Introduction}',
            r'\\section\*{Background & Introduction}',
            r'\\chapter{Background & Introduction}',
            r'\\chapter\*{Background & Introduction}',
            r'1\.\s+Background & Introduction',
            r'I\.\s+Background & Introduction',
            r'\\noindent\s*\\textbf{Background & Introduction}',

            # Introduction and Background 模式
            r'\\section{Introduction and Background}',
            r'\\section\*{Introduction and Background}',
            r'\\chapter{Introduction and Background}',
            r'\\chapter\*{Introduction and Background}',
            r'1\.\s+Introduction and Background',
            r'I\.\s+Introduction and Background',
            r'\\noindent\s*\\textbf{Introduction and Background}',

            # Introduction & Background 模式
            r'\\section{Introduction & Background}',
            r'\\section\*{Introduction & Background}',
            r'\\chapter{Introduction & Background}',
            r'\\chapter\*{Introduction & Background}',
            r'1\.\s+Introduction & Background',
            r'I\.\s+Introduction & Background',
            r'\\noindent\s*\\textbf{Introduction & Background}',

            # 通用组合模式 (其他词 + Introduction)
            r'\\section{.+?\s+(?:and|&)\s+Introduction}',
            r'\\section\*{.+?\s+(?:and|&)\s+Introduction}',
            r'\\chapter{.+?\s+(?:and|&)\s+Introduction}',
            r'\\chapter\*{.+?\s+(?:and|&)\s+Introduction}',
            r'1\.\s+.+?\s+(?:and|&)\s+Introduction',
            r'I\.\s+.+?\s+(?:and|&)\s+Introduction',
            r'\\noindent\s*\\textbf{.+?\s+(?:and|&)\s+Introduction}',

            # Introduction + 其他词模式
            r'\\section{Introduction\s+(?:and|&)\s+.+?}',
            r'\\section\*{Introduction\s+(?:and|&)\s+.+?}',
            r'\\chapter{Introduction\s+(?:and|&)\s+.+?}',
            r'\\chapter\*{Introduction\s+(?:and|&)\s+.+?}',
            r'1\.\s+Introduction\s+(?:and|&)\s+.+?',
            r'I\.\s+Introduction\s+(?:and|&)\s+.+?',
            r'\\noindent\s*\\textbf{Introduction\s+(?:and|&)\s+.+?}'
        ]
    ]
    END_INTRO_PATTERN = re.compile(r'(?:\n\s*\\(?:section|chapter)|\n\s*(?:\d+\.|II\.))|\\Z', re.DOTALL | re.IGNORECASE)

    PRIMARY_PATTERNS = [
        re.compile(pattern, re.DOTALL | re.IGNORECASE) for pattern in [
            r'\\(?:section|subsection|chapter)\*?{(?:Related Works?|Previous Work|Prior Research)}',
            r'(?:2|II)\.\s+(?:Related Works?|Previous Work|Prior Research)',
            r'\\noindent\s*\\textbf{(?:Related Works?|Previous Work|Prior Research)}'
        ]
    ]

    SECONDARY_PATTERNS = [
        re.compile(pattern, re.DOTALL | re.IGNORECASE) for pattern in [
            r'\\(?:section|subsection|chapter)\*?{(?:Background|State of the Art|Literature Review)}',
            r'(?:2|II)\.\s+(?:Background|State of the Art|Literature Review)',
            r'\\noindent\s*\\textbf{(?:Background|State of the Art|Literature Review)}'
        ]
    ]

    END_RW_PATTERN = re.compile(r'(?:\n\s*\\(?:section|chapter|paragraph)|\n\s*'
                                r'(?:\d+\.|[IVX]+\.)|\\begin{|\\end{document})|\\Z', re.DOTALL | re.IGNORECASE)
    SUBSECTION_PATTERN = re.compile(r'\\subsection{[^}]+}')
    PARAGRAPH_PATTERN = re.compile(r'\n\n(.+?)\n\n', re.DOTALL)

    # 预定义关键词集合
    PRIMARY_KEYWORDS = {'related work', 'previous work', 'prior research'}
    SECONDARY_KEYWORDS = {'background', 'state of the art', 'literature review'}

    rw_pattern_list = [PRIMARY_PATTERNS, SECONDARY_PATTERNS, END_RW_PATTERN, SUBSECTION_PATTERN,
                       PARAGRAPH_PATTERN, PRIMARY_KEYWORDS, SECONDARY_KEYWORDS]
    if pattern_type == "intro":
        return [INTRO_PATTERN, END_INTRO_PATTERN]
    elif pattern_type == "related_work":
        return rw_pattern_list
    else:
        print("Invalid pattern type")
        return None


def extract_related_work_new(tex_content):
    # 可能的related work section标题变体
    related_headers = [
        r'\section{Related Work}',
        r'\section{Related Works}',
        r'\section{RELATED WORK}',
        r'\section{RELATED WORKS}',
        r'\section{Previous Work}',
        r'\section{Background and Related Work}',
        r'\section*{Related Work}',
        r'\section*{Related Works}',
        r'\section*{RELATED WORK}',
        r'\section*{RELATED WORKS}',
        r'\section{Background}',
        r'\section*{Background}'
    ]

    # 找到related work部分的开始
    start_pos = -1
    start_header = ''
    for header in related_headers:
        pos = tex_content.find(header)
        if pos != -1:
            if start_pos == -1 or pos < start_pos:
                start_pos = pos
                start_header = header

    if start_pos == -1:
        return ""  # 没找到related work部分

    # 直接从标题后开始
    content_start = start_pos + len(start_header)

    # 寻找下一个section作为结束位置
    next_section_markers = [
        r'\section{',
        r'\section*{',
        r'\chapter{',
        r'\chapter*{'
    ]

    end_pos = len(tex_content)
    for marker in next_section_markers:
        pos = tex_content.find(marker, content_start)
        if pos != -1 and pos < end_pos:
            end_pos = pos

    # 提取内容
    content = tex_content[content_start:end_pos].strip()

    return content


def extract_intro(pattern_list, content):
    intro_patterns, end_intro_patterns = pattern_list
    # print("extract_intro, content length", len(content))
    intro_content = None
    min_start_pos = len(content)  # 记录最早出现的介绍部分

    # 对每个模式进行搜索
    for pattern in intro_patterns:
        match = pattern.search(content)
        if match:
            start_pos = match.start()
            if start_pos < min_start_pos:
                # 找到介绍标题后，搜索结束位置
                title_end = match.end()
                end_match = end_intro_patterns.search(content, title_end)
                if end_match:
                    intro_text = content[title_end:end_match.start()].strip()
                    if intro_text:
                        intro_content = intro_text
                        # intro_content = clean_intro_content(intro_text)
                        min_start_pos = start_pos

    return intro_content


def clean_intro_content(content):
    # 移除LaTeX注释
    content = re.sub(r'(?<!\\)%.*?\n', '\n', content)

    # 移除多余的空行，但保留段落之间的单个空行
    content = re.sub(r'\n{3,}', '\n\n', content)

    # 移除行首和行尾的空白字符
    content = '\n'.join(line.strip() for line in content.split('\n'))

    return content.strip()


def extract_related_work(pattern_list, content):
    primary_patterns, secondary_patterns, end_rw_pattern, subsection_pattern, \
        paragraph_pattern, primary_keywords, secondary_keywords = pattern_list

    def search_patterns(patterns, text):
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                # 找到标题后搜索结束位置
                title_end = match.end()
                end_match = end_rw_pattern.search(text, title_end)
                if end_match:
                    return match.group(0), text[title_end:end_match.start()].strip()
        return None, None

    # 首先搜索主要模式
    title, related_work_content = search_patterns(primary_patterns, content)

    # 如果没有找到，搜索次要模式
    if not related_work_content:
        title, related_work_content = search_patterns(secondary_patterns, content)

    if related_work_content:
        # 处理可能的子节
        subsections = subsection_pattern.split(related_work_content)
        if len(subsections) > 1:
            related_work_content = '\n\n'.join(subsections)
        return related_work_content.strip()

    # 如果没有找到明确的部分，尝试查找可能包含相关工作内容的段落
    potential_paragraphs = paragraph_pattern.findall(content)

    # 检查主要关键词
    for paragraph in potential_paragraphs:
        paragraph_lower = paragraph.lower()
        if any(keyword in paragraph_lower for keyword in primary_keywords):
            return paragraph.strip()

    # 检查次要关键词
    for paragraph in potential_paragraphs:
        paragraph_lower = paragraph.lower()
        if any(keyword in paragraph_lower for keyword in secondary_keywords):
            return paragraph.strip()
    return None


def extract_title(content):
    match = re.search(r'\\title{(.+?)}', content, re.DOTALL)
    if not match:
        match = re.search(r'\\title\[.+?\]{(.+?)}', content, re.DOTALL)
    if match:
        title = match.group(1)
        # Remove newlines and extra spaces
        title = re.sub(r'\s+', ' ', title).strip()
        title = clean_title(title)
        return title
    return None


def clean_title(title):
    pattern = r'[\s\x00-\x20\x7f\xa0]'
    title = re.sub(pattern, ' ', title)
    title = title.replace("{", "").replace("}", "").replace(".", "").replace("\n", "")
    while "  " in title:
        title = title.replace("  ", " ")
    return title


def extract_bbl_items(bbl_content):
    # 存储所有bibitem的列表
    bibitems = []

    # 分行处理
    lines = bbl_content.split('\n')

    # 当前bibitem的内容
    current_bibitem = []
    in_bibitem = False

    for line in lines:
        line = line.strip()

        # 检查bibitem的开始
        if line.startswith('\\bibitem'):
            # 如果之前有未结束的bibitem,先保存它
            if in_bibitem and current_bibitem:
                bibitems.append('\n'.join(current_bibitem))

            # 开始新的bibitem
            current_bibitem = [line]
            in_bibitem = True

        # 如果在bibitem内,继续收集内容
        elif in_bibitem:
            # 检查是否到达bibitem结尾
            if line.startswith('\\bibitem') or line.startswith('\\end{thebibliography}'):
                bibitems.append('\n'.join(current_bibitem))
                current_bibitem = []
                in_bibitem = False

                # 如果是新的bibitem,开始处理它
                if line.startswith('\\bibitem'):
                    current_bibitem = [line]
                    in_bibitem = True
            else:
                current_bibitem.append(line)

    # 处理最后一个bibitem
    if in_bibitem and current_bibitem:
        bibitems.append('\n'.join(current_bibitem))

    return bibitems


def extract_bib_citations(bib_content):
    citations = []
    current_citation = []
    in_citation = False
    brace_count = 0

    # 分行处理
    lines = bib_content.split('\n')

    for line in lines:
        line = line.strip()

        # 跳过空行和注释行
        if not line or line.startswith('%'):
            continue

        # 检查是否是一个新的引用条目开始
        if line.startswith('@'):
            # 如果之前有未完成的引用，先保存它
            if in_citation and current_citation:
                citations.append('\n'.join(current_citation))
                current_citation = []

            in_citation = True
            brace_count = 0
            current_citation = [line]

            # 计算这一行中的大括号数量
            brace_count += line.count('{') - line.count('}')

        # 如果在引用条目内
        elif in_citation:
            current_citation.append(line)
            brace_count += line.count('{') - line.count('}')

            # 如果大括号配对完成，说明当前引用条目结束
            if brace_count == 0:
                citations.append('\n'.join(current_citation))
                current_citation = []
                in_citation = False

    # 处理最后一个引用条目
    if in_citation and current_citation:
        citations.append('\n'.join(current_citation))

    return citations


def extract_abstract(tex_content):
    # 定义可能的abstract开始和结束标记
    abstract_patterns = [
        (r'\begin{abstract}', r'\end{abstract}'),
        (r'\begin{Abstract}', r'\end{Abstract}'),
        (r'\abstract{', '}'),  # 某些模板使用\abstract{...}格式
        (r'\begin{ABSTRACT}', r'\end{ABSTRACT}')
    ]

    for start_pattern, end_pattern in abstract_patterns:
        # 查找起始位置
        # print("start_pattern, end_pattern", start_pattern, end_pattern)
        start_pos = tex_content.find(start_pattern)
        if start_pos == -1:
            continue

        # 找到起始位置后，查找对应的结束位置
        content_start = start_pos + len(start_pattern)
        end_pos = tex_content.find(end_pattern, content_start)

        if end_pos != -1:
            # 提取abstract内容
            abstract_text = tex_content[content_start:end_pos].strip()

            abstract_text = ' '.join(abstract_text.split())

            # 移除可能的LaTeX注释
            abstract_text = re.sub(r'(?<!\\)%.*$', '', abstract_text, flags=re.MULTILINE)

            return abstract_text

    return None


def check_unique_occurrence(content, text):
    if not text:
        return False

    first_pos = content.find(text)
    if first_pos == -1:
        return False

    # 从第一次出现位置之后继续查找
    second_pos = content.find(text, first_pos + 1)
    return second_pos == -1  # 如果找不到第二次出现，则为唯一


def get_other_tex(content, intro, related_work):
    # 检查文本片段是否唯一
    if not check_unique_occurrence(content, intro):
        print("Warning: Introduction content appears multiple times or not found")
        return content

    if not check_unique_occurrence(content, related_work):
        print("Warning: Related work content appears multiple times or not found")
        intro_end_index = content.find(intro) + len(intro)
        return content[intro_end_index:]

    # 如果introduction为空或内容太少,返回全部内容
    if not intro or len(intro) < 100:
        return content

    # 如果related work为空或内容太少,返回introduction之后的内容
    if not related_work or len(related_work) < 100:
        intro_end_index = content.find(intro) + len(intro)
        return content[intro_end_index:]

    other_tex = ""

    # 精确定位introduction的起始和结束位置
    intro_start_index = content.find(intro)
    intro_end_index = intro_start_index + len(intro)

    # 精确定位related work的起始和结束位置
    related_work_start_index = content.find(related_work)
    related_work_end_index = related_work_start_index + len(related_work)

    # 确保找到了两段文本，且related work在introduction之后
    if (intro_start_index != -1 and related_work_start_index != -1 and
            related_work_start_index > intro_start_index):

        # 如果introduction和related work之间间隔足够大，提取中间内容
        if related_work_start_index - intro_end_index > 2000:
            other_tex += content[intro_end_index:related_work_start_index]

        # 添加related work之后的内容
        other_tex += content[related_work_end_index:]

    return other_tex


def extract_parts(intro_patterns, related_work_patterns, paper_dir_path):
    message = {"no_intro_no_rw": [], "no_intro": [], "no_related_work": [], "no_bib_citations": [],
               "no_abstract": [], "no_main_tex_file": [], "no_full_tex_file": []}
    arxiv_id = paper_dir_path.split("/")[-1]
    if not os.path.exists(os.path.join(paper_dir_path, "main_tex_25_step_2_result_0109.tex")):
        message["no_main_tex_file"].append(arxiv_id)
        if not os.path.exists(os.path.join(paper_dir_path, "full.tex")):
            message["no_full_tex_file"].append(arxiv_id)
            return None
    with open(os.path.join(paper_dir_path, "main_tex_25_step_2_result_0109.tex"), "r") as fi:
        content = fi.read()
    content = remove_preamble(content)
    title = extract_title(content)
    abstract = extract_abstract(content)
    intro = extract_intro(intro_patterns, content)
    related_work = extract_related_work_new(content)
    other_tex = get_other_tex(content, intro, related_work, arxiv_id)
    bib_items = extract_bib_citations(content)
    bbl_items = extract_bbl_items(content)
    if os.path.exists(os.path.join(paper_dir_path, "reference.bbl")):
        with open(os.path.join(paper_dir_path, "reference.bbl"), "r") as fi:
            reference_content = fi.read()
            temp = extract_bbl_items(reference_content)
            for each in temp:
                if each not in bbl_items:
                    bbl_items.append(each)
    if os.path.exists(os.path.join(paper_dir_path, "reference.bib")):
        with open(os.path.join(paper_dir_path, "reference.bib"), "r") as fi:
            reference_content = fi.read()
            temp = extract_bib_citations(reference_content)
            for each in temp:
                if each not in bib_items:
                    bib_items.append(each)
    if intro:
        intro = clean_intro_content(intro)
    step_2_info = {"arxiv_id": arxiv_id, "title": title, "intro": intro,
                   "related_work": related_work, "other_tex": other_tex,
                   "bib_items": bib_items, "bbl_items": bbl_items}
    with open(os.path.join(paper_dir_path, "parsed_parts_25_step_3_result_0109.json"), "w") as fo:
        fo.write(json.dumps(step_2_info, indent=2))
    if not abstract:
        message["no_abstract"].append(arxiv_id)
    if not intro:
        message["no_intro"].append(arxiv_id)
    if not related_work:
        message["no_related_work"].append(arxiv_id)
    if not bib_items and not bbl_items:
        message["no_bib_citations"].append(arxiv_id)
    if not intro and not related_work:
        message["no_intro_no_rw"].append(arxiv_id)
    return message


def run_on_darth_server(input_dir, output_log_path):
    output_log = {"no_intro_no_rw": [], "no_intro": [], "no_related_work": [],
                  "no_bib_citations": [], "no_abstract": [], "no_main_tex_file": [], "no_full_tex_file": []}
    intro_patterns = extract_by_patterns("intro")
    related_work_patterns = extract_by_patterns("related_work")
    total = 0
    for sub_dir in os.listdir(input_dir):
        print("Processing", sub_dir)
        if os.path.isdir(os.path.join(input_dir, sub_dir)):
            for paper_dir in tqdm(os.listdir(os.path.join(input_dir, sub_dir))):
                if not paper_dir.startswith(sub_dir):
                    print("skip", paper_dir)
                    continue
                total += 1
                paper_dir_path = os.path.join(input_dir, sub_dir, paper_dir)
                message = extract_parts(intro_patterns, related_work_patterns, paper_dir_path)
                for k, v in message.items():
                    output_log[k] += v
        print("Current papers processed:", total)
        for k, v in output_log.items():
            print("output_log update:\n", k, len(v))
    print("Total papers processed:", total)
    for k, v in output_log.items():
        print(k, len(v))
    with open(output_log_path, "w") as fo:
        fo.write(json.dumps(output_log, indent=2))
    return


if __name__ == "__main__":
    # os.makedirs("../local_1028", exist_ok=True)
    run_on_darth_server("/data/yubowang/arxiv_plain_latex_data_1028", "step_2_log_0109.json")



