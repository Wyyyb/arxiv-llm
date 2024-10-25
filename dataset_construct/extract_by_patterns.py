import re
from collections import defaultdict, OrderedDict

# 在模块级别预编译所有正则表达式
INTRO_PATTERNS = [
    re.compile(pattern, re.DOTALL | re.IGNORECASE) for pattern in [
        # LaTeX 命令模式 (section/chapter 带或不带星号)
        r'\\(?:section|subsection|chapter)\*?{(?:Introduction|INTRODUCTION)}',
        # 编号模式
        r'(?:1|I)\.\s+(?:Introduction|INTRODUCTION)',
        # noindent 模式
        r'\\noindent\s*\\textbf{(?:Introduction|INTRODUCTION)}',
        # Introduction 与其他词组合的 LaTeX 命令模式
        r'\\(?:section|chapter)\*?{(?:.+?\s+(?:and|&)\s+Introduction|Introduction\s+(?:and|&)\s+.+?)}',
        # Introduction 与其他词组合的编号模式
        r'(?:1|I)\.\s+(?:.+?\s+(?:and|&)\s+Introduction|Introduction\s+(?:and|&)\s+.+?)',
        # Introduction 与其他词组合的 noindent 模式
        r'\\noindent\s*\\textbf{(?:.+?\s+(?:and|&)\s+Introduction|Introduction\s+(?:and|&)\s+.+?)}'
    ]
]

# 预编译结束模式
END_INTRO_PATTERN = re.compile(r'\\n(?:\\section|\\chapter|\d+\.|II\.)|\\Z', re.DOTALL | re.IGNORECASE)


def extract_intro(content):
    intro_content = None
    min_start_pos = len(content)  # 记录最早出现的介绍部分

    # 对每个模式进行搜索
    for pattern in INTRO_PATTERNS:
        match = pattern.search(content)
        if match:
            start_pos = match.start()
            if start_pos < min_start_pos:
                # 找到介绍标题后，搜索结束位置
                title_end = match.end()
                end_match = END_INTRO_PATTERN.search(content, title_end)
                if end_match:
                    intro_text = content[title_end:end_match.start()].strip()
                    if intro_text:
                        intro_content = clean_intro_content(intro_text)
                        min_start_pos = start_pos

    return intro_content


def clean_intro_content(content):
    # 移除LaTeX注释
    # content = re.sub(r'%.*?\n', '\n', content)
    content = re.sub(r'(?<!\\)%.*?\n', '\n', content)

    # 移除多余的空行，但保留段落之间的单个空行
    content = re.sub(r'\n{3,}', '\n\n', content)

    # 移除行首和行尾的空白字符
    content = '\n'.join(line.strip() for line in content.split('\n'))

    return content.strip()


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


# 预编译所有正则表达式
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

# 预编译其他常用模式
END_RW_PATTERN = re.compile(r'\\n(\\section|\\chapter|\d+\.|[IV]+\.|\begin{|\end{document})', re.DOTALL | re.IGNORECASE)
SUBSECTION_PATTERN = re.compile(r'\\subsection{[^}]+}')
PARAGRAPH_PATTERN = re.compile(r'\n\n(.+?)\n\n', re.DOTALL)

# 预定义关键词集合
PRIMARY_KEYWORDS = {'related work', 'previous work', 'prior research'}
SECONDARY_KEYWORDS = {'background', 'state of the art', 'literature review'}


def extract_related_work(content):
    def search_patterns(patterns, text):
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                # 找到标题后搜索结束位置
                title_end = match.end()
                end_match = END_RW_PATTERN.search(text, title_end)
                if end_match:
                    return match.group(0), text[title_end:end_match.start()].strip()
        return None, None

    # 首先搜索主要模式
    title, related_work_content = search_patterns(PRIMARY_PATTERNS, content)

    # 如果没有找到，搜索次要模式
    if not related_work_content:
        title, related_work_content = search_patterns(SECONDARY_PATTERNS, content)

    if related_work_content:
        # 处理可能的子节
        subsections = SUBSECTION_PATTERN.split(related_work_content)
        if len(subsections) > 1:
            related_work_content = '\n\n'.join(subsections)
        return related_work_content.strip()

    # 如果没有找到明确的部分，尝试查找可能包含相关工作内容的段落
    potential_paragraphs = PARAGRAPH_PATTERN.findall(content)

    # 检查主要关键词
    for paragraph in potential_paragraphs:
        paragraph_lower = paragraph.lower()
        if any(keyword in paragraph_lower for keyword in PRIMARY_KEYWORDS):
            return paragraph.strip()

    # 检查次要关键词
    for paragraph in potential_paragraphs:
        paragraph_lower = paragraph.lower()
        if any(keyword in paragraph_lower for keyword in SECONDARY_KEYWORDS):
            return paragraph.strip()

    return None


# def extract_related_work(content):
#     primary_patterns = [
#         # LaTeX 命令模式 (section/subsection/chapter 带或不带星号)
#         r'\\(?:section|subsection|chapter)\*?{(?:Related Works?|Previous Work|Prior Research)}',
# 
#         # 编号模式
#         r'(?:2|II)\.\s+(?:Related Works?|Previous Work|Prior Research)',
# 
#         # noindent 模式
#         r'\\noindent\s*\\textbf{(?:Related Works?|Previous Work|Prior Research)}'
#     ]
# 
#     secondary_patterns = [
#         # LaTeX 命令模式 (section/subsection/chapter 带或不带星号)
#         r'\\(?:section|subsection|chapter)\*?{(?:Background|State of the Art|Literature Review)}',
# 
#         # 编号模式
#         r'(?:2|II)\.\s+(?:Background|State of the Art|Literature Review)',
# 
#         # noindent 模式
#         r'\\noindent\s*\\textbf{(?:Background|State of the Art|Literature Review)}'
#     ]
# 
#     def search_patterns(patterns, text):
#         combined_pattern = '|'.join(f'({pattern})' for pattern in patterns)
#         match = re.search(
#             f'({combined_pattern})(.+?)\\n(\\\\section|\\\\chapter|\\d+\\.|[IV]+\\.|\\\\begin{{|\\\\end{{document}})',
#             text,
#             re.DOTALL | re.IGNORECASE
#         )
#         if match:
#             this_title = match.group(1)
#             this_content = match.group(len(patterns) + 2).strip()
#             return this_title, this_content
#         return None, None
# 
#     # 首先搜索主要模式
#     title, related_work_content = search_patterns(primary_patterns, content)
# 
#     # 如果没有找到，搜索次要模式
#     if not related_work_content:
#         title, related_work_content = search_patterns(secondary_patterns, content)
# 
#     if related_work_content:
#         # 处理可能的子节
#         subsections = re.split(r'\\subsection{[^}]+}', related_work_content)
#         if len(subsections) > 1:
#             related_work_content = '\n\n'.join(subsections)
# 
#         return related_work_content.strip()
# 
#     # 如果没有找到明确的部分，尝试查找可能包含相关工作内容的段落
#     potential_paragraphs = re.findall(r'\n\n(.+?)\n\n', content, re.DOTALL)
#     for paragraph in potential_paragraphs:
#         if any(keyword in paragraph.lower() for keyword in ['related work', 'previous work', 'prior research']):
#             return paragraph.strip()
# 
#     # 如果仍然没有找到，尝试查找次要关键词
#     for paragraph in potential_paragraphs:
#         if any(keyword in paragraph.lower() for keyword in ['background', 'state of the art', 'literature review']):
#             return paragraph.strip()
# 
#     return None


def extract_citation_keys(latex_text):
    patterns = [
        # 基础引用命令
        r'(~?\\cite{([^}]+)})',
        r'(~?\\citet{([^}]+)})',
        r'(~?\\citep{([^}]+)})',
        r'(~?\\citealp{([^}]+)})',
        r'(~?\\citeauthor{([^}]+)})',
        r'(~?\\citeyear{([^}]+)})',
        r'(~?\\citealt{([^}]+)})',
        r'(~?\\parencite{([^}]+)})',
        r'(~?\\textcite{([^}]+)})',
        r'(~?\\autocite{([^}]+)})',

        # 带选项的引用命令
        r'(~?\\cite\[([^\]]+)\]{([^}]+)})',
        r'(~?\\citep\[([^\]]+)\]{([^}]+)})',
        r'(~?\\citet\[([^\]]+)\]{([^}]+)})',
        r'(~?\\parencite\[([^\]]+)\]{([^}]+)})',
        r'(~?\\autocite\[([^\]]+)\]{([^}]+)})',

        # 带前后缀的引用命令
        r'(~?\\cite\[([^\]]+)\]\[([^\]]+)\]{([^}]+)})',
        r'(~?\\citep\[([^\]]+)\]\[([^\]]+)\]{([^}]+)})',
        r'(~?\\parencite\[([^\]]+)\]\[([^\]]+)\]{([^}]+)})',

        # natbib 特殊命令
        r'(~?\\citenum{([^}]+)})',
        r'(~?\\citeyearpar{([^}]+)})',
        r'(~?\\citetext{([^}]+)})',
        r'(~?\\citepalias{([^}]+)})',
        r'(~?\\citetalias{([^}]+)})',

        # biblatex 特殊命令
        r'(~?\\smartcite{([^}]+)})',
        r'(~?\\footcite{([^}]+)})',
        r'(~?\\footcitetext{([^}]+)})',
        r'(~?\\supercite{([^}]+)})',
        r'(~?\\autocite\*{([^}]+)})',
        r'(~?\\citeauthor\*{([^}]+)})',
        r'(~?\\parencite\*{([^}]+)})',
        r'(~?\\textcite\*{([^}]+)})',

        # 多重引用命令
        r'(~?\\cites{([^}]+)})',
        r'(~?\\parencites{([^}]+)})',
        r'(~?\\textcites{([^}]+)})',
        r'(~?\\autocites{([^}]+)})',

        # 带注释的引用
        r'(~?\\nocite{([^}]+)})',
        r'(~?\\citefield{([^}]+)}{([^}]+)})',
        r'(~?\\citereset{([^}]+)})',

        # 自定义格式引用
        r'(~?\\citestyle{([^}]+)})',
        r'(~?\\defcitealias{([^}]+)}{([^}]+)})',

        # 特殊用途引用
        r'(~?\\fullcite{([^}]+)})',
        r'(~?\\footfullcite{([^}]+)})',
        r'(~?\\volcite{([^}]+)}{([^}]+)})',
        r'(~?\\pvolcite{([^}]+)}{([^}]+)})',
        r'(~?\\tvolcite{([^}]+)}{([^}]+)})',
        r'(~?\\avolcite{([^}]+)}{([^}]+)})',

        # 复杂引用格式
        r'(~?\\citeurl{([^}]+)})',
        r'(~?\\citedate{([^}]+)})',
        r'(~?\\citetitle{([^}]+)})',
        r'(~?\\citepublisher{([^}]+)})',

        # 特殊字符处理
        r'(~?\\cite@{([^}]+)})',
        r'(~?\\cite!{([^}]+)})',
        r'(~?\\cite\${([^}]+)})',
    ]

    # citations_dict = OrderedDict()
    citations_info = []
    combined_pattern = '|'.join(f'(?:{pattern})' for pattern in patterns)
    matches = re.finditer(combined_pattern, latex_text)

    for match in matches:
        full_citation = match.group(0)
        groups = match.groups()
        keys = next((group for group in reversed(groups) if group is not None), '')

        is_multi_citation = ',' in keys

        for key in keys.split(','):
            key = key.strip()
            citations_info.append((key, full_citation, is_multi_citation))
            # if key not in citations_dict:
            #     citations_dict[key] = (full_citation, is_multi_citation)
            # elif len(full_citation) > len(citations_dict[key][0]):
            #     citations_dict[key] = (full_citation, citations_dict[key][1] or is_multi_citation)

    return citations_info


def extract_citation_keys_bk(latex_text):
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



