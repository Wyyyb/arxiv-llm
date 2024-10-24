from parse_latex_dir_structure import *


def collect_formal_bib_entries(content):
    # Regular expression to match BibTeX entries
    bib_pattern = r'@\w+\s*\{[^@]*\}'

    # Find all matches in the content
    bib_entries = re.findall(bib_pattern, content, re.DOTALL)

    # Clean up each entry (remove leading/trailing whitespace)
    bib_entries = [entry.strip() for entry in bib_entries]

    return bib_entries


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


def add_bib_info_map(res, bib_info_map=None):
    if not bib_info_map:
        bib_info_map = {}
    for each in res:
        curr = parse_bibtex(each)
        if not curr or "key" not in curr:
            continue
        if curr["key"] not in bib_info_map:
            bib_info_map[curr["key"]] = curr
    return bib_info_map


def find_and_extract_bib(directory, main_tex_content):
    res = []
    bib_entries = collect_formal_bib_entries(main_tex_content)
    if bib_entries:
        res += bib_entries
    bib_info_map = add_bib_info_map(res)

    bib_res = []
    bbl_res = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith('.bib'):
                content = read_file_safely(file_path)
                bib_res += collect_formal_bib_entries(content)
            elif file.endswith('.bbl') or file.endswith('.bblx'):
                content = read_file_safely(file_path)
                bbl_res += collect_bib_item_entries(content)
            else:
                continue
    bib_info_map = add_bib_info_map(bib_res, bib_info_map)
    bib_entries = collect_bib_item_entries(main_tex_content)
    main_bbl_res = []
    if bib_entries:
        main_bbl_res += bib_entries
    bib_info_map = add_bib_info_map(main_bbl_res, bib_info_map)
    bib_info_map = add_bib_info_map(bbl_res, bib_info_map)

    return bib_info_map


def collect_bib_item_entries(bbl_content):
    # entries = []
    # 预处理：移除行尾的 % 注释
    content_cleaned = re.sub(r'%.*$', '', bbl_content, flags=re.MULTILINE)

    # 分离各个bibitem条目
    bibitem_pattern = r'\\bibitem(?:\[(?:[^][]|\[(?:[^][]|\[[^]]*\])*\])*\])?\s*\{([^}]+)\}(.*?)(?=\\bibitem|\Z)'
    bibitem_matches = re.finditer(bibitem_pattern, content_cleaned, re.DOTALL)

    bibtex_entries = []

    for match in bibitem_matches:
        key = match.group(1)
        content = match.group(2).strip()
        original_content = content  # 保存原始内容
        title = None

        # 1. 处理显式标记的标题 (\showarticletitle{...})
        showtitle_match = re.search(r'\\showarticletitle\{([^}]+)\}', content)
        if showtitle_match:
            title = showtitle_match.group(1)

        # 2. 处理在\newblock之后的标题，包括{\em ...}格式
        if not title and '\\newblock' in content:
            blocks = content.split('\\newblock')
            for block in blocks[1:]:
                block = block.strip()
                # 改进的{\em ...}格式处理
                em_match = re.search(r'\{\\em\s*(\{[^{}]*\}|[^{}])*\}', block)
                if em_match:
                    title = em_match.group(0)
                    # 移除最外层的{\em }
                    title = title[5:-1].strip()
                    break
                elif not any(block.lower().startswith(prefix.lower()) for prefix in
                             ('in ', '{\\em in', 'mit', 'springer', 'oxford', 'clarendon')):
                    potential_title = block.split('.')[0].strip()
                    if potential_title and not potential_title.startswith('{\\em'):
                        title = potential_title
                        break

        # 3. 处理引号包围的标题
        if not title:
            quote_match = re.search(r'``([^\']+)\'\'|"([^"]+)"', content)
            if quote_match:
                title = quote_match.group(1) or quote_match.group(2)

        # 4. 处理\emph包围的标题
        if not title:
            emph_match = re.search(r'\\emph\{([^}]+)\}', content)
            if emph_match:
                title = emph_match.group(1)

        # 5. 处理冒号后的标题
        if not title and ':' in content:
            lines = content.split('\n')
            for line in lines:
                if ':' in line:
                    title_part = line.split(':', 1)[1].strip()
                    if '.' in title_part:
                        title_part = title_part.split('.')[0]
                    if len(title_part) > 3 and not any(title_part.lower().startswith(prefix.lower())
                                                       for prefix in ['mit', 'springer', 'oxford', 'clarendon']):
                        title = title_part
                        break

        # 清理标题文本
        if title:
            # 处理嵌套的花括号，保持内容
            prev_title = None
            while '{' in title and '}' in title and title != prev_title:
                prev_title = title
                title = re.sub(r'\{([^{}]*)\}', r'\1', title)
            # 移除LaTeX命令
            title = re.sub(r'\\[a-zA-Z]+', '', title)
            # 规范化空白
            title = ' '.join(title.split())
            # 移除末尾的逗号或句点
            title = title.rstrip('.,')
            # 如果标题以特定词语开头，跳过
            if any(title.lower().startswith(prefix.lower()) for prefix in
                   ['oxford science publications', 'in proceedings', 'mit press',
                    'springer', 'clarendon press']):
                continue

            # 构建BibTeX条目
            entry = f"@misc{{{key},\n"
            entry += f"  title = {{{title}}},\n"
            # 将原始内容作为note字段，替换换行符为空格
            note = original_content
            entry += f"  note = {{{note}}}\n"
            entry += "}\n"
            bibtex_entries.append(entry)

    return bibtex_entries





