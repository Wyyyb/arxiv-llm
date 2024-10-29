import re
import os
import json


valid_step_3_count = 0


def extract_citations(text):
    # Dictionary to store citations
    cite_dict = {}

    # Counter for all citations
    counter = 1

    # Patterns for different citation commands
    cite_patterns = [
        r'(?:~|\s)*\\cite\{([^}]+)\}',
        r'(?:~|\s)*\\citep\{([^}]+)\}',
        r'(?:~|\s)*\\citet\{([^}]+)\}',
        r'(?:~|\s)*\\citeyear\{([^}]+)\}',
        r'(?:~|\s)*\\citeauthor\{([^}]+)\}',
        r'(?:~|\s)*\\citealt\{([^}]+)\}',
        r'(?:~|\s)*\\citealp\{([^}]+)\}',
        r'(?:~|\s)*\\citenum\{([^}]+)\}'
    ]

    def replace_match(match):
        nonlocal counter
        keys = match.group(1).split(',')

        if len(keys) == 1:
            # Single citation
            placeholder = f"<|cite_{counter}|>"
            cite_dict[placeholder] = keys[0].strip()
            counter += 1
            return placeholder
        else:
            # Multiple citations
            placeholders = []
            for i, key in enumerate(keys, 1):
                placeholder = f"<|multi_cite_{counter}_{i}|>"
                cite_dict[placeholder] = key.strip()
                placeholders.append(placeholder)
            counter += 1
            return ''.join(placeholders)

    # Process text for each citation pattern
    processed_text = text
    for pattern in cite_patterns:
        processed_text = re.sub(pattern, replace_match, processed_text)

    return processed_text, cite_dict


def extract_bibitem_key(bibitem):
    import re

    # Pattern to match citation key in bibitem
    # Handles both \bibitem{key} and \bibitem[...]{key} formats
    pattern = r'\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}'

    match = re.search(pattern, bibitem)
    if match:
        return match.group(1)
    print("----------Failed to extract bibitem key", bibitem)
    return None


def extract_bib_item(bib_item):
    # Pattern to match the citation key
    key_pattern = r'@\w+\s*\{([^,]+),'

    # Pattern to match the title
    # Handles both title = {...} and title = "..." formats
    # Also handles multi-line titles
    title_pattern = r'title\s*=\s*[{\"]([^}\"]+)[}\"]'

    # Extract key
    key_match = re.search(key_pattern, bib_item)
    key = key_match.group(1).strip() if key_match else None

    # Extract title
    title_match = re.search(title_pattern, bib_item, re.IGNORECASE)
    title = title_match.group(1).strip() if title_match else None

    return key, title


def collect_bib_info(paper_dir_path):
    global valid_step_3_count
    step_2_res_path = os.path.join(paper_dir_path, "step_2_info.json")
    if not os.path.exists(step_2_res_path):
        return None
    with open(step_2_res_path, "r") as fi:
        curr = json.load(fi)
    arxiv_id = curr["arxiv_id"]
    intro = "Introduction\n" + curr["intro"]
    related_work = curr["related_work"]
    if related_work and related_work != "":
        intro = intro + "\nRelated Work\n" + related_work
    intro, citations = extract_citations(intro)
    bib_info = {}
    for each in curr["bib_items"]:
        citation_key, title = extract_bib_item(each)









