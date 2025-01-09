import re
import os
from tqdm import tqdm


def extract_main_content(latex_text):
    # 1. Try to match conclusion section - include conclusion section but exclude everything after
    conclusion_patterns = [
        r'\\section\*?{conclusion',
        r'\\section\*?{conclusions',
        r'\\section\*?{summary',
        r'\\section\*?{concluding\s+remarks',
        r'\\section\*?{concluding\s+discussion',
        r'\\section\*?{final\s+remarks',
        r'\\section\*?{concluding\s+section',
        r'\\section\*?{discussion\s+and\s+conclusion',
        r'\\section\*?{conclusion\s+and\s+discussion',
        r'\\section\*?{conclusions\s+and\s+future\s+work',
        r'\\section\*?{summary\s+and\s+conclusions'
    ]

    for pattern in conclusion_patterns:
        match = re.search(pattern, latex_text, re.IGNORECASE)
        if match:
            # Find the start of the next section after conclusion
            next_section = re.search(r'\\section', latex_text[match.end():])
            if next_section:
                # Return everything up to the next section after conclusion
                return latex_text[:match.end() + next_section.start()]
            else:
                # If no next section found, return everything up to potential bibliography/acknowledgment
                temp_text = latex_text[:]
                # Check for bibliography or acknowledgment
                end_patterns = [
                    r'\\begin{thebibliography}',
                    r'\\bibliographystyle{',
                    r'\\bibliography{',
                    r'\\section\*?{references}',
                    r'\\section\*?{bibliography}',
                    r'\\section\*?{acknowledge?ments?}',  # matches Acknowledgment(s), Acknowledgement(s)
                    r'\\section\*?{acknowledge?ment\s+and',  # matches "Acknowledgement and..."
                    r'\\paragraph\*?{acknowledge?ments?}',
                    r'\\paragraph\*?{acknowledge?ment\s+and',
                    r'\\subsection\*?{acknowledge?ments?}',
                    r'\\subsection\*?{acknowledge?ment\s+and',
                    r'\\textbf{acknowledgement}'
                    r'\\appendix',
                    r'\\appendices',
                    r'\\section\*?{appendix',
                    r'\\section\*?{appendices',
                    r'\\chapter\*?{appendix',
                    r'\\chapter\*?{appendices',
                    r'\\begin{appendix}',
                    r'\\begin{appendices}'
                ]
                for end_pattern in end_patterns:
                    end_match = re.search(end_pattern, temp_text[match.end():], re.IGNORECASE)
                    if end_match:
                        return latex_text[:match.end() + end_match.start()]

                # If no end markers found, return up to the end
                return temp_text

    # 2. Try to match acknowledgment section
    ack_patterns = [
        r'\\section\*?{acknowledge?ments?}',  # matches Acknowledgment(s), Acknowledgement(s)
        r'\\section\*?{acknowledge?ment\s+and',  # matches "Acknowledgement and..."
        r'\\paragraph\*?{acknowledge?ments?}',
        r'\\paragraph\*?{acknowledge?ment\s+and',
        r'\\subsection\*?{acknowledge?ments?}',
        r'\\subsection\*?{acknowledge?ment\s+and',
    ]

    for pattern in ack_patterns:
        match = re.search(pattern, latex_text, re.IGNORECASE)
        if match:
            return latex_text[:match.start()]

    # 3. Try to match bibliography
    bib_patterns = [
        r'\\begin{thebibliography}',
        r'\\bibliographystyle{',
        r'\\bibliography{',
        r'\\section\*?{references}',
        r'\\section\*?{bibliography}'
    ]

    for pattern in bib_patterns:
        match = re.search(pattern, latex_text, re.IGNORECASE)
        if match:
            temp_text = latex_text[:match.start()]
            break
    else:
        temp_text = latex_text

    # 4. Always remove appendix sections
    appendix_patterns = [
        r'\\appendix',
        r'\\appendices',
        r'\\section\*?{appendix',
        r'\\section\*?{appendices',
        r'\\chapter\*?{appendix',
        r'\\chapter\*?{appendices',
        r'\\begin{appendix}',
        r'\\begin{appendices}'
    ]

    for pattern in appendix_patterns:
        match = re.search(pattern, temp_text, re.IGNORECASE)
        if match:
            temp_text = temp_text[:match.start()]
            break

    return temp_text


def single_process(full_tex_path, paper_dir_path):
    main_tex_file_path = os.path.join(paper_dir_path, "main_tex.tex")
    # if os.path.exists(main_tex_file_path):
    #     return
    with open(full_tex_path, "r") as fi:
        full_tex = fi.read()
        main_tex = extract_main_content(full_tex)
    with open(main_tex_file_path, "w") as fo:
        fo.write(main_tex)


def run_on_darth_server(input_dir):
    total = 0
    full_tex_file_not_exist_count = 0
    for sub_dir in os.listdir(input_dir):
        print("Processing", sub_dir)
        if os.path.isdir(os.path.join(input_dir, sub_dir)):
            for paper_dir in tqdm(os.listdir(os.path.join(input_dir, sub_dir))):
                if not paper_dir.startswith(sub_dir):
                    print("skip", paper_dir)
                    continue
                total += 1
                paper_dir_path = os.path.join(input_dir, sub_dir, paper_dir)
                full_tex_path = os.path.join(paper_dir_path, "full.tex")
                if not os.path.exists(full_tex_path):
                    full_tex_file_not_exist_count += 1
                    continue
                single_process(full_tex_path, paper_dir_path)
    print("Total papers processed:", total)
    print("full_tex_file_not_exist_count:", full_tex_file_not_exist_count)
    return


if __name__ == "__main__":
    run_on_darth_server("/data/yubowang/arxiv_plain_latex_data_1028")
