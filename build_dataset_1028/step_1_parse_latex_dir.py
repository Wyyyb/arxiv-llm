import os
import re
import shutil
from typing import Dict, List, Set
from pathlib import Path
from tqdm import tqdm


class LatexProcessor:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = os.path.abspath(input_dir)
        self.output_dir = os.path.abspath(output_dir)
        self.processed_files: Set[str] = set()

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def read_file(self, file_path: str) -> str:
        """Read a file with different encodings"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception as e:
                print(f"Warning: Could not read {file_path}", e)
                return ""

    def write_file(self, file_path: str, content: str) -> None:
        """Write content to file"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

    def clean_text(self, text: str) -> str:
        """Remove comments and normalize newlines"""
        # Remove line comments (not preceded by backslash)
        text = re.sub(r'(?<!\\)%.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'\\begin{comment}.*?\\end{comment}', '', text, flags=re.DOTALL)
        text = re.sub(r'\\iffalse.*?\\fi', '', text, flags=re.DOTALL)
        text = re.sub(r'\\if0.*?\\fi', '', text, flags=re.DOTALL)
        text = re.sub(r'\\ifdef{[^}]*}.*?\\fi', '', text, flags=re.DOTALL)
        text = re.sub(r'\\ifndef{[^}]*}.*?\\fi', '', text, flags=re.DOTALL)
        # Normalize multiple newlines to just two
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text

    def process_includes(self, text: str, current_file_path: str) -> str:
        """Process all include/input commands in the text"""
        current_abs_path = os.path.abspath(current_file_path)
        if current_abs_path in self.processed_files:
            print(f"Warning: Circular inclusion detected for {current_file_path}")
            return text

        self.processed_files.add(current_abs_path)
        current_dir = os.path.dirname(current_abs_path)

        # Find all \input and \include commands
        pattern = r'\\(?:input|include){([^}]+)}'
        matches = list(re.finditer(pattern, text))

        for match in matches:
            include_file = match.group(1)
            # Add .tex extension if not present
            if not include_file.endswith('.tex'):
                include_file += '.tex'

            include_path = os.path.join(current_dir, include_file)

            if os.path.exists(include_path):
                include_content = self.read_file(include_path)
                include_content = self.process_includes(include_content, include_path)
                text = text.replace(match.group(0), include_content)
            else:
                print(f"Warning: Could not find included file {include_path}")

        return text

    def find_main_files(self) -> List[str]:
        """Find all tex files containing \documentclass"""
        main_files = []
        for file in os.listdir(self.input_dir):
            if file.endswith('.tex'):
                file_path = os.path.join(self.input_dir, file)
                content = self.read_file(file_path)
                if '\\documentclass' in content:
                    main_files.append(file)
        return main_files

    def process_single_file(self, file_name: str) -> str:
        """Process a single tex file"""
        self.processed_files.clear()
        file_path = os.path.join(self.input_dir, file_name)
        content = self.read_file(file_path)
        content = self.process_includes(content, file_path)
        content = self.clean_text(content)
        return content

    def merge_files_by_extension(self, extension: str, output_filename: str) -> None:
        """Merge all files with given extension into one file"""
        merged_content = []

        # Recursively find all files with the given extension
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith(extension):
                    file_path = os.path.join(root, file)
                    content = self.read_file(file_path)
                    if content:
                        merged_content.append(f"% From file: {file}")
                        merged_content.append(content)
                        merged_content.append("\n")

        if merged_content:
            output_path = os.path.join(self.output_dir, output_filename)
            self.write_file(output_path, "\n".join(merged_content))
            print(f"Merged {extension} files into {output_path}")
        else:
            print(f"No {extension} files found to merge")

    def process_folder(self) -> None:
        """Main function to process the latex folder"""
        # Find all main tex files
        main_files = self.find_main_files()
        if not main_files:
            raise Exception("Could not find any tex file with \\documentclass")

        # Process each file
        file_contents: Dict[str, str] = {}
        for file in main_files:
            processed_content = self.process_single_file(file)
            file_contents[file] = processed_content

        # Find the longest content
        longest_file = max(file_contents.keys(), key=lambda x: len(file_contents[x]))

        # Write the processed content to output directory
        output_tex_path = os.path.join(self.output_dir, "full.tex")
        self.write_file(output_tex_path, file_contents[longest_file])

        # Merge bibliography files
        self.merge_files_by_extension(".bib", "reference.bib")
        self.merge_files_by_extension(".bbl", "reference.bbl")


def process_latex_project(input_dir: str, output_dir: str) -> None:
    """Wrapper function for easy use"""
    processor = LatexProcessor(input_dir, output_dir)
    processor.process_folder()


def unit_test():
    folder_path = '../local_1028/test_data_1028/latex_dir_1'
    output_path = "../local_1028/test_data_1028/step_1_output"
    process_latex_project(folder_path, output_path)


def run_on_darth_server(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for sub_dir in os.listdir(input_dir):
        print("Processing", sub_dir)
        if os.path.isdir(os.path.join(input_dir, sub_dir)):
            os.makedirs(os.path.join(output_dir, sub_dir), exist_ok=True)
            for paper_dir in tqdm(os.listdir(os.path.join(input_dir, sub_dir))):
                if not paper_dir.startswith(sub_dir):
                    print("skip", paper_dir)
                    continue
                paper_dir_path = os.path.join(input_dir, sub_dir, paper_dir)
                target_dir_path = os.path.join(output_dir, sub_dir, paper_dir)
                os.makedirs(target_dir_path, exist_ok=True)
                process_latex_project(paper_dir_path, target_dir_path)


# unit_test()
if __name__ == "__main__":
    input_dir_path = "/data/yubowang/arxiv-latex-filtered_1014"
    output_dir_path = "/data/yubowang/arxiv_plain_latex_data_1028"

