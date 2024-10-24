import os
import chardet
import re


def find_main_tex_file(directory):
    res = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.tex'):
                file_path = os.path.join(root, file)
                content = read_file_safely(file_path)
                if '\\begin{document}' in content:
                    res.append(os.path.join(root, file))
    return res


def read_file_safely(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            res = f.read()
            # print("successfully read file by utf-8")
            return res
    except UnicodeDecodeError:
        with open(file_path, 'rb') as f:
            raw_data = f.read()
        detected = chardet.detect(raw_data)
        encoding = detected['encoding']

        try:
            return raw_data.decode(encoding)
        except Exception as e:
            print("Error decoding", e)
            return raw_data.decode(encoding, errors='replace')


def handle_input_commands_bk(content, base_dir, depth=0, max_depth=10):
    def replace_input(match):
        if depth >= max_depth:
            print(f"Warning: Maximum recursion depth ({max_depth}) reached. Stopping recursion.")
            return match.group(0)

        input_file = match.group(1)
        input_path = os.path.join(base_dir, input_file)
        if not input_path.endswith('.tex'):
            input_path += '.tex'

        if os.path.exists(input_path):
            file_content = read_file_safely(input_path)
            return handle_input_commands_bk(file_content, os.path.dirname(input_path), depth + 1, max_depth)
        else:
            return match.group(0)

    content = re.sub(r'\\input{(.+?)}', replace_input, content)
    return content


def handle_input_commands(content, base_dir, depth=0, max_depth=10):
    def replace_input(match):
        if depth >= max_depth:
            print(f"Warning: Maximum recursion depth ({max_depth}) reached. Stopping recursion.")
            return match.group(0)

        input_file = match.group(1).strip()
        # 处理可能的路径分隔符
        input_file = input_file.replace('/', os.sep).replace('\\', os.sep)

        # 尝试多个可能的文件扩展名和路径组合
        possible_paths = [
            os.path.join(base_dir, input_file),
            os.path.join(base_dir, input_file + '.tex'),
            os.path.join(base_dir, input_file + '.ltx'),
            os.path.join(base_dir, input_file + '.latex')
        ]

        # 如果输入的是绝对路径，也添加到可能的路径列表中
        if os.path.isabs(input_file):
            possible_paths.extend([
                input_file,
                input_file + '.tex',
                input_file + '.ltx',
                input_file + '.latex'
            ])

        # 尝试所有可能的路径
        for input_path in possible_paths:
            if os.path.exists(input_path) and os.path.isfile(input_path):
                try:
                    file_content = read_file_safely(input_path)
                    # 递归处理新文件中的 \input 命令
                    processed_content = handle_input_commands(
                        file_content,
                        os.path.dirname(input_path),
                        depth + 1,
                        max_depth
                    )
                    return processed_content
                except Exception as e:
                    print(f"Warning: Error processing {input_path}: {str(e)}")
                    continue

        print(f"Warning: Could not find input file: {input_file}")
        return match.group(0)

    # 处理标准的 \input{} 命令
    content = re.sub(r'\\input\s*{(.+?)}', replace_input, content)

    # 处理不带花括号的 \input 命令
    content = re.sub(r'\\input\s+([^\s{}\n]+)', replace_input, content)

    # 处理 \include{} 命令（类似于 \input）
    content = re.sub(r'\\include\s*{(.+?)}', replace_input, content)

    return content





