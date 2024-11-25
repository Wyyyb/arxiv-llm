import json
import shutil
import glob
import gzip
import os


def split_and_compress_jsonl(input_file, output_dir):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取基础文件名(不含路径和扩展名)
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    # 读取源文件的所有行
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 初始化变量
    current_size = 0
    current_lines = []
    file_index = 0
    temp_files = []

    # 按大小分割文件
    for line in lines:
        line_size = len(line.encode('utf-8'))
        if current_size + line_size > 50 * 1024 * 1024:  # 50MB
            # 写入当前批次的行
            temp_file = os.path.join(output_dir, f"{base_name}_{file_index}.jsonl")
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.writelines(current_lines)
            temp_files.append(temp_file)

            # 重置计数器
            current_lines = [line]
            current_size = line_size
            file_index += 1
        else:
            current_lines.append(line)
            current_size += line_size

    # 写入最后一批数据
    if current_lines:
        temp_file = os.path.join(output_dir, f"{base_name}_{file_index}.jsonl")
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.writelines(current_lines)
        temp_files.append(temp_file)

    # 压缩每个临时文件
    for temp_file in temp_files:
        output_gz = temp_file + '.gz'
        with open(temp_file, 'rb') as f_in:
            with gzip.open(output_gz, 'wb', compresslevel=9) as f_out:
                shutil.copyfileobj(f_in, f_out)

        # 删除未压缩的临时文件
        os.remove(temp_file)

    return len(temp_files)


def merge_compressed_jsonl(input_dir, output_path):
    # 确保输出文件的目录存在
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # 获取目录下所有.gz文件
    gz_files = glob.glob(os.path.join(input_dir, "*.gz"))

    # 按文件名排序，确保按正确顺序合并
    gz_files.sort()

    # 创建输出文件
    with open(output_path, 'w', encoding='utf-8') as outfile:
        # 依次处理每个gz文件
        for gz_file in gz_files:
            with gzip.open(gz_file, 'rt', encoding='utf-8') as infile:
                # 逐行读取并写入
                shutil.copyfileobj(infile, outfile)

                # 确保每个文件之间有换行
                if not infile.peek().endswith(b'\n'):
                    outfile.write('\n')

    return len(gz_files)

# split_and_compress_jsonl('input.jsonl', 'output_directory')
# merge_compressed_jsonl('input_directory', 'output.jsonl')


split_and_compress_jsonl('input.jsonl', 'output_directory')






