import os
import tarfile
import glob


def extract_arxiv_tars(input_dir, output_dir):
    # 创建输出目录(如果不存在)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 查找所有符合条件的tar文件
    tar_pattern = os.path.join(input_dir, "arXiv_src_2410*.tar")
    tar_files = glob.glob(tar_pattern)

    # 遍历并解压每个tar文件
    for tar_file in tar_files:
        try:
            with tarfile.open(tar_file, 'r') as tar:
                # 解压到输出目录
                tar.extractall(path=output_dir)
                print(f"Successfully extracted {tar_file}")
        except Exception as e:
            print(f"Error extracting {tar_file}: {str(e)}")

# 使用示例:
# extract_arxiv_tars("/path/to/input", "/path/to/output")