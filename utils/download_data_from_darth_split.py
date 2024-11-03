import os
import math
import subprocess
from multiprocessing import Pool
import json
import time
import sys
from datetime import datetime


def format_size(size):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0


def get_file_size(ssh_host, remote_path):
    cmd = f"ssh {ssh_host} 'ls -l {remote_path}'"
    result = subprocess.check_output(cmd, shell=True).decode()
    return int(result.split()[4])


def load_progress(progress_file):
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {}


def save_progress(progress_file, progress):
    with open(progress_file, 'w') as f:
        json.dump(progress, f)


def get_total_downloaded_size(local_dir, chunk_count):
    total_size = 0
    for i in range(chunk_count):
        chunk_file = os.path.join(local_dir, f"chunk_{i}")
        if os.path.exists(chunk_file):
            total_size += os.path.getsize(chunk_file)
    return total_size


class ProgressTracker:
    def __init__(self, total_size):
        self.total_size = total_size
        self.last_size = 0
        self.last_time = time.time()
        self.start_time = time.time()

    def update(self, current_size):
        current_time = time.time()
        time_diff = current_time - self.last_time
        size_diff = current_size - self.last_size

        if time_diff >= 1:  # 每秒更新一次
            speed = size_diff / time_diff
            progress = (current_size / self.total_size) * 100
            elapsed_time = current_time - self.start_time

            if speed > 0:
                eta = (self.total_size - current_size) / speed
            else:
                eta = 0

            self.last_size = current_size
            self.last_time = current_time

            # 清除当前行并更新进度
            sys.stdout.write('\r' + ' ' * 80 + '\r')
            sys.stdout.write(
                f"进度: {progress:.1f}% "
                f"({format_size(current_size)}/{format_size(self.total_size)}) "
                f"速度: {format_size(speed)}/s "
                f"已用时: {int(elapsed_time)}s "
                f"剩余时间: {int(eta)}s"
            )
            sys.stdout.flush()


def download_chunk(args):
    start, end, chunk_id, ssh_host, remote_path, local_dir, progress_file = args
    chunk_file = os.path.join(local_dir, f"chunk_{chunk_id}")

    progress = load_progress(progress_file)
    if str(chunk_id) in progress and progress[str(chunk_id)] == "completed":
        if os.path.exists(chunk_file):
            return chunk_file

    if os.path.exists(chunk_file):
        current_size = os.path.getsize(chunk_file)
        start = start + current_size

    if start < end:
        cmd = (f"ssh {ssh_host} 'dd if={remote_path} bs=1024k skip={start // 1024 // 1024} "
               f"count={math.ceil((end - start) / 1024 / 1024)}' >> {chunk_file}")

        try:
            subprocess.run(cmd, shell=True, check=True)
            progress[str(chunk_id)] = "completed"
            save_progress(progress_file, progress)
        except subprocess.CalledProcessError:
            print(f"\nChunk {chunk_id} download failed, will retry on next run")
            return None

    return chunk_file


def merge_files(chunks, output_file):
    print("\n合并文件中...")
    with open(output_file, 'wb') as outfile:
        for chunk in chunks:
            if chunk and os.path.exists(chunk):
                with open(chunk, 'rb') as infile:
                    outfile.write(infile.read())
                os.remove(chunk)


def main():
    ssh_host = "yubo@darth.cs.uwaterloo.ca"
    remote_path = "/data/yubowang/arxiv-llm/local_1031/train_data_1103.zip"
    local_dir = "local_1103"
    num_processes = 5
    progress_file = os.path.join(local_dir, "download_progress.json")

    os.makedirs(local_dir, exist_ok=True)

    # 获取文件大小
    print("获取文件信息...")
    file_size = get_file_size(ssh_host, remote_path)
    chunk_size = math.ceil(file_size / num_processes)

    # 准备分片参数
    chunks = []
    for i in range(num_processes):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, file_size)
        chunks.append((start, end, i, ssh_host, remote_path, local_dir, progress_file))

    # 创建进度跟踪器
    progress_tracker = ProgressTracker(file_size)

    # 使用进程池下载
    print(f"开始下载文件，使用 {num_processes} 个进程...")
    with Pool(num_processes) as pool:
        chunk_files = []
        for _ in pool.imap_unordered(download_chunk, chunks):
            chunk_files.append(_)
            # 更新总进度
            current_size = get_total_downloaded_size(local_dir, num_processes)
            progress_tracker.update(current_size)

    # 检查是否所有分片都下载成功
    if None in chunk_files:
        print("\n部分分片下载失败，请重新运行脚本继续下载")
        return

    # 合并文件
    output_file = os.path.join(local_dir, "train_data_1103.zip")
    merge_files(chunk_files, output_file)

    # 清理进度文件
    if os.path.exists(progress_file):
        os.remove(progress_file)

    print(f"\n文件已成功下载并合并到: {output_file}")


if __name__ == "__main__":
    main()