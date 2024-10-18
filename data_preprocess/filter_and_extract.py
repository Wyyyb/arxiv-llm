import os
import tarfile
import gzip
import json
import shutil
import io


def load_meta_data(meta_data_dir):
    arxiv_ids = {}
    for file in os.listdir(meta_data_dir):
        if not file.endswith('.json'):
            continue
        file_path = os.path.join(meta_data_dir, file)
        print("loading", file_path)
        with open(file_path, "r") as fi:
            for line in fi.readlines():
                curr = json.loads(line)
                if "physics" not in curr["categories"] and "/" not in curr["id"]:
                    arxiv_ids[curr["id"]] = "None"
                elif "cs/" in curr["id"] and "physics" not in curr["id"]:
                    arxiv_ids[curr["id"]] = "None"
    return arxiv_ids


def load_cache(cache_file):
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)
    return {}


def save_cache(cache_file, cache):
    with open(cache_file, 'w') as f:
        json.dump(cache, f, indent=2)


def process_archives(input_dir, latex_dir, meta_dir="meta_data/", cache_file="cache.json"):
    # arxiv_ids = load_meta_data(meta_dir)
    meta_records_path = "meta_records_1014.json"
    if os.path.exists(meta_records_path):
        with open(meta_records_path, "r") as fi:
            arxiv_ids = json.load(fi)
    else:
        arxiv_ids = load_meta_data(meta_dir)
    cache = load_cache(cache_file)

    for filename in os.listdir(input_dir):
        if filename.endswith('.tar'):
            tar_path = os.path.join(input_dir, filename)
            print(f"Processing tar file: {tar_path}")

            if tar_path in cache and cache[tar_path] == "processed":
                print(f"Skipping {tar_path} as it has already been processed.")
                continue

            with tarfile.open(tar_path, 'r') as tar:
                for member in tar.getmembers():
                    if not member.name.endswith('.gz'):
                        continue

                    arxiv_id = os.path.splitext(os.path.basename(member.name))[0]
                    # print(f"Processing arxiv_id: {arxiv_id}")

                    if arxiv_id not in arxiv_ids:
                        # print(f"Skipping {arxiv_id} as it's not in arxiv_ids")
                        continue
                    print(f"Find {arxiv_id} is in arxiv_ids")
                    gz_file = tar.extractfile(member)
                    if gz_file is None:
                        print(f"Failed to extract {member.name}")
                        continue

                    sub_dir = os.path.dirname(member.name)
                    folder_path = os.path.join(latex_dir, sub_dir, arxiv_id)
                    os.makedirs(folder_path, exist_ok=True)
                    print(f"Created folder: {folder_path}")

                    try:
                        with gzip.open(gz_file, 'rb') as f_in:
                            content = f_in.read()
                            print(f"Decompressed content size: {len(content)} bytes")

                            if tarfile.is_tarfile(io.BytesIO(content)):
                                print("Content is a tar file, extracting...")
                                with tarfile.open(fileobj=io.BytesIO(content)) as inner_tar:
                                    inner_tar.extractall(path=folder_path)
                                print(f"Extracted tar content to {folder_path}")
                            else:
                                print("Content is not a tar file, writing directly...")
                                with open(os.path.join(folder_path, arxiv_id), 'wb') as f_out:
                                    f_out.write(content)
                                print(f"Wrote content to {os.path.join(folder_path, arxiv_id)}")

                        arxiv_ids[arxiv_id] = "True"
                        print(f"Successfully processed {arxiv_id}")
                    except Exception as e:
                        print(f"Error processing {arxiv_id}: {str(e)}")

            cache[tar_path] = "processed"
            save_cache(cache_file, cache)

    with open(meta_records_path, "w") as fo:
        json.dump(arxiv_ids, fo, indent=2)

    count = sum(1 for v in arxiv_ids.values() if v == "True")
    total_count = sum(1 for v in arxiv_ids.values() if v in ["True", "None"])
    print(f"Number of processed arxiv papers: {count}")
    print(f"Number of total arxiv papers: {total_count}")


# Example usage
# process_archives('/data/yubowang/arxiv-sample', '/data/yubowang/arxiv-sample-latex_dir')
process_archives('/data/xueguang/arxiv-latex', '/data/yubowang/arxiv-latex-filtered_1014')

