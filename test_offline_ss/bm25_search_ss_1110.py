import json
import multiprocessing as mp
from pathlib import Path
from typing import List, Tuple, Optional
import os
import hashlib
from tqdm import tqdm
import re
import copy
from rank_bm25 import BM25Okapi
import pickle
import numpy as np
from concurrent.futures import ProcessPoolExecutor


def normalize_title(text: str) -> str:
    """
    将文本转小写，去除特殊字符和多余空白
    """
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class PaperSearcher:
    def __init__(self, jsonl_path: str, index_dir: str = "paper_index"):
        self.jsonl_path = jsonl_path
        self.index_dir = index_dir
        self.bm25 = None
        self.title_to_doc = {}  # 用于精确匹配的标题索引
        self.corpus = []  # 存储所有文档的标题
        self.doc_info = []  # 存储所有文档的完整信息

        # 检查索引是否需要更新
        self._init_index()

    def _get_file_hash(self) -> str:
        """获取数据文件的MD5哈希值"""
        hash_md5 = hashlib.md5()
        with open(self.jsonl_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _init_index(self):
        """初始化索引，如果需要则创建新索引"""
        os.makedirs(self.index_dir, exist_ok=True)

        bm25_file = os.path.join(self.index_dir, "bm25.pkl")
        corpus_file = os.path.join(self.index_dir, "corpus.pkl")
        doc_info_file = os.path.join(self.index_dir, "doc_info.pkl")
        exact_match_file = os.path.join(self.index_dir, "exact_match.json")
        hash_file = os.path.join(self.index_dir, "data_hash.txt")

        # current_hash = self._get_file_hash()
        need_update = True

        # if all(os.path.exists(f) for f in [bm25_file, corpus_file, doc_info_file, exact_match_file, hash_file]):
        #     with open(hash_file, 'r') as f:
        #         stored_hash = f.read().strip()
        #         if stored_hash == current_hash:
        #             need_update = False

        if need_update:
            print("Creating new index...")
            self.create_index()
            # with open(hash_file, 'w') as f:
            #     f.write(current_hash)
        else:
            print("Loading existing index...")
            with open(bm25_file, 'rb') as f:
                self.bm25 = pickle.load(f)
            with open(corpus_file, 'rb') as f:
                self.corpus = pickle.load(f)
            with open(doc_info_file, 'rb') as f:
                self.doc_info = pickle.load(f)
            with open(exact_match_file, 'r') as f:
                self.title_to_doc = json.load(f)

    def process_chunk(self, chunk: List[str]):
        """处理文档块"""
        local_corpus = []
        local_doc_info = []
        local_title_to_doc = {}

        for line in chunk:
            try:
                paper_id, title, abstract = json.loads(line)
                normalized_title = normalize_title(title)

                local_corpus.append(normalized_title.split())
                local_doc_info.append({
                    'id': paper_id,
                    'title': title,
                    'abstract': abstract
                })

                local_title_to_doc[normalized_title] = {
                    'id': paper_id,
                    'title': title,
                    'abstract': abstract
                }

            except Exception as e:
                print(f"Error processing line: {e}")

        return local_corpus, local_doc_info, local_title_to_doc

    def create_index(self):
        """创建文档索引"""
        print("Reading and indexing documents...")

        chunk_size = 1000000  # 每个进程处理的行数
        current_chunk = []

        self.corpus = []
        self.doc_info = []
        self.title_to_doc = {}

        with open(self.jsonl_path, 'r') as f:
            with ProcessPoolExecutor() as executor:
                for line in tqdm(f, desc="Reading documents"):
                    current_chunk.append(line)
                    if len(current_chunk) == chunk_size:
                        results = executor.submit(self.process_chunk, current_chunk)
                        local_corpus, local_doc_info, local_title_to_doc = results.result()

                        self.corpus.extend(local_corpus)
                        self.doc_info.extend(local_doc_info)
                        self.title_to_doc.update(local_title_to_doc)

                        current_chunk = []

                if current_chunk:
                    results = executor.submit(self.process_chunk, current_chunk)
                    local_corpus, local_doc_info, local_title_to_doc = results.result()

                    self.corpus.extend(local_corpus)
                    self.doc_info.extend(local_doc_info)
                    self.title_to_doc.update(local_title_to_doc)

        print("Creating BM25 index...")
        self.bm25 = BM25Okapi(self.corpus)

        print("Saving index...")
        with open(os.path.join(self.index_dir, "bm25.pkl"), 'wb') as f:
            pickle.dump(self.bm25, f)
        with open(os.path.join(self.index_dir, "corpus.pkl"), 'wb') as f:
            pickle.dump(self.corpus, f)
        with open(os.path.join(self.index_dir, "doc_info.pkl"), 'wb') as f:
            pickle.dump(self.doc_info, f)
        with open(os.path.join(self.index_dir, "exact_match.json"), 'w') as f:
            json.dump(self.title_to_doc, f)

    def search(self, query: str) -> Optional[Tuple[str, str, str, float]]:
        """
        搜索最相似的论文

        Args:
            query: 查询的论文标题

        Returns:
            Optional[Tuple[str, str, str, float]]: 返回 (paper_id, title, abstract, score) 或 None
        """
        query_normalized = normalize_title(query)

        # 精确匹配
        if query_normalized in self.title_to_doc:
            doc = self.title_to_doc[query_normalized]
            return (
                str(doc['id']),  # 确保转换为字符串
                str(doc['title']),
                str(doc['abstract']),
                float(1.0)
            )

        # BM25搜索
        tokenized_query = query_normalized.split()
        scores = self.bm25.get_scores(tokenized_query)
        best_idx = int(np.argmax(scores))  # 确保是整数索引
        best_score = float(scores[best_idx])  # 确保是浮点数

        if best_score > 0:
            doc = self.doc_info[best_idx]
            return (
                str(doc['id']),  # 确保转换为字符串
                str(doc['title']),
                str(doc['abstract']),
                float(best_score)
            )

        return None


def load_query(dir_path):
    query_data = []
    file_path_list = []
    for file in os.listdir(dir_path):
        if not file.endswith("json"):
            continue
        file_path = os.path.join(dir_path, file)
        with open(file_path, "r") as fi:
            curr = json.load(fi)
            query_data.append(curr)
            file_path_list.append(file_path)
    return query_data, file_path_list


def main():
    os.makedirs("../local_darth_1110", exist_ok=True)
    os.makedirs("../local_darth_1110/index_directory", exist_ok=True)
    query_dir_path = "/gpfs/public/research/xy/yubowang/data_trans_1030/ss_data_query_1110/"
    query_data_list, query_data_path_list = load_query(query_dir_path)

    searcher = PaperSearcher(
        jsonl_path='/gpfs/public/research/xy/yubowang/ss_offline_data/ss_offline_data_1109.jsonl',
        index_dir='../local_darth_1110/index_directory/'
    )

    for i, query_data in enumerate(query_data_list):
        success_count = 0
        res = copy.deepcopy(query_data)
        for k, v in tqdm(query_data.items()):
            if v is not None:
                res[k] = v
                continue
            query = k
            result = searcher.search(query)
            if result:
                paper_id, title, abstract, score = result
                res[k] = {"paper_id": paper_id, "title": title,
                          "abstract": abstract, "score": score}
                success_count += 1
                if success_count % 10 == 0:
                    print("query: ", query)
                    print("result: ", res[k])
                    with open(query_data_path_list[i], "w") as fo:
                        fo.write(json.dumps(res))
            else:
                res[k] = None
                continue


if __name__ == "__main__":
    main()