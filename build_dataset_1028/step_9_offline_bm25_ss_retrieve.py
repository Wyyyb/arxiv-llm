import json
import numpy as np
from typing import Dict, Tuple, List
from tqdm import tqdm
import pickle
from collections import defaultdict
import math
from concurrent.futures import ProcessPoolExecutor
import os
import mmh3  # MurmurHash3 for faster hashing
import copy


class DiskBM25Index:
    def __init__(self, index_dir: str):
        """
        初始化索引

        Args:
            index_dir: 索引文件存储目录
        """
        self.index_dir = index_dir
        self.index_path = os.path.join(index_dir, "bm25_index.pkl")
        self.docs_path = os.path.join(index_dir, "docs.pkl")
        self.stats_path = os.path.join(index_dir, "stats.pkl")

        # BM25参数
        self.k1 = 1.5
        self.b = 0.75

    def build_index(self, file_path: str, batch_size: int = 100000):
        """
        构建索引并存储到磁盘

        Args:
            file_path: jsonl文件路径
            batch_size: 每批处理的文档数
        """
        os.makedirs(self.index_dir, exist_ok=True)

        # 第一遍遍历：计算文档数量和平均长度
        doc_count = 0
        total_len = 0

        print("第一遍遍历：计算基础统计信息...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                doc = json.loads(line.strip())
                tokens = doc[1].lower().split()
                doc_count += 1
                total_len += len(tokens)

        avgdl = total_len / doc_count

        # 第二遍遍历：构建倒排索引
        print("第二遍遍历：构建倒排索引...")

        # 使用defaultdict避免频繁的键检查
        index = defaultdict(list)  # {term: [(doc_id, tf), ...]}
        doc_lens = {}  # {doc_id: doc_length}
        docs = {}  # {doc_id: document}

        with open(file_path, 'r', encoding='utf-8') as f:
            for doc_id, line in enumerate(tqdm(f)):
                doc = json.loads(line.strip())
                tokens = doc[1].lower().split()

                # 计算词频
                term_freq = defaultdict(int)
                for token in tokens:
                    term_freq[token] += 1

                # 更新索引
                for token, freq in term_freq.items():
                    index[token].append((doc_id, freq))

                doc_lens[doc_id] = len(tokens)
                docs[doc_id] = doc

        # 保存索引和统计信息
        print("保存索引到磁盘...")
        stats = {
            'doc_count': doc_count,
            'avgdl': avgdl,
            'doc_lens': doc_lens
        }

        with open(self.index_path, 'wb') as f:
            pickle.dump(dict(index), f)

        with open(self.docs_path, 'wb') as f:
            pickle.dump(docs, f)

        with open(self.stats_path, 'wb') as f:
            pickle.dump(stats, f)

    def load_index(self):
        """加载索引"""
        with open(self.index_path, 'rb') as f:
            self.index = pickle.load(f)

        with open(self.stats_path, 'rb') as f:
            stats = pickle.load(f)
            self.doc_count = stats['doc_count']
            self.avgdl = stats['avgdl']
            self.doc_lens = stats['doc_lens']

    def _score_one(self, query_tokens: List[str], doc_id: int) -> float:
        """计算单个文档的BM25分数"""
        score = 0.0
        doc_len = self.doc_lens[doc_id]

        for token in query_tokens:
            if token not in self.index:
                continue

            # 获取包含该词的文档列表
            docs_with_term = self.index[token]
            df = len(docs_with_term)
            idf = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)

            # 查找当前文档的词频
            tf = 0
            for d_id, freq in docs_with_term:
                if d_id == doc_id:
                    tf = freq
                    break

            # BM25计算公式
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * numerator / denominator

        return score

    def search(self, query: str, top_k: int = 1) -> List[Tuple[Dict, float]]:
        """
        搜索最相似的文档

        Args:
            query: 查询字符串
            top_k: 返回的结果数量

        Returns:
            [(文档, 分数), ...]
        """
        query_tokens = query.lower().split()

        # 获取包含查询词的所有文档ID
        candidate_docs = set()
        for token in query_tokens:
            if token in self.index:
                candidate_docs.update(doc_id for doc_id, _ in self.index[token])

        # 计算候选文档的分数
        scores = []
        for doc_id in candidate_docs:
            score = self._score_one(query_tokens, doc_id)
            scores.append((doc_id, score))

        # 排序并获取top-k结果
        scores.sort(key=lambda x: x[1], reverse=True)
        top_results = scores[:top_k]

        # 加载文档内容
        with open(self.docs_path, 'rb') as f:
            docs = pickle.load(f)
            results = [(docs[doc_id], score) for doc_id, score in top_results]

        return results


def load_documents(file_path):
    with open(file_path, 'r') as fi:
        documents = json.load(fi)
    documents = {k.strip(): v for k, v in documents.items()}
    return documents


def main():
    os.makedirs("../local_darth_1110", exist_ok=True)
    os.makedirs("../local_darth_1110/index_directory", exist_ok=True)
    index = DiskBM25Index("../local_darth_1110/index_directory")
    index.build_index("/data/yubowang/ss_offline_data/ss_offline_data_1109.jsonl")
    index = DiskBM25Index("../local_darth_1110/index_directory")
    index.load_index()
    document_path = "../local_1031/offline_query_ss_1110.json"
    documents = load_documents(document_path)
    success_count = 0
    res = copy.deepcopy(documents)
    for k, v in tqdm(documents.items()):
        if v is not None:
            res[k] = v
            continue
        query = k
        results = index.search(query, top_k=1)
        best_match, score = results[0]
        curr_res = {"corpus_id": best_match[0], "ss_title": best_match[1], "abstract": best_match[2],
                    "bm25_score": score}
        res[k] = curr_res
        success_count += 1
        if success_count % 100 == 0:
            print("query: ", query)
            print("result: ", curr_res)
            with open(document_path, "w") as fo:
                fo.write(json.dumps(res))


main()
# 使用示例

# 1. 首次建立索引
"""
index = DiskBM25Index("index_directory")
index.build_index("ss_offline_data_1109.jsonl")
"""

# 2. 搜索示例
"""
# 初始化索引
index = DiskBM25Index("index_directory")
index.load_index()

# 单次查询
query = "Attention is all you need"
results = index.search(query, top_k=1)
best_match, score = results[0]
print(f"最相似论文: {best_match}")
print(f"BM25分数: {score}")

# 批量查询
queries = ["query1", "query2", "query3", ...]
for query in tqdm(queries):
    results = index.search(query)
    # 处理结果
"""