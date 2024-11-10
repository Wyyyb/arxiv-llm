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
import re


def normalize_text(text: str) -> str:
    """
    将文本转小写，去除特殊字符和多余空白
    """
    # 转小写
    text = text.lower()
    # 替换特殊字符为空格
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # 将多个空白字符替换为单个空格
    text = re.sub(r'\s+', ' ', text)
    # 去除首尾空白
    text = text.strip()
    return text


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
        self.normalized_titles_path = os.path.join(index_dir, "normalized_titles.pkl")

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
        normalized_titles = {}  # 存储规范化后的标题

        print("第一遍遍历：计算基础统计信息...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                doc = json.loads(line.strip())
                tokens = doc[1].lower().split()
                doc_count += 1
                total_len += len(tokens)
                # 存储规范化后的标题
                normalized_titles[doc_count - 1] = normalize_text(doc[1])

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

        with open(self.normalized_titles_path, 'wb') as f:
            pickle.dump(normalized_titles, f)

    def load_index(self):
        print("loading index...")
        with open(self.index_path, 'rb') as f:
            self.index = pickle.load(f)

        with open(self.stats_path, 'rb') as f:
            stats = pickle.load(f)
            self.doc_count = stats['doc_count']
            self.avgdl = stats['avgdl']
            self.doc_lens = stats['doc_lens']

        with open(self.normalized_titles_path, 'rb') as f:
            self.normalized_titles = pickle.load(f)
        print("index loaded!")

    def _score_one(self, query_tokens: List[str], doc_id: int) -> float:
        score = 0.0
        doc_len = self.doc_lens[doc_id]

        for token in query_tokens:
            if token not in self.index:
                continue

            docs_with_term = self.index[token]
            df = len(docs_with_term)
            idf = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)

            tf = 0
            for d_id, freq in docs_with_term:
                if d_id == doc_id:
                    tf = freq
                    break

            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * numerator / denominator

        return score

    def search_best(self, query: str):
        """
        搜索最相似的文档

        Args:
            query: 查询字符串

        Returns:
            (最相似文档, 分数)
        """
        # 首先进行精确匹配
        normalized_query = normalize_text(query)
        for doc_id, title in self.normalized_titles.items():
            if normalized_query == title:
                with open(self.docs_path, 'rb') as f:
                    docs = pickle.load(f)
                    doc = docs[doc_id]
                return doc, float('inf')  # 返回一个极大值表示精确匹配

        # 如果没有精确匹配，进行BM25搜索
        query_tokens = normalized_query.split()

        # 获取候选文档ID
        token_docs = defaultdict(set)
        for token in query_tokens:
            if token in self.index:
                for doc_id, _ in self.index[token]:
                    token_docs[token].add(doc_id)

        # 只选择至少包含5个查询词的文档作为候选
        min_tokens_required = min(5, len(query_tokens))  # 如果查询词少于5个，则使用查询词的数量
        candidate_docs = set()
        doc_counts = defaultdict(int)

        for token, docs in token_docs.items():
            for doc_id in docs:
                doc_counts[doc_id] += 1

        for doc_id, count in doc_counts.items():
            if count >= min_tokens_required:
                candidate_docs.add(doc_id)

        print("number of candidates", len(candidate_docs))

        if not candidate_docs:
            return None, 0.0

        # 仅追踪最高分数的文档
        best_score = float('-inf')
        best_doc_id = None

        # 计算每个候选文档的分数
        for doc_id in candidate_docs:
            score = self._score_one(query_tokens, doc_id)
            if score > best_score:
                best_score = score
                best_doc_id = doc_id

        # 如果没有找到匹配的文档
        if best_doc_id is None:
            return None, 0.0

        # 只加载最佳匹配文档的内容
        with open(self.docs_path, 'rb') as f:
            docs = pickle.load(f)
            best_doc = docs[best_doc_id]

        return best_doc, best_score


def load_documents(file_path):
    with open(file_path, 'r') as fi:
        documents = json.load(fi)
    documents = {k.strip(): v for k, v in documents.items()}
    return documents


def main():
    os.makedirs("../local_darth_1110", exist_ok=True)
    os.makedirs("../local_darth_1110/index_directory", exist_ok=True)
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
        results = index.search_best(query)
        best_match, score = results
        if best_match is None:
            res[k] = v
            continue
        curr_res = {"corpus_id": best_match[0], "ss_title": best_match[1], "abstract": best_match[2],
                    "bm25_score": score}
        res[k] = curr_res
        success_count += 1
        if success_count % 10 == 0:
            print("query: ", query)
            print("result: ", curr_res)
            with open(document_path, "w") as fo:
                fo.write(json.dumps(res))


if __name__ == "__main__":
    main()

