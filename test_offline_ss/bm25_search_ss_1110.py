import json
import multiprocessing as mp
from pathlib import Path
from typing import List, Tuple, Optional
from pyserini.index import IndexWriter
from pyserini.search import LuceneSearcher
import os
import hashlib
from tqdm import tqdm
import re
import copy


def normalize_title(text: str) -> str:
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


class PaperSearcher:
    def __init__(self, jsonl_path: str, index_dir: str = "paper_index"):
        """
        初始化搜索器

        Args:
            jsonl_path: JSONL文件路径
            index_dir: 索引存储目录
        """
        self.jsonl_path = jsonl_path
        self.index_dir = index_dir
        self.searcher = None
        self.title_to_doc = {}  # 用于精确匹配的标题索引

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
        # 检查索引目录是否存在
        index_exists = os.path.exists(self.index_dir)
        hash_file = os.path.join(self.index_dir, "data_hash.txt")

        current_hash = self._get_file_hash()
        need_update = True

        if index_exists:
            # 检查数据文件是否有更新
            if os.path.exists(hash_file):
                with open(hash_file, 'r') as f:
                    stored_hash = f.read().strip()
                if stored_hash == current_hash:
                    need_update = False

        if need_update:
            print("Creating new index...")
            self.create_index()
            # 保存数据文件哈希值
            os.makedirs(self.index_dir, exist_ok=True)
            with open(hash_file, 'w') as f:
                f.write(current_hash)
        else:
            print("Loading existing index...")

        # 初始化搜索器
        self.searcher = LuceneSearcher(self.index_dir)
        # 加载精确匹配索引
        self._load_exact_match_index()

    def _load_exact_match_index(self):
        """加载精确匹配索引"""
        exact_match_file = os.path.join(self.index_dir, "exact_match.json")
        if os.path.exists(exact_match_file):
            with open(exact_match_file, 'r') as f:
                self.title_to_doc = json.load(f)

    def create_index(self):
        """创建文档索引"""
        # 确保索引目录存在
        os.makedirs(self.index_dir, exist_ok=True)

        # 创建索引写入器
        writer = IndexWriter(self.index_dir)

        # 用于精确匹配的标题索引
        title_to_doc = {}

        print("Reading and indexing documents...")

        # 分块读取大文件
        def process_chunk(chunk: List[str]):
            for line in chunk:
                try:
                    paper_id, title, abstract = json.loads(line)
                    # 构建索引文档
                    doc = {
                        'id': paper_id,
                        'title': title,
                        'abstract': abstract,
                        'contents': title  # 仅对标题建立索引
                    }
                    writer.add_doc(doc)

                    # 添加到精确匹配索引
                    title_to_doc[normalize_title(title)] = {
                        'id': paper_id,
                        'title': title,
                        'abstract': abstract
                    }

                except Exception as e:
                    print(f"Error processing line: {e}")
            return title_to_doc

        # 使用多进程处理数据
        chunk_size = 1000000  # 每个进程处理的行数
        current_chunk = []

        with open(self.jsonl_path, 'r') as f:
            for line in tqdm(f, desc="Processing documents"):
                current_chunk.append(line)
                if len(current_chunk) == chunk_size:
                    title_to_doc.update(process_chunk(current_chunk))
                    current_chunk = []

            # 处理剩余的行
            if current_chunk:
                title_to_doc.update(process_chunk(current_chunk))

        writer.close()

        # 保存精确匹配索引
        with open(os.path.join(self.index_dir, "exact_match.json"), 'w') as f:
            json.dump(title_to_doc, f)

        self.title_to_doc = title_to_doc

    def search(self, query: str):
        """
        搜索最相似的论文

        Args:
            query: 查询的论文标题

        Returns:
            (id, title, abstract, score): 最相似论文的信息和BM25分数
        """
        if not self.searcher:
            raise ValueError("Searcher not initialized")

        # 首先尝试精确匹配
        query_lower = normalize_title(query)
        if query_lower in self.title_to_doc:
            doc = self.title_to_doc[query_lower]
            return (
                doc['id'],
                doc['title'],
                doc['abstract'],
                1.0  # 精确匹配返回最高分
            )

        # 如果没有精确匹配，使用BM25搜索
        hits = self.searcher.search(query, k=1)

        if not hits:
            return None

        # 获取最相似的文档
        doc = self.searcher.doc(hits[0].docid)
        return (
            doc.get('id'),
            doc.get('title'),
            doc.get('abstract'),
            hits[0].score
        )


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

