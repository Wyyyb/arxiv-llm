import json
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID, STORED
from whoosh.qparser import QueryParser
from whoosh.analysis import StandardAnalyzer
from whoosh import scoring
from typing import Dict, Tuple
from tqdm import tqdm
import os
import copy


class OptimizedBM25Search:
    def __init__(self, index_dir: str):
        """
        初始化搜索引擎

        Args:
            index_dir: 索引存储目录
        """
        self.index_dir = index_dir
        # 定义文档模式
        self.schema = Schema(
            id=ID(stored=True),
            title=TEXT(stored=True, analyzer=StandardAnalyzer(), field_boost=2.0),
            abstract=STORED
        )

    def build_index(self, file_path: str, batch_size: int = 10000):
        """
        构建索引

        Args:
            file_path: jsonl文件路径
            batch_size: 批处理大小
        """
        # 创建索引目录
        os.makedirs(self.index_dir, exist_ok=True)
        ix = create_in(self.index_dir, self.schema)

        # 批量写入文档
        writer = ix.writer(procs=8, multisegment=True)  # 使用多进程写入

        print("构建索引...")
        doc_count = 0
        batch_count = 0

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                paper = json.loads(line.strip())

                # 添加文档到索引
                writer.add_document(
                    id=paper['id'],
                    title=paper['title'],
                    abstract=paper['abstract']
                )

                doc_count += 1
                batch_count += 1

                # 批量提交
                if batch_count >= batch_size:
                    writer.commit()
                    writer = ix.writer(procs=8, multisegment=True)
                    batch_count = 0

        # 提交剩余文档
        if batch_count > 0:
            writer.commit()

        print(f"索引构建完成，共索引了 {doc_count} 条文档")

    def search_best(self, query: str):
        """
        搜索最相似的文档

        Args:
            query: 查询字符串

        Returns:
            (最相似文档, 分数)
        """
        ix = open_dir(self.index_dir)

        # 使用BM25F评分器
        searcher = ix.searcher(weighting=scoring.BM25F(B=0.75, K1=1.5))

        # 创建查询解析器，主要匹配标题
        parser = QueryParser("title", ix.schema)
        q = parser.parse(query)

        # 搜索最佳匹配
        results = searcher.search(q, limit=1)

        if len(results) == 0:
            searcher.close()
            return None, 0.0

        # 构造返回结果
        best_match = results[0]
        doc = {
            'id': best_match['id'],
            'title': best_match['title'],
            'abstract': best_match['abstract']
        }
        score = best_match.score

        searcher.close()
        return doc, score


def load_documents(file_path):
    with open(file_path, 'r') as fi:
        documents = json.load(fi)
    documents = {k.strip(): v for k, v in documents.items()}
    return documents


def main():
    os.makedirs("../local_darth_1110", exist_ok=True)
    os.makedirs("../local_darth_1110/whoosh_index_directory", exist_ok=True)
    index = OptimizedBM25Search("../local_darth_1110/whoosh_index_directory")
    index.build_index("/data/yubowang/ss_offline_data/ss_offline_data_1109.jsonl")
    index = OptimizedBM25Search("../local_darth_1110/whoosh_index_directory")
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



