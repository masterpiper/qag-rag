"""
知识图谱构建模块。

从 Milvus 读取查询和文档数据，在 Neo4j 中构建知识图谱：
- Query 节点：表示问题
- Doc 节点：表示文档
- GENERATE_BY 关系：Query → Doc
- RELATED 关系：Query ↔ Query（带相似度分数）
"""

import ast
import sys
from tqdm import tqdm
from pymilvus import Collection, connections
from neo4j import GraphDatabase
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 导入本地模块
sys.path.insert(0, '..')
from config import milvus_config, neo4j_config


def build_knowledge_graph(
    milvus_host: str = "127.0.0.1",
    milvus_port: str = "19530",
    batch_size: int = 1000,
    skip_batches: int = 366,
):
    """
    构建知识图谱。
    
    Args:
        milvus_host: Milvus 主机地址
        milvus_port: Milvus 端口
        batch_size: 批处理大小
        skip_batches: 跳过的批次数（用于断点续建）
    """
    print("=" * 60)
    print("QAG-RAG: Question-Chunk Graph构建")
    print("=" * 60)
    print(f"Milvus: {milvus_host}:{milvus_port}")
    print(f"Neo4j: {neo4j_config.uri}")
    print("=" * 60)
    
    # 连接 Milvus
    print("\n连接 Milvus...")
    connections.connect(
        host=milvus_host,
        port=milvus_port,
        db_name=milvus_config.db_name
    )
    
    # 获取集合
    collection = Collection(name="bioasq_query")
    iterator = collection.query_iterator(
        batch_size=batch_size,
        output_fields=["id", "question", "doc_ids", "related_questions", "embedding"],
        expr=""
    )
    
    # 连接 Neo4j
    print("连接 Neo4j...")
    driver = GraphDatabase.driver(neo4j_config.uri, auth=neo4j_config.auth)
    
    # 估算总数（用于进度条）
    total_estimate = 400000  # 估计值
    pbar = tqdm(total=total_estimate, desc="Processing")
    
    cnt = 0
    batch_cnt = skip_batches
    
    try:
        with driver:
            while True:
                batch = iterator.next()
                
                # 跳过指定批次
                if batch_cnt > 0:
                    batch_cnt -= 1
                    cnt += len(batch)
                    pbar.update(len(batch))
                    continue
                
                # 处理每个查询
                for item in batch:
                    # 创建或更新查询节点
                    driver.execute_query(
                        """
                        MERGE (q:Query {question: $question})
                        RETURN q.question AS question
                        """,
                        question=item["question"],
                        database_="neo4j"
                    )
                    
                    # 获取并创建相关文档节点
                    milvus_client = milvus_config.uri
                    from pymilvus import MilvusClient
                    client = MilvusClient(
                        uri=milvus_config.uri,
                        token=milvus_config.token,
                        db_name=milvus_config.db_name
                    )
                    
                    try:
                        content = client.get(
                            collection_name="native_rag",
                            ids=ast.literal_eval(item["doc_ids"]),
                            output_fields=["text"],
                        )
                        
                        for doc in content:
                            # 创建文档节点
                            driver.execute_query(
                                """
                                MERGE (d:Doc {text: $text, id: $id})
                                RETURN d.text AS text
                                """,
                                text=doc["text"],
                                id=doc["id"],
                                database_="neo4j"
                            )
                            
                            # 创建 GENERATE_BY 关系
                            driver.execute_query(
                                """
                                MATCH (q:Query {question: $question}), (d:Doc {text: $text})
                                MERGE (q)-[:GENERATE_BY]-(d)
                                """,
                                question=item["question"],
                                text=doc["text"],
                                database_="neo4j"
                            )
                        
                        # 获取并创建相关查询节点
                        related_content = client.get(
                            collection_name="bioasq_query",
                            ids=ast.literal_eval(item["related_questions"]),
                            output_fields=["question", "embedding"],
                        )
                        
                        for related in related_content:
                            # 创建相关查询节点
                            driver.execute_query(
                                """
                                MERGE (q:Query {question: $question})
                                RETURN q.question AS question
                                """,
                                question=related["question"],
                                database_="neo4j"
                            )
                            
                            # 计算相似度
                            sim = cosine_similarity(
                                np.asarray(item["embedding"]).reshape(1, -1),
                                np.asarray(related["embedding"]).reshape(1, -1)
                            )[0][0]
                            
                            # 创建 RELATED 关系
                            driver.execute_query(
                                """
                                MATCH (q:Query {question: $question}), (r:Query {question: $related_question})
                                MERGE (q)-[:RELATED {sim: $sim}]-(r)
                                """,
                                question=item["question"],
                                related_question=related["question"],
                                sim=float(sim),
                                database_="neo4j"
                            )
                    
                    except Exception as e:
                        print(f"处理查询 '{item['question']}' 时出错：{e}")
                    
                    cnt += 1
                    pbar.update(1)
                
                if not batch:
                    break
                    
    except Exception as e:
        print(f"构建过程中出错：{e}")
    finally:
        pbar.close()
        print(f"\n处理完成，共处理 {cnt} 条记录")


def main():
    """主函数。"""
    build_knowledge_graph()


if __name__ == "__main__":
    main()
