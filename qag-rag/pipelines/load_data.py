"""
QAG-RAG 数据加载流水线。

从 Neo4j 导出的 CSV 文件加载数据到 Milvus 和 Neo4j。

流程：
1. 解析 CSV：分离 Query 节点、Doc 节点、GENERATE_BY 关系、RELATED 关系
2. 清空 Milvus 集合和 Neo4j 数据库（可用 --skip-milvus 跳过 Milvus）
3. Milvus 集合创建与索引构建
4. 嵌入计算与 Milvus 插入（Doc → Query，带进度条）
5. Neo4j 加载（节点 + 关系，批量操作带进度条）

CSV 格式（Neo4j apoc.export.csv.all 导出）：
- Query 节点: _labels=":Query", question 有值
- Doc 节点: _labels=":Doc", id/text 有值
- GENERATE_BY: _type="GENERATE_BY", _start/_end 为 _id 引用
- RELATED: _type="RELATED", _start/_end 为 _id 引用, sim 为相似度
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

# 处理导入路径：使 qag-rag/ 目录可 import
sys.path.insert(0, '..')

from config import (
    MilvusConfig,
    Neo4jConfig,
    embedding_config as default_embedding_config,
    milvus_config as default_milvus_config,
    neo4j_config as default_neo4j_config,
)
from core.llm_provider import get_embedding_provider, BaseLLMProvider


# ---------------------------------------------------------------------------
# Step 1：解析 CSV
# ---------------------------------------------------------------------------

def parse_csv(csv_path: str) -> Tuple[
    Dict[str, Dict[str, Any]],
    Dict[str, Dict[str, Any]],
    List[Dict[str, str]],
    List[Dict[str, Any]],
]:
    """
    解析 Neo4j 导出的 CSV 文件。

    Returns:
        queries: {_id: {"question": str}}
        docs: {_id: {"id": str, "text": str}}
        generate_by: [{"query_id": _id, "doc_id": _id}]
        related: [{"source_id": _id, "target_id": _id, "sim": float}]
    """
    queries: Dict[str, Dict[str, Any]] = {}
    docs: Dict[str, Dict[str, Any]] = {}
    generate_by: List[Dict[str, str]] = []
    related: List[Dict[str, Any]] = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels = row.get('_labels', '').strip()
            rel_type = row.get('_type', '').strip()

            if ':Query' in labels:
                question = row.get('question', '').strip()
                if question:
                    queries[row['_id'].strip()] = {'question': question}

            elif ':Doc' in labels:
                doc_id = row.get('id', '').strip()
                text = row.get('text', '').strip()
                if doc_id and text:
                    docs[row['_id'].strip()] = {'id': doc_id, 'text': text}

            elif rel_type == 'GENERATE_BY':
                start_id = row.get('_start', '').strip()
                end_id = row.get('_end', '').strip()
                if start_id and end_id:
                    generate_by.append({'query_id': start_id, 'doc_id': end_id})

            elif rel_type == 'RELATED':
                start_id = row.get('_start', '').strip()
                end_id = row.get('_end', '').strip()
                sim_str = row.get('sim', '0').strip()
                if start_id and end_id:
                    related.append({
                        'source_id': start_id,
                        'target_id': end_id,
                        'sim': float(sim_str) if sim_str else 0.0,
                    })

    return queries, docs, generate_by, related


# ---------------------------------------------------------------------------
# Step 2：清空数据库
# ---------------------------------------------------------------------------

def clear_databases(milvus_config: MilvusConfig, neo4j_config: Neo4jConfig):
    """清空 Milvus 集合和 Neo4j 数据库的所有数据。"""
    from pymilvus import MilvusClient
    from neo4j import GraphDatabase

    # 清空 Milvus
    print("\n[清空] Milvus ...")
    client = MilvusClient(
        uri=milvus_config.uri,
        token=milvus_config.token,
        db_name=milvus_config.db_name,
    )

    for coll_name in [milvus_config.query_collection, milvus_config.chunk_collection]:
        if client.has_collection(collection_name=coll_name):
            client.drop_collection(collection_name=coll_name)
            print(f"  已删除集合 {coll_name}")
        else:
            print(f"  集合 {coll_name} 不存在，跳过")

    # 清空 Neo4j
    print("[清空] Neo4j ...")
    driver = GraphDatabase.driver(neo4j_config.uri, auth=neo4j_config.auth)
    with driver:
        driver.execute_query(
            "MATCH (n) DETACH DELETE n",
            database_="neo4j",
        )
        print("  已删除所有 Neo4j 节点和关系")
    driver.close()


# ---------------------------------------------------------------------------
# Step 3：Milvus 集合创建
# ---------------------------------------------------------------------------

def create_milvus_collections(milvus_config: MilvusConfig, embedding_dim: int) -> Any:
    """创建 Milvus 集合和索引。"""
    from pymilvus import MilvusClient, DataType

    client = MilvusClient(
        uri=milvus_config.uri,
        token=milvus_config.token,
        db_name=milvus_config.db_name,
    )

    index_type = milvus_config.index_type

    # query_collection
    query_schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    query_schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    query_schema.add_field(field_name="question", datatype=DataType.VARCHAR, max_length=65535)
    query_schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=embedding_dim)
    query_schema.add_field(field_name="doc_ids", datatype=DataType.VARCHAR, max_length=65535)
    query_schema.add_field(field_name="related_questions", datatype=DataType.VARCHAR, max_length=65535)

    query_index_params = client.prepare_index_params()
    query_index_params.add_index(field_name="id", index_type="AUTOINDEX")
    query_index_params.add_index(field_name="embedding", index_type=index_type, metric_type="COSINE")

    client.create_collection(
        collection_name=milvus_config.query_collection,
        schema=query_schema,
        index_params=query_index_params,
    )
    print(f"[Milvus] 已创建集合 {milvus_config.query_collection} (index={index_type})")

    # chunk_collection
    chunk_schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    chunk_schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=65535, is_primary=True)
    chunk_schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
    chunk_schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=embedding_dim)

    chunk_index_params = client.prepare_index_params()
    chunk_index_params.add_index(field_name="id", index_type="AUTOINDEX")
    chunk_index_params.add_index(field_name="embedding", index_type=index_type, metric_type="COSINE")

    client.create_collection(
        collection_name=milvus_config.chunk_collection,
        schema=chunk_schema,
        index_params=chunk_index_params,
    )
    print(f"[Milvus] 已创建集合 {milvus_config.chunk_collection} (index={index_type})")

    return client


# ---------------------------------------------------------------------------
# Step 4：嵌入计算与 Milvus 插入
# ---------------------------------------------------------------------------

def load_milvus(
    client: Any,
    queries: Dict[str, Dict[str, Any]],
    docs: Dict[str, Dict[str, Any]],
    generate_by: List[Dict[str, str]],
    related: List[Dict[str, Any]],
    embedding_provider: BaseLLMProvider,
    milvus_config: MilvusConfig,
    batch_size: int = 50,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    将 Doc 和 Query 批量插入 Milvus。

    Returns:
        query_csv_id_map: {csv_id: question}
        doc_csv_id_map: {csv_id: doc_id}
    """
    # 建立 doc_csv_id -> [query_csv_id] 映射
    doc_to_queries: Dict[str, List[str]] = defaultdict(list)
    for rel in generate_by:
        doc_to_queries[rel['doc_id']].append(rel['query_id'])

    # 建立 related_questions 映射
    csv_id_to_question = {cid: d['question'] for cid, d in queries.items()}
    related_map: Dict[str, List[str]] = defaultdict(list)
    for rel in related:
        src_q = csv_id_to_question.get(rel['source_id'])
        tgt_q = csv_id_to_question.get(rel['target_id'])
        if src_q and tgt_q:
            related_map[src_q].append(tgt_q)
            related_map[tgt_q].append(src_q)

    # --- 4a: Doc 嵌入与插入 ---
    print(f"\n[Milvus] 正在嵌入 {len(docs)} 个 Doc ...")
    doc_csv_id_map: Dict[str, str] = {}  # csv_id -> doc_id

    doc_items = list(docs.items())
    for i in tqdm(range(0, len(doc_items), batch_size), desc="嵌入 Doc"):
        batch_items = doc_items[i:i + batch_size]
        texts = [item[1]['text'] for item in batch_items]
        embeddings = [embedding_provider.embed(t) for t in texts]

        rows = []
        for (csv_id, doc_data), emb in zip(batch_items, embeddings):
            doc_csv_id_map[csv_id] = doc_data['id']
            rows.append({
                'id': doc_data['id'],
                'text': doc_data['text'],
                'embedding': emb,
            })

        client.insert(collection_name=milvus_config.chunk_collection, data=rows)

    print(f"[Milvus] 已插入 {len(doc_items)} 个 Doc 到 {milvus_config.chunk_collection}")

    # --- 4b: Query 嵌入与插入 ---
    print(f"\n[Milvus] 正在嵌入 {len(queries)} 个 Query ...")
    query_csv_id_map: Dict[str, str] = {}  # csv_id -> question

    query_items = list(queries.items())
    for i in tqdm(range(0, len(query_items), batch_size), desc="嵌入 Query"):
        batch_items = query_items[i:i + batch_size]
        questions = [item[1]['question'] for item in batch_items]
        embeddings = [embedding_provider.embed(q) for q in questions]

        rows = []
        for idx, ((csv_id, q_data), emb) in enumerate(zip(batch_items, embeddings)):
            milvus_id = i * batch_size + idx + 1
            query_csv_id_map[csv_id] = q_data['question']

            # 该 Query 关联的 Doc 业务 ID
            related_doc_csv_ids = doc_to_queries.get(csv_id, [])
            related_doc_ids = [
                docs[dcsv_id]['id']
                for dcsv_id in related_doc_csv_ids
                if dcsv_id in docs
            ]

            rows.append({
                'id': milvus_id,
                'question': q_data['question'],
                'embedding': emb,
                'doc_ids': json.dumps(related_doc_ids, ensure_ascii=False),
                'related_questions': json.dumps(
                    list(set(related_map.get(q_data['question'], []))),
                    ensure_ascii=False,
                ),
            })

        client.insert(collection_name=milvus_config.query_collection, data=rows)

    print(f"[Milvus] 已插入 {len(query_items)} 个 Query 到 {milvus_config.query_collection}")

    return query_csv_id_map, doc_csv_id_map


# ---------------------------------------------------------------------------
# Step 5：Neo4j 加载
# ---------------------------------------------------------------------------

def load_neo4j(
    queries: Dict[str, Dict[str, Any]],
    docs: Dict[str, Dict[str, Any]],
    generate_by: List[Dict[str, str]],
    related: List[Dict[str, Any]],
    neo4j_config: Neo4jConfig,
    batch_size: int = 100,
):
    """批量加载数据到 Neo4j。"""
    from neo4j import GraphDatabase

    print(f"\n[Neo4j] 正在连接到 {neo4j_config.uri} ...")
    driver = GraphDatabase.driver(neo4j_config.uri, auth=neo4j_config.auth)

    csv_id_to_question = {cid: d for cid, d in queries.items()}
    csv_id_to_doc = {cid: d for cid, d in docs.items()}

    with driver:
        # Doc 节点
        doc_values = list(docs.values())
        print(f"[Neo4j] 正在创建 {len(doc_values)} 个 Doc 节点 ...")
        for i in tqdm(range(0, len(doc_values), batch_size), desc="创建 Doc 节点"):
            batch = doc_values[i:i + batch_size]
            driver.execute_query(
                "UNWIND $batch AS row MERGE (d:Doc {id: row.id, text: row.text})",
                batch=batch,
                database_="neo4j",
            )

        # Query 节点
        query_values = list(queries.values())
        print(f"[Neo4j] 正在创建 {len(query_values)} 个 Query 节点 ...")
        for i in tqdm(range(0, len(query_values), batch_size), desc="创建 Query 节点"):
            batch = query_values[i:i + batch_size]
            driver.execute_query(
                "UNWIND $batch AS row MERGE (q:Query {question: row.question})",
                batch=batch,
                database_="neo4j",
            )

        # GENERATE_BY 关系
        gb_mapped = []
        for rel in generate_by:
            q = csv_id_to_question.get(rel['query_id'])
            d = csv_id_to_doc.get(rel['doc_id'])
            if q and d:
                gb_mapped.append({'question': q['question'], 'doc_id': d['id']})

        print(f"[Neo4j] 正在创建 {len(gb_mapped)} 个 GENERATE_BY 关系 ...")
        for i in tqdm(range(0, len(gb_mapped), batch_size), desc="创建 GENERATE_BY"):
            batch = gb_mapped[i:i + batch_size]
            driver.execute_query(
                """
                UNWIND $batch AS row
                MATCH (q:Query {question: row.question})
                MATCH (d:Doc {id: row.doc_id})
                MERGE (q)-[:GENERATE_BY]-(d)
                """,
                batch=batch,
                database_="neo4j",
            )

        # RELATED 关系
        rel_mapped = []
        for rel in related:
            src = csv_id_to_question.get(rel['source_id'])
            tgt = csv_id_to_question.get(rel['target_id'])
            if src and tgt:
                rel_mapped.append({
                    'src_question': src['question'],
                    'tgt_question': tgt['question'],
                    'sim': rel['sim'],
                })

        print(f"[Neo4j] 正在创建 {len(rel_mapped)} 个 RELATED 关系 ...")
        for i in tqdm(range(0, len(rel_mapped), batch_size), desc="创建 RELATED"):
            batch = rel_mapped[i:i + batch_size]
            driver.execute_query(
                """
                UNWIND $batch AS row
                MATCH (q:Query {question: row.src_question})
                MATCH (r:Query {question: row.tgt_question})
                MERGE (q)-[:RELATED {sim: row.sim}]-(r)
                """,
                batch=batch,
                database_="neo4j",
            )

    driver.close()
    print("[Neo4j] 连接已关闭")


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="QAG-RAG 数据加载流水线")
    parser.add_argument("--csv", type=str, required=True, help="Neo4j 导出的 CSV 文件路径")
    parser.add_argument("--batch-size", type=int, default=50, help="批量插入大小（默认 50）")
    parser.add_argument("--skip-milvus", action="store_true", help="跳过 Milvus 清空与重建（仅加载 Neo4j）")
    args = parser.parse_args()

    emb_cfg = default_embedding_config
    milvus_cfg = default_milvus_config
    neo4j_cfg = default_neo4j_config

    print("=" * 60)
    print("QAG-RAG: 数据加载流水线")
    print("=" * 60)
    print(f"CSV: {args.csv}")
    print(f"Milvus: {milvus_cfg.uri} (db={milvus_cfg.db_name})")
    print(f"Neo4j: {neo4j_cfg.uri}")
    print(f"Embedding: {emb_cfg.provider}/{emb_cfg.model} (dim={emb_cfg.dimension})")
    print("=" * 60)

    # Step 1: 解析 CSV
    print("\n[Step 1] 解析 CSV ...")
    queries, docs, generate_by, related = parse_csv(args.csv)
    print(f"[解析] Query: {len(queries)}, Doc: {len(docs)}, "
          f"GENERATE_BY: {len(generate_by)}, RELATED: {len(related)}")

    # Step 2: 清空数据库
    if not args.skip_milvus:
        print("\n[Step 2] 清空数据库 ...")
        clear_databases(milvus_cfg, neo4j_cfg)

        # Step 3: 创建 Milvus 集合
        print("\n[Step 3] 创建 Milvus 集合 ...")
        client = create_milvus_collections(milvus_cfg, emb_cfg.dimension)

        # Step 4: 嵌入计算与 Milvus 插入
        print("\n[Step 4] 嵌入计算与 Milvus 插入 ...")
        embedding_provider = get_embedding_provider(
            emb_cfg.provider,
            api_key=emb_cfg.api_key,
            model=emb_cfg.model,
            base_url=emb_cfg.base_url,
        )
        load_milvus(
            client=client,
            queries=queries,
            docs=docs,
            generate_by=generate_by,
            related=related,
            embedding_provider=embedding_provider,
            milvus_config=milvus_cfg,
            batch_size=args.batch_size,
        )
    else:
        print("\n[跳过] Milvus 清空与重建（--skip-milvus）")

    # Step 5: Neo4j 加载
    print("\n[Neo4j] 开始加载 ...")
    load_neo4j(
        queries=queries,
        docs=docs,
        generate_by=generate_by,
        related=related,
        neo4j_config=neo4j_cfg,
        batch_size=args.batch_size,
    )

    print("\n" + "=" * 60)
    print("数据加载完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
