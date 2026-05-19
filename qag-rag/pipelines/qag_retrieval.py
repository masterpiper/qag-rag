"""
QAG-RAG 检索流水线：子图检索与自底向上汇总。

流程：
1. 输入问题 q → Milvus 召回相似问题集合 Q
2. 以 Q 为根节点在 Neo4j 图谱中遍历，收集子图
3. 子图转化为树形结构
4. 自底向上使用 Hierarchical_Summary 提示词逐层汇总
5. 返回子图、树、最终答案

配置来源：
- qag-rag/pipelines/config.yml（检索与遍历参数）
- .env（API Key）
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

import yaml

# 处理导入路径：使 qag-rag/ 目录可 import
sys.path.insert(0, '..')

from config import (
    LLMConfig,
    EmbeddingConfig,
    MilvusConfig,
    Neo4jConfig,
    RetrievalConfig,
    llm_config as default_llm_config,
    embedding_config as default_embedding_config,
    milvus_config as default_milvus_config,
    neo4j_config as default_neo4j_config,
    retrieval_config as default_retrieval_config,
)
from core.llm_provider import get_llm_provider, get_embedding_provider, BaseLLMProvider
from core.prompt import PROMPT
from core.retrieval import QueryDocumentRetrieval


# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------

def load_pipeline_config(yaml_path: str = "config.yml") -> Dict[str, Any]:
    """读取 pipelines/config.yml，与 .env 合并为最终配置对象。"""
    yaml_path = os.path.join(os.path.dirname(__file__), yaml_path)
    with open(yaml_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}

    # 优先用 .env 覆盖（已注入 os.environ）
    llm_cfg = cfg.get('llm', {})
    emb_cfg = cfg.get('embedding', {})
    ret_cfg = cfg.get('retrieval', {})

    llm_config = LLMConfig(
        provider=os.getenv('LLM_PROVIDER', llm_cfg.get('provider', default_llm_config.provider)),
        model=os.getenv('LLM_MODEL', llm_cfg.get('model', default_llm_config.model)),
        api_key=os.getenv('LLM_API_KEY', default_llm_config.api_key),
        base_url=os.getenv('LLM_BASE_URL', default_llm_config.base_url),
        ollama_host=os.getenv('OLLAMA_HOST', default_llm_config.ollama_host),
    )

    embedding_config = EmbeddingConfig(
        provider=os.getenv('EMBEDDING_PROVIDER', emb_cfg.get('provider', default_embedding_config.provider)),
        model=os.getenv('EMBEDDING_MODEL', emb_cfg.get('model', default_embedding_config.model)),
        api_key=os.getenv('EMBEDDING_API_KEY', default_embedding_config.api_key),
        base_url=os.getenv('EMBEDDING_BASE_URL', default_embedding_config.base_url),
        ollama_host=os.getenv('OLLAMA_HOST', default_embedding_config.ollama_host),
        dimension=int(os.getenv('EMBEDDING_DIMENSION', emb_cfg.get('dimension', default_embedding_config.dimension))),
    )

    milvus_config = default_milvus_config
    neo4j_config = default_neo4j_config

    retrieval_config = RetrievalConfig(
        merge_threshold=float(ret_cfg.get('merge_threshold', default_retrieval_config.merge_threshold)),
        related_threshold=float(ret_cfg.get('related_threshold', default_retrieval_config.related_threshold)),
        limite_num_q=int(ret_cfg.get('limite_num_q', default_retrieval_config.limite_num_q)),
        limite_related_docs=int(ret_cfg.get('limite_related_docs', default_retrieval_config.limite_related_docs)),
        graph_depth=int(ret_cfg.get('graph_depth', default_retrieval_config.graph_depth)),
        doc_top_k=int(ret_cfg.get('doc_top_k', default_retrieval_config.doc_top_k)),
        query_top_k=int(ret_cfg.get('query_top_k', default_retrieval_config.query_top_k)),
    )

    return {
        'llm': llm_config,
        'embedding': embedding_config,
        'milvus': milvus_config,
        'neo4j': neo4j_config,
        'retrieval': retrieval_config,
    }


# ---------------------------------------------------------------------------
# Step 2：子图检索
# ---------------------------------------------------------------------------

def retrieve_subgraph(
    question: str,
    llm_provider: BaseLLMProvider,
    embedding_provider: BaseLLMProvider,
    milvus_config: Optional[MilvusConfig] = None,
    neo4j_config: Optional[Neo4jConfig] = None,
    retrieval_config: Optional[RetrievalConfig] = None,
) -> Dict[str, Any]:
    """
    根据问题 q 检索相似问题集合 Q，再在 Neo4j 中遍历得到子图。

    Returns:
        {
            "root_queries": [{"question": str, "distance": float}],
            "nodes": {"Query": [...], "Doc": [...]},
            "relationships": [...]
        }
    """
    qdr = QueryDocumentRetrieval(
        llm_provider=llm_provider,
        embedding_provider=embedding_provider,
        milvus_config=milvus_config,
        neo4j_config=neo4j_config,
        retrieval_config=retrieval_config,
    )

    top_k = retrieval_config.query_top_k if retrieval_config else 5
    graph_depth = retrieval_config.graph_depth if retrieval_config else 2

    # 1) Milvus 召回相似问题 Q
    similar_queries = qdr.query_retrieval(question, top_k=top_k)
    root_queries = []
    for sq in similar_queries:
        entity = sq.get('entity', {})
        root_queries.append({
            'question': entity.get('question', ''),
            'distance': sq.get('distance', 0.0),
        })

    # 2) Neo4j 图遍历
    from neo4j import GraphDatabase

    neo4j_cfg = neo4j_config or default_neo4j_config
    nodes_query: Set[str] = set()
    nodes_doc: Dict[str, str] = {}  # id -> text
    relationships: List[Dict[str, Any]] = []

    with GraphDatabase.driver(neo4j_cfg.uri, auth=neo4j_cfg.auth) as driver:
        for rq in root_queries:
            q_text = rq['question']
            if not q_text:
                continue

            # 遍历 RELATED 关系，收集 Query 节点
            records, _, _ = driver.execute_query(
                f"""
                MATCH (start_q:Query {{question: $question}})
                MATCH path=(start_q)-[:RELATED*0..{graph_depth}]-(q:Query)
                WHERE ALL(r IN relationships(path) WHERE type(r)="RELATED")
                RETURN DISTINCT q.question AS question,
                       [r IN relationships(path) | {{start: startNode(r).question,
                                                    end: endNode(r).question,
                                                    sim: r.sim}}] AS rels
                """,
                question=q_text,
                database_="neo4j",
            )

            for record in records:
                q_node = record['question']
                nodes_query.add(q_node)
                for rel in record['rels']:
                    if rel['start'] and rel['end']:
                        rel_entry = {
                            'source': {'label': 'Query', 'question': rel['start']},
                            'target': {'label': 'Query', 'question': rel['end']},
                            'type': 'RELATED',
                            'properties': {'sim': rel['sim']} if rel.get('sim') is not None else {},
                        }
                        # 去重：基于无向边 key（先检查类型避免访问不存在的 'question' key）
                        rel_key = tuple(sorted((rel['start'], rel['end'])))
                        if not any(
                            r['type'] == 'RELATED'
                            and tuple(sorted((r['source']['question'], r['target']['question']))) == rel_key
                            for r in relationships
                        ):
                            relationships.append(rel_entry)

            # 收集这些 Query 关联的 Doc
            records2, _, _ = driver.execute_query(
                f"""
                MATCH (start_q:Query {{question: $question}})
                MATCH path=(start_q)-[:RELATED*0..{graph_depth}]-(q:Query)
                WHERE ALL(r IN relationships(path) WHERE type(r)="RELATED")
                MATCH (q)-[:GENERATE_BY]->(d:Doc)
                RETURN DISTINCT d.id AS id, d.text AS text, q.question AS q_question
                """,
                question=q_text,
                database_="neo4j",
            )

            for record in records2:
                doc_id = record['id']
                doc_text = record['text']
                q_question = record['q_question']
                nodes_doc[doc_id] = doc_text
                rel_entry = {
                    'source': {'label': 'Query', 'question': q_question},
                    'target': {'label': 'Doc', 'id': doc_id},
                    'type': 'GENERATE_BY',
                    'properties': {},
                }
                # 去重
                rel_key = (q_question, doc_id, 'GENERATE_BY')
                if not any(
                    (r['source'].get('question'), r['target'].get('id'), r['type']) == rel_key
                    for r in relationships
                ):
                    relationships.append(rel_entry)

    subgraph = {
        'root_queries': root_queries,
        'nodes': {
            'Query': [{'question': q} for q in sorted(nodes_query)],
            'Doc': [{'id': k, 'text': v} for k, v in sorted(nodes_doc.items())],
        },
        'relationships': relationships,
    }
    return subgraph


# ---------------------------------------------------------------------------
# Step 3：Feature-based filtering (论文 Section 3.3 Filter)
# ---------------------------------------------------------------------------

def filter_retrieval_tree(
    subgraph: Dict[str, Any],
    target_question: str,
    embedding_provider: BaseLLMProvider,
    threshold: float = 0.6,
) -> Dict[str, Any]:
    """
    基于特征的过滤机制（论文 Eq.7-9）。

    从四个来源提取特征：(1) 原始查询 (2) 匹配的相似问题 (3) 直接关联的 Doc
    对每个超过第 1 跳的 Doc 计算与特征矩阵的余弦相似度均值，低于阈值的剪枝。

    Returns:
        过滤后的子图（Doc 节点和关系被剪枝）
    """
    import numpy as np

    # --- 构建特征集 ---
    features: List[str] = []

    # (1) 原始查询
    features.append(target_question)

    # (2) 匹配的相似问题节点（Q_S）
    for rq in subgraph['root_queries']:
        q = rq.get('question', '')
        if q:
            features.append(q)

    # (3) 第 1 跳直接关联的 Doc（通过 GENERATE_BY 与 root queries 相连的）
    root_questions = {rq['question'] for rq in subgraph['root_queries']}
    layer1_docs: Set[str] = set()
    for rel in subgraph['relationships']:
        if rel['type'] == 'GENERATE_BY':
            src = rel['source'].get('question', '')
            if src in root_questions:
                doc_id = rel['target'].get('id', '')
                if doc_id:
                    layer1_docs.add(doc_id)

    # 添加 layer-1 docs 的文本作为特征
    doc_text_map = {d['id']: d['text'] for d in subgraph['nodes']['Doc']}
    for doc_id in layer1_docs:
        if doc_id in doc_text_map and doc_text_map[doc_id]:
            features.append(doc_text_map[doc_id])

    if not features:
        return subgraph

    # --- 构建特征矩阵的嵌入 ---
    feature_embeddings = []
    for f in features:
        emb = embedding_provider.embed(f)
        feature_embeddings.append(emb)

    # --- 对每个非 layer-1 的 Doc 计算相似度得分 ---
    nodes_query = subgraph['nodes']['Query']
    nodes_doc = subgraph['nodes']['Doc']
    relationships = subgraph['relationships']

    # 找出所有 Doc id 到其关联 Query 的映射
    doc_to_queries: Dict[str, List[str]] = defaultdict(list)
    for rel in relationships:
        if rel['type'] == 'GENERATE_BY':
            doc_id = rel['target'].get('id', '')
            q = rel['source'].get('question', '')
            if doc_id and q:
                doc_to_queries[doc_id].append(q)

    # 计算每个非 layer-1 Doc 的 score
    kept_doc_ids = set(layer1_docs)  # layer-1 docs 不过滤
    for doc_id, doc_text in list(doc_text_map.items()):
        if doc_id in layer1_docs:
            continue  # layer-1 跳过过滤
        if not doc_text:
            continue
        doc_emb = embedding_provider.embed(doc_text)
        # score = 平均余弦相似度
        sims = []
        for f_emb in feature_embeddings:
            sim = _cosine_sim(doc_emb, f_emb)
            sims.append(sim)
        score = float(np.mean(sims))
        if score >= threshold:
            kept_doc_ids.add(doc_id)

    # --- 找出仍有 Doc 的 Query 节点 ---
    queries_with_docs: Set[str] = set()
    for rel in relationships:
        if rel['type'] == 'GENERATE_BY' and rel['target'].get('id', '') in kept_doc_ids:
            q = rel['source'].get('question', '')
            if q:
                queries_with_docs.add(q)

    # 保留 root queries（即使没有 Doc）和仍有 Doc 的 Query 节点
    root_q_set = {rq['question'] for rq in subgraph['root_queries']}
    kept_queries = queries_with_docs | root_q_set

    # --- 构建过滤后的子图 ---
    filtered_nodes_query = [q for q in nodes_query if q['question'] in kept_queries]
    filtered_nodes_doc = [d for d in nodes_doc if d['id'] in kept_doc_ids]
    filtered_relationships = [
        r for r in relationships
        if (r['type'] == 'RELATED'
            and r['source'].get('question', '') in kept_queries
            and r['target'].get('question', '') in kept_queries)
        or (r['type'] == 'GENERATE_BY'
            and r['target'].get('id', '') in kept_doc_ids
            and r['source'].get('question', '') in kept_queries)
    ]

    original_doc_count = len(nodes_doc)
    filtered_doc_count = len(filtered_nodes_doc)
    pruned_queries = len(nodes_query) - len(filtered_nodes_query)

    result = dict(subgraph)
    result['nodes'] = dict(subgraph['nodes'])
    result['nodes']['Query'] = filtered_nodes_query
    result['nodes']['Doc'] = filtered_nodes_doc
    result['relationships'] = filtered_relationships
    result['_filter_info'] = {
        'original_docs': original_doc_count,
        'kept_docs': filtered_doc_count,
        'pruned_docs': original_doc_count - filtered_doc_count,
        'pruned_queries': pruned_queries,
    }
    return result


def _cosine_sim(a: List[float], b: List[float]) -> float:
    """计算两个向量的余弦相似度。"""
    import numpy as np
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    dot = float(np.dot(a_arr, b_arr))
    norm_a = float(np.linalg.norm(a_arr))
    norm_b = float(np.linalg.norm(b_arr))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Step 4：子图 → 树
# ---------------------------------------------------------------------------

def subgraph_to_tree(
    subgraph: Dict[str, Any],
    target_question: str,
    max_depth: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    将子图按 RELATED 关系层级展开为一棵树。

    target_question 作为虚拟根节点，所有召回的查询作为其子节点，
    最终只返回一棵树，避免多树合并导致数据膨胀。

    Args:
        max_depth: 树的最大深度（不含虚拟根），默认为 None 表示不限制。
                   建议设为 retrieval_config.graph_depth + 1 以与图遍历一致。

    Returns:
        [root_tree]
    """
    root_questions = [rq['question'] for rq in subgraph['root_queries']]
    query_nodes = {n['question'] for n in subgraph['nodes']['Query']}
    doc_map = {d['id']: d for d in subgraph['nodes']['Doc']}

    # 构建邻接表：question -> [(related_question, sim)]
    adj: Dict[str, List[tuple]] = defaultdict(list)
    for rel in subgraph['relationships']:
        if rel['type'] == 'RELATED':
            s = rel['source']['question']
            t = rel['target']['question']
            sim = rel['properties'].get('sim', 0.0)
            adj[s].append((t, sim))
            adj[t].append((s, sim))

    # 构建 Doc 关联：question -> [doc_dict]
    q_docs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rel in subgraph['relationships']:
        if rel['type'] == 'GENERATE_BY':
            q = rel['source']['question']
            doc_id = rel['target']['id']
            if doc_id in doc_map:
                q_docs[q].append(doc_map[doc_id])

    def _build_tree(q: str, visited: Set[str], depth: int = 0) -> Optional[Dict[str, Any]]:
        if q in visited:
            return None
        if max_depth is not None and depth > max_depth:
            return None
        visited.add(q)
        node = {
            'question': q,
            'docs': q_docs.get(q, []),
            'children': [],
        }
        for child_q, sim in sorted(adj.get(q, []), key=lambda x: x[1], reverse=True):
            if child_q not in visited:
                child_tree = _build_tree(child_q, visited, depth + 1)
                if child_tree:
                    node['children'].append(child_tree)
        return node

    # 用 target_question 作为根，所有召回的查询作为子节点构建一棵树
    visited: Set[str] = {target_question}
    children = []
    for rq in root_questions:
        if rq in query_nodes:
            tree = _build_tree(rq, visited)
            if tree:
                children.append(tree)

    # Filter 剪枝后可能产生独立连通分量——保留但未被访问的 Query 节点
    # 直接作为根的子节点挂载
    for q in query_nodes:
        if q not in visited:
            tree = _build_tree(q, visited)
            if tree:
                children.append(tree)

    root = {
        'question': target_question,
        'docs': [],
        'children': children,
    }
    return [root]


# ---------------------------------------------------------------------------
# Step 4：自底向上汇总
# ---------------------------------------------------------------------------

def hierarchical_summarize(
    trees: List[Dict[str, Any]],
    target_question: str,
    llm_provider: BaseLLMProvider,
    on_layer_summary=None,
) -> str:
    """
    自底向上逐层使用 Hierarchical_Summary 提示词汇总。
    每层做一次 LLM 调用，将子层摘要作为输入传递给上一层。

    target_question: 用于构建虚拟根节点（不填入提示词）
    on_layer_summary: 回调函数，每层汇总后调用，签名为 (depth, summary)
    """
    prompt_template = PROMPT["Hierarchical_Summary"]

    if not trees:
        return "No relevant information found."

    # 将多棵树合并为一棵虚拟根树，统一按层处理
    if len(trees) == 1:
        root = trees[0]
    else:
        root = {
            'question': target_question,
            'docs': [],
            'children': trees,
        }

    # 按深度收集节点，同时记录每个节点的 depth
    def _collect_by_depth(tree: Dict[str, Any]) -> tuple[Dict[int, List[Dict[str, Any]]], Dict[str, int]]:
        by_depth: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        node_depth: Dict[str, int] = {}  # question -> depth

        def _walk(node: Dict[str, Any], depth: int):
            by_depth[depth].append(node)
            node_depth[node['question']] = depth
            for child in node.get('children', []):
                _walk(child, depth + 1)

        _walk(tree, 0)
        return by_depth, node_depth

    nodes_by_depth, _ = _collect_by_depth(root)
    max_depth = max(nodes_by_depth.keys())

    # 自底向上逐层汇总：从最深的叶子层开始
    # prev_summary 保存上一层（更深一层）的汇总摘要，作为当前层 chunks 的输入
    prev_summary: str = ""

    for depth in range(max_depth, -1, -1):
        # 构造本层 dataset
        dataset = []
        for node in nodes_by_depth[depth]:
            chunks = [d['text'] for d in node.get('docs', [])]
            dataset.append({
                "sub_question": node['question'],
                "chunks": chunks,
            })

        dataset_json = json.dumps(dataset, ensure_ascii=False, indent=2)
        context = prev_summary if prev_summary else "None"
        prompt = prompt_template.replace("{dataset}", dataset_json).replace("{context}", context)

        response = llm_provider.generate(
            prompt=prompt,
            max_tokens=2048,
            skip_extra_body=True,
        )
        prev_summary = response.strip()

        if on_layer_summary:
            on_layer_summary(depth, prev_summary)

    return prev_summary


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="QAG-RAG 检索流水线")
    parser.add_argument("--question", "-q", type=str, required=True, help="输入问题")
    parser.add_argument("--config", "-c", type=str, default="config.yml", help="配置文件路径")
    parser.add_argument("--output", "-o", type=str, default=None, help="输出 JSON 文件路径")
    args = parser.parse_args()

    # 加载配置
    cfg = load_pipeline_config(args.config)
    llm_cfg = cfg['llm']
    emb_cfg = cfg['embedding']
    milvus_cfg = cfg['milvus']
    neo4j_cfg = cfg['neo4j']
    ret_cfg = cfg['retrieval']

    # 初始化 provider
    llm_provider = get_llm_provider(
        llm_cfg.provider,
        api_key=llm_cfg.api_key,
        model=llm_cfg.model,
        base_url=llm_cfg.base_url,
    )
    embedding_provider = get_embedding_provider(
        emb_cfg.provider,
        api_key=emb_cfg.api_key,
        model=emb_cfg.model,
        base_url=emb_cfg.base_url,
    )

    # 执行检索
    print(f"[检索] 问题: {args.question}")
    subgraph = retrieve_subgraph(
        question=args.question,
        llm_provider=llm_provider,
        embedding_provider=embedding_provider,
        milvus_config=milvus_cfg,
        neo4j_config=neo4j_cfg,
        retrieval_config=ret_cfg,
    )
    print(f"[子图] 根节点数: {len(subgraph['root_queries'])}, "
          f"Query节点数: {len(subgraph['nodes']['Query'])}, "
          f"Doc节点数: {len(subgraph['nodes']['Doc'])}, "
          f"关系数: {len(subgraph['relationships'])}")

    # 特征过滤（论文 Section 3.3 Filter）
    print(f"[过滤] 阈值={ret_cfg.filter_threshold}")
    subgraph = filter_retrieval_tree(
        subgraph, target_question=args.question,
        embedding_provider=embedding_provider,
        threshold=ret_cfg.filter_threshold,
    )
    fi = subgraph.get('_filter_info', {})
    print(f"[过滤] 原始 Doc={fi.get('original_docs', '?')}, "
          f"保留 Doc={fi.get('kept_docs', '?')}, "
          f"剪枝 Doc={fi.get('pruned_docs', '?')}, "
          f"剪枝 Query={fi.get('pruned_queries', '?')}")

    # 转化为树（限制深度与图遍历一致）
    trees = subgraph_to_tree(
        subgraph, target_question=args.question, max_depth=ret_cfg.graph_depth - 1
    )
    print(f"[树] 根树数量: {len(trees)}")

    # 自底向上汇总
    print("[汇总] 自底向上生成答案...")
    answer = hierarchical_summarize(
        trees=trees,
        target_question=args.question,
        llm_provider=llm_provider,
    )
    print(f"[答案]\n{answer}\n")

    result = {
        "question": args.question,
        "subgraph": subgraph,
        "trees": trees,
        "answer": answer,
    }

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[保存] 结果已写入 {args.output}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
