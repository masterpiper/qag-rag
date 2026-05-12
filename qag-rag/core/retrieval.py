"""
查询 - 文档检索模块。

封装基于 Milvus 向量数据库和 Neo4j 图数据库的混合检索功能，
支持多跳推理和关键词过滤。
"""

from typing import Optional, Set, Dict, Any, List
from pymilvus import MilvusClient
from neo4j import GraphDatabase
import re
from keybert import KeyBERT

from config import MilvusConfig, Neo4jConfig, RetrievalConfig
from .llm_provider import BaseLLMProvider, get_embedding_provider


class QueryDocumentRetrieval:
    """
    查询 - 文档检索类。

    封装了 Milvus 向量数据库和 Neo4j 图数据库的操作，
    提供多种检索策略包括向量检索、图遍历和关键词过滤。
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        embedding_provider: BaseLLMProvider,
        milvus_config: Optional[MilvusConfig] = None,
        neo4j_config: Optional[Neo4jConfig] = None,
        retrieval_config: Optional[RetrievalConfig] = None,
    ):
        """
        初始化检索器。

        Args:
            llm_provider: LLM 提供者实例
            embedding_provider: 嵌入提供者实例
            milvus_config: Milvus 配置
            neo4j_config: Neo4j 配置
            retrieval_config: 检索参数配置
        """
        self.debug = False
        
        # 配置初始化
        self.milvus_config = milvus_config or MilvusConfig()
        self.neo4j_config = neo4j_config or Neo4jConfig()
        self.retrieval_config = retrieval_config or RetrievalConfig()
        
        # 提供者初始化
        self.llm_provider = llm_provider
        self.embedding_provider = embedding_provider
        
        # Milvus 客户端初始化
        self.milvus_client = MilvusClient(
            uri=self.milvus_config.uri,
            token=self.milvus_config.token,
            db_name=self.milvus_config.db_name
        )
        
        # KeyBERT 关键词提取模型
        self.kw_model = KeyBERT()

    def query_retrieval(self, input_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        根据初始查询召回相似查询。

        使用向量相似度搜索 query_collection 集合中的相似问题。

        Args:
            input_text: 输入查询文本
            top_k: 返回的相似查询数量，默认为 5

        Returns:
            相似查询结果列表
        """
        input_emb = [self.embedding_provider.embed(input_text)]

        q_res = self.milvus_client.search(
            collection_name=self.milvus_config.query_collection,
            data=input_emb,
            limit=top_k,
            output_fields=["question", "doc_ids", "related_questions"]
        )
        return q_res[0] if q_res else []

    def document_retrieval(self, input_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        根据查询召回相关文档。

        使用向量相似度搜索 chunk_collection 集合中的相关文档。

        Args:
            input_text: 输入查询文本
            top_k: 返回的文档数量，默认为 5

        Returns:
            相关文档结果列表
        """
        input_emb = [self.embedding_provider.embed(input_text)]

        doc_res = self.milvus_client.search(
            collection_name=self.milvus_config.chunk_collection,
            data=input_emb,
            limit=top_k,
            output_fields=["text", "id"]
        )
        return doc_res[0] if doc_res else []

    def keyword_generation(self, text: str) -> List[str]:
        """
        从文档内容生成关键词。

        使用 KeyBERT 模型从文本中提取关键词和关键短语。

        Args:
            text: 输入文本内容

        Returns:
            提取的关键词列表
        """
        # 使用 KeyBERT 提取关键词，ngram 范围为 1-2，去除英文停用词，返回前 5 个关键词
        keywords = self.kw_model.extract_keywords(
            text, 
            keyphrase_ngram_range=(1, 2), 
            stop_words='english', 
            top_n=5
        )
        
        # 提取关键词字符串
        return [kw[0] for kw in keywords]

    def keywords_filter(self, doc_ids: Set[str], keyword_list: List[str]) -> Set[str]:
        """
        根据关键词过滤文档。

        从文档列表中筛选包含指定关键词的文档。

        Args:
            doc_ids: 文档 ID 集合
            keyword_list: 关键词列表

        Returns:
            包含关键词的文档 ID 集合
        """
        if not keyword_list:
            return doc_ids

        # 从 Milvus 获取文档内容
        text_list = self.milvus_client.get(
            collection_name=self.milvus_config.chunk_collection,
            ids=list(doc_ids),
            output_fields=['id', 'content']
        )

        # 转义关键词并构建正则表达式模式
        escaped_keywords = [re.escape(kw) for kw in keyword_list]
        pattern = r'\b(?:' + '|'.join(escaped_keywords) + r')\b'
        compiled_pattern = re.compile(pattern, re.IGNORECASE)

        # 遍历文档，匹配包含关键词的文档
        filtered_ids = set()
        for doc in text_list:
            if compiled_pattern.search(doc['content']):
                filtered_ids.add(doc['id'])
        
        return filtered_ids

    def deep_recall_documents(
        self, 
        queries: List[Dict[str, Any]], 
        graph_depth: Optional[int] = None
    ) -> Set[str]:
        """
        根据查询进行深度召回。

        在 Neo4j 图数据库中遍历查询节点间的 RELATED 关系，
        找到多跳关联的文档。

        Args:
            queries: 查询字典列表，每个字典包含 question 字段
            graph_depth: 图遍历的最大深度

        Returns:
            召回的文档 ID 集合
        """
        if graph_depth is None:
            graph_depth = self.retrieval_config.graph_depth
            
        related_docs = set()
        
        with GraphDatabase.driver(
            uri=self.neo4j_config.uri, 
            auth=self.neo4j_config.auth
        ) as driver:
            for query in queries:
                # Cypher 查询：从起始查询节点出发，沿 RELATED 关系遍历图
                records, _, _ = driver.execute_query(
                    f"""
                    MATCH (start_q:Query {{question: $question}})
                    MATCH path=(start_q)-[:RELATED*0..{graph_depth}]-(q:Query)
                    WHERE ALL(r IN relationships(path) WHERE type(r)="RELATED")
                    MATCH (q)-[:GENERATE_BY]->(d:Doc)
                    RETURN DISTINCT d.id AS id
                    """,
                    question=query['question'],
                    database_="neo4j"
                )
                for record in records:
                    related_docs.add(record['id'])
        
        return related_docs

    def qag_recall_documents(
        self, 
        input_text: str, 
        doc_top_k: Optional[int] = None,
        query_top_k: Optional[int] = None,
        graph_depth: Optional[int] = None,
    ) -> Set[str]:
        """
        PoKE 深度召回策略。

        结合多种召回策略：
        1. 向量检索召回文档
        2. 相似查询召回
        3. 关键词生成与过滤
        4. 图数据库多跳遍历召回

        Args:
            input_text: 输入查询文本
            doc_top_k: 文档检索的 top_k 值
            query_top_k: 查询检索的 top_k 值
            graph_depth: 图遍历深度

        Returns:
            召回的文档 ID 集合
        """
        doc_top_k = doc_top_k or self.retrieval_config.doc_top_k
        query_top_k = query_top_k or self.retrieval_config.query_top_k
        graph_depth = graph_depth or self.retrieval_config.graph_depth
        
        # 文档向量检索召回
        doc_results = self.document_retrieval(input_text, top_k=doc_top_k)

        # 相似查询检索召回
        query_results = self.query_retrieval(input_text, top_k=query_top_k)

        # 收集检索到的文档 ID（Milvus 返回格式：{'id': ..., 'entity': {...}}）
        # 使用列表保持顺序
        recall_ids = []
        seen = set()
        for doc in doc_results:
            doc_id = doc['entity']['id']
            if doc_id not in seen:
                recall_ids.append(doc_id)
                seen.add(doc_id)

        # 暂时禁用关键词过滤
        # # 获取文档内容用于关键词生成
        # if recall_ids:
        #     text_list = self.milvus_client.get(
        #         collection_name=self.milvus_config.chunk_collection,
        #         ids=list(recall_ids),
        #         output_fields=['id', 'content']
        #     )

        #     # 生成关键词
        #     keyword_list = []
        #     for doc in text_list:
        #         keyword_list.extend(self.keyword_generation(doc['content']))

        #     # 使用关键词过滤
        #     recall_ids = self.keywords_filter(recall_ids, keyword_list)

        # 图数据库多跳遍历
        with GraphDatabase.driver(
            uri=self.neo4j_config.uri,
            auth=self.neo4j_config.auth
        ) as driver:
            # 文档多跳遍历：查找与召回文档 1-2 跳内相连的其他文档
            for doc_id in list(recall_ids):
                records, _, _ = driver.execute_query(
                    """
                    MATCH (d:Doc {id: $id})
                    MATCH (d)-[*1..2]-(d2:Doc)
                    RETURN DISTINCT d2.id AS id
                    """,
                    id=doc_id,
                    database_="neo4j"
                )
                for record in records:
                    if record['id'] not in seen:
                        recall_ids.append(record['id'])
                        seen.add(record['id'])

            # 查询多跳遍历：沿 RELATED 关系遍历查询图
            for query in query_results:
                records, _, _ = driver.execute_query(
                    f"""
                    MATCH (start_q:Query {{question: $question}})
                    MATCH path=(start_q)-[:RELATED*0..{graph_depth}]-(q:Query)
                    WHERE ALL(r IN relationships(path) WHERE type(r)="RELATED")
                    MATCH (q)-[:GENERATE_BY]->(d:Doc)
                    RETURN DISTINCT d.id AS id
                    """,
                    question=query['question'],
                    database_="neo4j"
                )
                for record in records:
                    if record['id'] not in seen:
                        recall_ids.append(record['id'])
                        seen.add(record['id'])

        return recall_ids

    def reachability_test(self, queries: List[Dict[str, Any]], target_ids: Set[str]) -> float:
        """
        可达性测试。

        测试源查询节点与目标文档节点之间是否存在路径连接。

        Args:
            queries: 源查询字典列表，每个字典包含 question 和 id 字段
            target_ids: 目标文档 ID 集合

        Returns:
            可达性比率（可达的文档数/总文档数）
        """
        reached_ids = set()
        
        with GraphDatabase.driver(
            self.neo4j_config.uri, 
            auth=self.neo4j_config.auth
        ) as driver:
            for query in queries:
                for target_id in target_ids:
                    if target_id in reached_ids:
                        continue
                    
                    # 查找查询节点与文档节点之间的最短路径
                    records, _, _ = driver.execute_query(
                        """
                        MATCH (a:Query {question: $question}), (b:Doc {id: $id})
                        OPTIONAL MATCH p = shortestPath((a)-[*]-(b))
                        WITH p, CASE WHEN p IS NULL THEN null ELSE length(p) END AS path_length
                        RETURN p, path_length
                        """,
                        question=query["question"],
                        id=target_id,
                        database_="neo4j"
                    )
                    if records:
                        reached_ids.add(target_id)
        
        return len(reached_ids) / len(target_ids) if target_ids else 0.0

    def recall_score(
        self, 
        recall_ids: Set[str], 
        result_ids: Set[str]
    ) -> tuple[float, Set[str], Set[str]]:
        """
        召回率计算。

        计算召回的文档 ID 中与结果 ID 重合的比例。

        Args:
            recall_ids: 召回的文档 ID 集合
            result_ids: 结果文档 ID 集合

        Returns:
            (召回率分数，成功召回的 ID 集合，未成功召回的 ID 集合)
        """
        if not result_ids:
            return 0.0, set(), set()
            
        success_recall_ids = recall_ids & result_ids
        unsuccess_recall_ids = result_ids - recall_ids
        
        recall_rate = len(success_recall_ids) / len(result_ids)
        return recall_rate, success_recall_ids, unsuccess_recall_ids

    def precision_score(
        self, 
        recall_ids: Set[str], 
        result_ids: Set[str]
    ) -> tuple[float, Set[str], Set[str]]:
        """
        准确率计算。

        计算召回的文档 ID 中与结果 ID 重合的比例（相对于召回结果）。

        Args:
            recall_ids: 召回的文档 ID 集合
            result_ids: 结果文档 ID 集合

        Returns:
            (准确率分数，可用的召回 ID 集合，不可用的召回 ID 集合)
        """
        if not recall_ids:
            return 0.0, set(), set()
            
        available_recall_ids = recall_ids & result_ids
        unavailable_recall_ids = recall_ids - result_ids
        
        precision = len(available_recall_ids) / len(recall_ids)
        return precision, available_recall_ids, unavailable_recall_ids

    @classmethod
    def from_configs(
        cls,
        llm_provider: BaseLLMProvider,
        embedding_provider: BaseLLMProvider,
    ) -> "QueryDocumentRetrieval":
        """
        从默认配置创建检索器实例。
        
        Args:
            llm_provider: LLM 提供者实例
            embedding_provider: 嵌入提供者实例
            
        Returns:
            QueryDocumentRetrieval 实例
        """
        return cls(
            llm_provider=llm_provider,
            embedding_provider=embedding_provider,
        )


if __name__ == "__main__":
    # 示例：使用 Ollama 提供者
    from .llm_provider import get_llm_provider, get_embedding_provider

    # 创建提供者
    llm_provider = get_llm_provider("ollama", model="qwen3:1.7b")
    embedding_provider = get_embedding_provider("ollama", model="bge-m3:latest")
    
    # 初始化检索器
    qdr = QueryDocumentRetrieval(
        llm_provider=llm_provider,
        embedding_provider=embedding_provider,
    )
    
    # 测试关键词生成
    keywords = qdr.keyword_generation("What is the capital of China?")
    print(f"Keywords: {keywords}")
