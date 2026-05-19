"""
配置模块。

提供 LLM、Embedding、Milvus、Neo4j 和检索参数的数据类定义。
优先从环境变量加载，未设置时使用默认值。
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# 加载项目根目录的 .env 文件
_dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
if os.path.exists(_dotenv_path):
    load_dotenv(_dotenv_path)
else:
    load_dotenv()


@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: str = ""
    base_url: str | None = None
    ollama_host: str | None = None


@dataclass
class EmbeddingConfig:
    provider: str = "openai"
    model: str = "text-embedding-3-small"
    api_key: str = ""
    base_url: str | None = None
    ollama_host: str | None = None
    dimension: int = 1024


@dataclass
class MilvusConfig:
    uri: str = "http://localhost:19530"
    token: str = "root:Milvus"
    db_name: str = "rag_mini_bioasq"
    query_collection: str = "bioasq_query"
    chunk_collection: str = "native_rag"


@dataclass
class Neo4jConfig:
    uri: str = "bolt://localhost:7687"
    auth: tuple = ("neo4j", "neo4j123")


@dataclass
class RetrievalConfig:
    merge_threshold: float = 0.9
    related_threshold: float = 0.7
    limite_num_q: int = 6
    limite_related_docs: int = 30
    graph_depth: int = 2
    doc_top_k: int = 5
    query_top_k: int = 5
    filter_threshold: float = 0.6  # tau_f: score below which chunks are pruned


def _load_from_env():
    """从 .env 环境变量加载配置。"""
    llm_config = LLMConfig(
        provider=os.getenv("LLM_PROVIDER", "openai"),
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("LLM_API_KEY", ""),
        base_url=os.getenv("LLM_BASE_URL", None),
        ollama_host=os.getenv("OLLAMA_HOST", None),
    )

    embedding_config = EmbeddingConfig(
        provider=os.getenv("EMBEDDING_PROVIDER", "openai"),
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key=os.getenv("EMBEDDING_API_KEY", ""),
        base_url=os.getenv("EMBEDDING_BASE_URL", None),
        ollama_host=os.getenv("OLLAMA_HOST", None),
        dimension=int(os.getenv("EMBEDDING_DIMENSION", "1024")),
    )

    milvus_config = MilvusConfig(
        uri=os.getenv("MILVUS_URI", "http://localhost:19530"),
        token=os.getenv("MILVUS_TOKEN", "root:Milvus"),
        db_name=os.getenv("MILVUS_DB_NAME", "rag_mini_bioasq"),
        query_collection=os.getenv("MILVUS_QUERY_COLLECTION", "bioasq_query"),
        chunk_collection=os.getenv("MILVUS_CHUNK_COLLECTION", "native_rag"),
    )

    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "neo4j123")
    neo4j_config = Neo4jConfig(
        uri=neo4j_uri,
        auth=(neo4j_username, neo4j_password),
    )

    retrieval_config = RetrievalConfig(
        merge_threshold=float(os.getenv("MERGE_THRESHOLD", "0.9")),
        related_threshold=float(os.getenv("RELATED_THRESHOLD", "0.7")),
        limite_num_q=int(os.getenv("LIMITE_NUM_Q", "6")),
        limite_related_docs=int(os.getenv("LIMITE_RELATED_DOCS", "30")),
        graph_depth=int(os.getenv("GRAPH_DEPTH", "2")),
        doc_top_k=int(os.getenv("DOC_TOP_K", "5")),
        query_top_k=int(os.getenv("QUERY_TOP_K", "5")),
    )

    return llm_config, embedding_config, milvus_config, neo4j_config, retrieval_config


# 默认从环境变量加载
llm_config, embedding_config, milvus_config, neo4j_config, retrieval_config = _load_from_env()
