# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

语言使用中文

## 项目概述

**QAG-RAG** — 基于知识图谱增强的检索增强生成（RAG）系统，面向生物医学问答任务，在 **BioASQ** 数据集上评估。系统从语料中生成子查询并构建知识图谱，结合 Milvus 向量检索与 Neo4j 图遍历实现混合检索。

## 架构设计

```
QAG-RAG/
├── core/                    # 核心库
│   ├── llm_provider.py      # LLM 抽象层：OllamaProvider + OpenAIProvider（文本生成 + 嵌入）
│   ├── prompt.py            # 分隔符常量 + Prompt 模板（实体/关系抽取、查询生成、摘要）
│   ├── retrieval.py         # QueryDocumentRetrieval：Milvus + Neo4j 混合检索（PoKE召回、深度召回、关键词过滤）
│   ├── reranker.py          # EntityReranker：两阶段重排序（向量余弦相似度 + spaCy 实体语义匹配）
│   └── utils.py             # 工具函数：query_generation、entity_extractor、tuple_extractor、query_summary
├── pipelines/               # 数据处理流水线
│   ├── generate_queries.py  # 段落 → LLM 子查询生成 → Milvus 插入（bioasq_query 集合）
│   └── build_graph.py       # Milvus → Neo4j 知识图谱构建（Query/Doc 节点，RELATED/GENERATE_BY 边）
├── config/                  # YAML 实验配置（需手动加载）
│   └── bioasq_openai_qag.yaml
├── docker-compose.yaml      # Neo4j（4 实例）、Ollama、Milvus GPU、OpenWebUI、Etcd、MinIO
├── demo/                    # 空 — 预留 demo 脚本
├── experiments/             # 空 — 预留实验脚本
└── notebooks/               # 空 — 预留 Jupyter 笔记本
```

### 关键设计模式

- **LLM 后端无关**：`BaseLLMProvider` 抽象类，`OllamaProvider` 和 `OpenAIProvider` 两种实现。工厂函数 `get_llm_provider(type, **kwargs)` 选择后端，均支持 `generate()` 和 `embed()`。
- **基于分隔符的结构化抽取**：Prompt 使用自定义 XML 分隔符（`<q>`、`<entity>`、`<name>`、`<type>`、`<description>`、`<relation>`、`<tuple>`），通过正则表达式解析。
- **双向量数据库架构**：Milvus 负责向量相似度搜索（query collection + chunk collection），Neo4j 负责图遍历（多跳 RELATED/GENERATE_BY 关系）。
- **QAG 召回策略**：`qag_recall_documents()` 结合四种策略——向量文档检索、相似查询召回、关键词过滤（KeyBERT，已禁用）、图多跳遍历。
- **实体感知重排序**：`EntityReranker` 融合向量余弦相似度（v1）与实体嵌入交叉相似度（v2），加权公式 `v3 = α*v1 + (1-α)*v2`。

## 开发环境

### 缺失的 `config.py` 模块

流水线代码引用 `from config import milvus_config, neo4j_config, retrieval_config, llm_config, embedding_config`，但**项目根目录不存在 `config.py`**。需在 `core/` 同级目录创建该文件，定义以下数据类对象（字段从代码调用处推导）：

```python
from dataclasses import dataclass

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

milvus_config = MilvusConfig()
neo4j_config = Neo4jConfig()
retrieval_config = RetrievalConfig()
llm_config = LLMConfig()
embedding_config = EmbeddingConfig()
```

### `requirements.txt` 为空

需手动安装以下依赖（从代码导入推导）：
`openai`, `ollama`, `pymilvus`, `neo4j`, `scikit-learn`, `numpy`, `spacy`, `keybert`, `datasets`, `tqdm`

### 运行流水线

```bash
# 在项目根目录：
python -m pipelines.generate_queries   # 从语料生成查询 → Milvus
python -m pipelines.build_graph        # 构建知识图谱：Milvus → Neo4j
```

两个流水线都做了 `sys.path.insert(0, '..')` 以从根目录导入模块，因此必须从 `pipelines/` 目录运行或用 `-m` 从根目录运行。

### Docker 服务

```bash
docker compose up -d                  # 启动所有服务
```

服务端点：
- **Neo4j（主）**：HTTP `localhost:17474`，Bolt `localhost:17687`（用户 `neo4j`，密码 `neo4j123`）
- **Neo4j（deepseek）**：HTTP `localhost:17475`，Bolt `localhost:17688`
- **Neo4j（bright qwen）**：HTTP `localhost:17476`，Bolt `localhost:17689`
- **Neo4j（test）**：HTTP `localhost:17477`，Bolt `localhost:17690`
- **Milvus**：gRPC `localhost:19530`，web UI `localhost:9091`
- **Ollama**：`localhost:11444`（本地 Ollama 在 `11434`）
- **OpenWebUI**：`localhost:18181`

### 数据集路径

语料从 `../data/rag_mini_bioasq_corpus` 加载（相对于 `pipelines/`）。预期格式为 HuggingFace `Dataset`，`passages` 包含 `id` 和 `passage` 字段。

## 注意事项

- `config/bioasq_openai_qag.yaml` 包含明文的 SiliconFlow API Key，应迁移至 `.env` 或环境变量
- `.env` 文件存在但为空，用于存放密钥
- `generate_queries.py` 中的 `RECOVER` 常量和 `MODE = "fix"` 用于中断恢复和重试失败 ID
- `build_graph.py` 的 `skip_batches = 366` 用于断点续建
- 代码注释主要为中文，Prompt 和文档字符串为英文
- 无 `tests/` 目录、无 `pyproject.toml`、无 `setup.py`，以脚本形式运行
