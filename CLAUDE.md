# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

语言使用中文

## 项目概述

**QAG-RAG** — 基于知识图谱增强的检索增强生成（RAG）系统，面向生物医学问答任务，在 **BioASQ** 数据集上评估。系统从语料中生成子查询并构建知识图谱，结合 Milvus 向量检索与 Neo4j 图遍历实现混合检索。

## 架构设计

```
QAG-RAG/
├── qag-rag/                    # 核心代码（注意目录名含连字符，无法作为 Python 包导入）
│   ├── core/                   # 核心库
│   │   ├── llm_provider.py     # LLM 抽象层：OllamaProvider + OpenAIProvider（文本生成 + 嵌入）
│   │   ├── prompt.py           # 分隔符常量 + Prompt 模板
│   │   ├── retrieval.py        # QueryDocumentRetrieval：Milvus + Neo4j 混合检索
│   │   ├── reranker.py         # EntityReranker：两阶段重排序（向量余弦 + spaCy 实体语义匹配）
│   │   └── utils.py            # 工具函数：query_generation、entity_extractor、tuple_extractor、query_summary
│   └── pipelines/              # 数据处理流水线
│       ├── generate_queries.py # 段落 → LLM 子查询生成 → Milvus 插入
│       └── build_graph.py      # Milvus → Neo4j 知识图谱构建
├── bioasq_openai_qag.yaml      # YAML 实验配置（原 config/ 目录已移除）
├── docker-compose.yaml         # Neo4j（4 实例）+ Milvus GPU（standalone + etcd + minio + attu）
├── data/                       # 语料数据集（HuggingFace Dataset 格式）
├── qag-rag-ui/                 # 空 — 预留前端目录
└── docs/                       # 空 — 预留文档目录
```

### 关键设计模式

- **LLM 后端无关**：`BaseLLMProvider` 抽象类，`OllamaProvider` 和 `OpenAIProvider` 两种实现。工厂函数 `get_llm_provider` / `get_embedding_provider` 选择后端。
- **基于分隔符的结构化抽取**：Prompt 使用自定义 XML 分隔符（`<q>`、`<entity>`、`<name>`、`<type>`、`<description>`、`<relation>`、`<tuple>`），通过正则表达式解析。
- **双向量数据库架构**：Milvus 负责向量相似度搜索（query collection + chunk collection），Neo4j 负责图遍历（多跳 RELATED/GENERATE_BY 关系）。
- **QAG 召回策略**：`qag_recall_documents()` 结合四种策略——向量文档检索、相似查询召回、关键词过滤（KeyBERT，已禁用）、图多跳遍历。
- **实体感知重排序**：`EntityReranker` 融合向量余弦相似度（v1）与实体嵌入交叉相似度（v2），加权公式 `v3 = α*v1 + (1-α)*v2`。

## 开发环境

### 依赖安装

`requirements.txt` 为空，需手动安装以下依赖（从代码导入推导）：

```bash
pip install openai ollama pymilvus neo4j scikit-learn numpy spacy keybert datasets tqdm
python -m spacy download en_core_web_sm
```

### 缺失的 `config.py` 模块

`qag-rag/core/retrieval.py` 与 `qag-rag/pipelines/*.py` 均引用 `from config import milvus_config, neo4j_config, retrieval_config, llm_config, embedding_config`，但**项目根目录不存在 `config.py`**。需在根目录创建该文件，内容参考如下：

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

### 路径与导入问题（重要）

`core/` 与 `pipelines/` 近期从根目录移入 `qag-rag/` 子目录，但脚本中的相对路径未同步更新，导致以下问题：

1. **包名含连字符**：`qag-rag` 含连字符，无法通过 `python -m qag-rag.pipelines.xxx` 运行。
2. **`sys.path.insert(0, '..')` 失效**：`qag-rag/pipelines/*.py` 使用 `..` 期望导入根目录的 `config` 模块；现在 `..` 指向 `qag-rag/`，导致 `config` 无法解析。
3. **数据集路径失效**：`generate_queries.py` 中 `load_from_disk("../data/rag_mini_bioasq_corpus")` 以 cwd 为基准；在当前的嵌套目录结构下，`../data/` 不再指向根目录的 `data/`。

**当前可行的运行方式**：
- 方案 A：在 `qag-rag/` 内新建 `config.py`，并将 `data/` 复制或软链接到 `qag-rag/data/`，然后从 `qag-rag/pipelines/` 运行：
  ```bash
  cd qag-rag/pipelines
  python generate_queries.py
  ```
- 方案 B：在根目录运行，手动修正 `PYTHONPATH` 并将数据路径改为绝对路径或调整 cwd。

### 运行流水线

```bash
# 方案 A（需先在 qag-rag/ 内准备好 config.py 与 data/）：
cd qag-rag/pipelines
python generate_queries.py   # 从语料生成查询 → Milvus
python build_graph.py        # 构建知识图谱：Milvus → Neo4j
```

### Docker 服务

```bash
docker compose up -d
```

服务端点：
- **Neo4j（主）**：HTTP `localhost:17474`，Bolt `localhost:17687`（用户 `neo4j`，密码 `neo4j123`）
- **Neo4j（deepseek）**：HTTP `localhost:17475`，Bolt `localhost:17688`
- **Neo4j（bright qwen3-1.7b）**：HTTP `localhost:17476`，Bolt `localhost:17689`
- **Neo4j（test）**：HTTP `localhost:17477`，Bolt `localhost:17690`
- **Milvus**：gRPC `localhost:19530`，web UI `localhost:9091`
- **Attu（Milvus UI）**：`localhost:18000`

### 数据集

语料预期为 HuggingFace `Dataset`，`passages` 包含 `id` 和 `passage` 字段。原始路径在 `data/rag_mini_bioasq_corpus`（根目录）。

## 注意事项

- `bioasq_openai_qag.yaml` 包含明文的 SiliconFlow API Key，应迁移至 `.env` 或环境变量。`.env.example` 已提供模板。
- `.env` 文件存在但为空，用于存放密钥。
- `generate_queries.py` 中的 `RECOVER` 常量和 `MODE = "fix"` 用于中断恢复和重试失败 ID。
- `build_graph.py` 的 `skip_batches = 366` 用于断点续建。
- 代码注释主要为中文，Prompt 和文档字符串为英文。
- 无 `tests/` 目录、无 `pyproject.toml`、无 `setup.py`，以脚本形式运行。
