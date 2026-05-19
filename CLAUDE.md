# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

语言使用中文

## 项目概述

**QAG-RAG** — 基于知识图谱增强的检索辅助生成（RAG）系统，面向生物医学问答任务，在 **BioASQ** 数据集上评估。系统从语料中生成子查询并构建知识图谱，结合 Milvus 向量检索与 Neo4j 图遍历实现混合检索。

## 架构设计

```
QAG-RAG/
├── qag-rag/                    # 核心代码（注意目录名含连字符，无法作为 Python 包导入）
│   ├── config.py               # 配置模块：LLM/Embedding/Milvus/Neo4j/Retrieval 数据类
│   ├── core/                   # 核心库
│   │   ├── llm_provider.py     # LLM 抽象层：OllamaProvider + OpenAIProvider
│   │   ├── prompt.py           # 分隔符常量 + Prompt 模板（含 Hierarchical_Summary）
│   │   ├── retrieval.py        # QueryDocumentRetrieval：Milvus + Neo4j 混合检索
│   │   ├── reranker.py         # EntityReranker：两阶段重排序
│   │   ├── utils.py            # 工具函数：query_generation、entity_extractor 等
│   │   └── generate_queries.py # 查询生成工具函数
│   ├── pipelines/              # 数据处理流水线
│   │   ├── config.yml          # 流水线配置（检索参数、LLM/Embedding 模型选择）
│   │   ├── build_graph.py      # Milvus → Neo4j 知识图谱构建
│   │   ├── qag_retrieval.py    # QAG 检索流水线：子图检索 + 自底向上汇总
│   │   ├── qag_response.py     # 空 — 预留
│   │   └── test/               # 单元测试
│   │       └── test_qag_retrieval.py
│   ├── api/                    # 空 — 预留目录
│   └── tools/                  # 空 — 预留目录
├── bioasq_openai_qag.yaml      # YAML 实验配置
├── docker-compose.yaml         # Neo4j + Milvus GPU
├── data/                       # 语料数据集 + QAG CSV 结果
├── docs/notebooks/             # reranker.ipynb
└── qag-rag-ui/                 # 空 — 预留前端目录
```

### 关键设计模式

- **LLM 后端无关**：`BaseLLMProvider` 抽象类，`OllamaProvider` 和 `OpenAIProvider` 两种实现。工厂函数 `get_llm_provider` / `get_embedding_provider` 选择后端。
- **基于分隔符的结构化抽取**：Prompt 使用自定义 XML 分隔符（`<q>`、`<entity>`、`<name>`、`<type>`、`<description>`、`<relation>`、`<tuple>`），通过正则表达式解析。
- **双向量数据库架构**：Milvus 负责向量相似度搜索（query collection + chunk collection），Neo4j 负责图遍历（多跳 RELATED/GENERATE_BY 关系）。
- **QAG 召回策略**：`qag_recall_documents()` 结合四种策略——向量文档检索、相似查询召回、关键词过滤（KeyBERT，已禁用）、图多跳遍历。
- **实体感知重排序**：`EntityReranker` 融合向量余弦相似度（v1）与实体嵌入交叉相似度（v2），加权公式 `v3 = α*v1 + (1-α)*v2`。
- **OpenAI 兼容 API 的 `enable_thinking` 兼容性**：`OpenAIProvider.generate()` 默认发送 `extra_body={"enable_thinking": False}`，但不支持该参数的模型（如 GLM-Z1）会报错。调用方可传 `skip_extra_body=True` 绕过。`utils.query_generation()` 在调用 OpenAI provider 时已自动设置 `skip_extra_body=True`。
- **配置优先级**：`.env` 环境变量 > `pipelines/config.yml` > `config.py` 硬编码默认值。

### QAG 检索流水线（`qag_retrieval.py`）

完整的子图检索与自底向上汇总流程：

1. **Milvus 召回**：输入问题 q → 向量搜索 `query_collection` → 得到相似问题集合 Q（top-k）
2. **Neo4j 图遍历**：以 Q 中每个问题为起点，遍历 `RELATED` 关系（多跳）收集 Query 节点，遍历 `GENERATE_BY` 关系收集 Doc 节点
3. **子图构建**：返回 `{root_queries, nodes: {Query, Doc}, relationships}` 字典
4. **子图 → 树**：按 RELATED 关系层级展开为多棵根树，每棵树节点包含 question/docs/children
5. **自底向上汇总**：使用 `Hierarchical_Summary` prompt，从叶子到根逐层 LLM 汇总，Target Question 始终为原始问题 q

### Web UI（`webui/`）

- `app.py`：FastAPI 后端，提供 `POST /api/retrieve` 接口
- `static/index.html`：前端可视化（vis-network.js），展示子图结构、树形结构和最终答案

### `config.py` 模块（`qag-rag/config.py`）

提供以下数据类和单例实例：

| 数据类 | 实例 | 来源 |
|---|---|---|
| `LLMConfig` | `llm_config` | `.env` 环境变量 |
| `EmbeddingConfig` | `embedding_config` | `.env` 环境变量 |
| `MilvusConfig` | `milvus_config` | `.env` 环境变量 |
| `Neo4jConfig` | `neo4j_config` | `.env` 环境变量 |
| `RetrievalConfig` | `retrieval_config` | `.env` 环境变量 |

所有配置通过 `python-dotenv` 加载 `.env` 文件，未设置时使用硬编码默认值。

### 路径与导入

`qag-rag/` 目录名含连字符，无法作为 Python 包导入。各脚本通过 `sys.path.insert(0, '..')` 或 `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))` 将 `qag-rag/` 加入 Python 路径，使 `config`、`core/`、`pipelines/` 可作为顶层模块导入。

### 开发环境

#### 依赖安装

```bash
pip install openai ollama pymilvus neo4j scikit-learn numpy spacy keybert datasets tqdm dashscope python-dotenv fastapi uvicorn pydantic vis-network
python -m spacy download en_core_web_sm
```

#### 运行流水线

```bash
# 1. 启动 Docker 服务（Milvus + Neo4j）
docker compose up -d

# 2. 从语料生成查询 → Milvus（如已有数据可跳过）
cd qag-rag/pipelines
python build_graph.py        # 构建知识图谱：Milvus → Neo4j

# 3. 检索流水线测试
cd qag-rag/pipelines
python test/test_qag_retrieval.py

# 4. Web UI
cd qag-rag/webui
python app.py                # http://localhost:8000
```

#### Docker 服务

```bash
docker compose up -d
```

服务端点：
- **Neo4j（主）**：HTTP `localhost:17474`，Bolt `localhost:17687`
- **Neo4j（deepseek）**：HTTP `localhost:17475`，Bolt `localhost:17688`
- **Neo4j（test）**：HTTP `localhost:17477`，Bolt `localhost:17690`
- **Milvus**：gRPC `localhost:19530`，web UI `localhost:9091`
- **Attu（Milvus UI）**：`localhost:18000`

> 注：当前 `.env` 中 `NEO4J_URI=bolt://localhost:17688`，指向 deepseek 实例。

#### 数据集

- **语料**：HuggingFace `Dataset`，`passages` 包含 `id` 和 `passage` 字段。原始路径在 `data/rag_mini_bioasq_corpus`（根目录）。
- **QAG 生成结果**：`data/` 下还包含各模型生成的 QAG CSV 文件（`mini-bioasq_qag_deepseekv3.2.csv`、`mini-bioasq_qag_qwen3_1.7b.csv`、`mini_bioasq_qag_glm_z1_9b.csv`）。

### 配置与密钥管理

- `.env` 文件包含真实 API Key（未提交到 git，已加入 `.gitignore`）。
- `bioasq_openai_qag.yaml` 已清理，不再包含明文密钥。

### 注意事项

- `build_graph.py` 的 `skip_batches = 366` 用于断点续建。
- 代码注释主要为中文，Prompt 和文档字符串为英文。
- 无 `pyproject.toml`、无 `setup.py`，以脚本形式运行。
- **检索前置条件**：运行 `qag_retrieval.py` 或 `webui/app.py` 前，必须先运行 `build_graph.py` 将 Milvus 中的数据导入 Neo4j，否则图数据库为空会导致检索无结果。
- `.gitignore` 中存在拼写错误 `.cluade/`，如需忽略 `.claude/` 目录需额外补正。
