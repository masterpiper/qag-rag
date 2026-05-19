# QAG-RAG: 知识图谱增强检索辅助生成系统

面向生物医学问答的 RAG 系统，在 [BioASQ](https://www.bioasq.org/) 数据集上评估。结合 **Milvus** 向量检索与 **Neo4j** 知识图谱遍历，实现混合召回与自底向上的分层摘要生成。

## 架构概览

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
│  BioASQ     │────▶│  LLM Query       │────▶│   Milvus     │
│  Corpus     │     │  Generation      │     │  (Vectors)   │
└─────────────┘     └──────────────────┘     └──────┬───────┘
                                                     │
┌─────────────┐     ┌──────────────────┐     ┌──────▼───────┐
│  Answer     │◀────│  Hierarchical    │◀────│   Neo4j      │
│  (Output)   │     │  Summarization   │     │  (Graph)     │
└─────────────┘     └──────────────────┘     └──────────────┘
```

### 核心流程

1. **语料入库** — LLM 将 BioASQ 语料段落分解为子查询，嵌入后存入 Milvus（`bioasq_query` collection）
2. **图谱构建** — 从 Milvus 读取数据，在 Neo4j 中创建 `Query` / `Doc` 节点及 `RELATED` / `GENERATE_BY` 边
3. **混合检索** — 输入问题 → Milvus 检索相似查询 → Neo4j 多跳遍历扩展子图 → 特征过滤剪枝 → 树形展开
4. **分层摘要** — 自底向上逐层 LLM 汇总，最终生成连贯回答

## 技术栈

| 组件 | 技术 |
|---|---|
| 向量数据库 | Milvus v2.6.1 (GPU) |
| 图数据库 | Neo4j 5.26 |
| LLM 后端 | OpenAI 兼容 API / Ollama |
| Embedding | BGE-M3 / text-embedding-3-small |
| NER | spaCy (`en_core_web_sm`) |
| Web UI | FastAPI + vis-network.js |
| 数据集 | HuggingFace `rag-mini-bioasq` |

## 快速开始

### 1. 环境准备

```bash
pip install openai ollama pymilvus neo4j scikit-learn numpy spacy keybert datasets tqdm dashscope python-dotenv fastapi uvicorn pydantic
python -m spacy download en_core_web_sm
```

### 2. 配置

创建 `.env` 文件并填写：

```env
# LLM
LLM_PROVIDER=openai
LLM_MODEL=Pro/deepseek-ai/DeepSeek-V3.2
LLM_API_KEY=sk-xxx
LLM_BASE_URL=https://api.siliconflow.cn/v1

# Embedding
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=Pro/BAAI/bge-m3
EMBEDDING_API_KEY=sk-xxx
EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
EMBEDDING_DIMENSION=1024

# Milvus
MILVUS_URI=http://localhost:19530
MILVUS_TOKEN=root:Milvus
MILVUS_DB_NAME=rag_mini_bioasq

# Neo4j
NEO4J_URI=bolt://localhost:17687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=neo4j123
```

### 3. 启动基础设施

```bash
docker compose up -d
```

| 服务 | 端点 |
|---|---|
| Neo4j (主) | HTTP `localhost:17474`, Bolt `localhost:17687` |
| Neo4j (deepseek) | HTTP `localhost:17475`, Bolt `localhost:17688` |
| Neo4j (test) | HTTP `localhost:17477`, Bolt `localhost:17690` |
| Milvus gRPC | `localhost:19530` |
| Milvus Web UI | `localhost:9091` |
| Attu (Milvus UI) | `localhost:18000` |

### 4. 构建知识图谱

```bash
cd qag-rag/pipelines
python build_graph.py
```

> 支持断点续建：修改脚本中的 `skip_batches` 参数跳过已处理批次。

### 5. 运行检索测试

```bash
cd qag-rag/pipelines
python test/test_qag_retrieval.py
```

### 6. Web UI 可视化

```bash
cd qag-rag/webui
python app.py
```

访问 `http://localhost:8000`，界面支持中英文切换，实时展示检索子图、树形结构与最终摘要。

## 项目结构

```
.
├── qag-rag/
│   ├── config.py                 # 配置数据类 (LLM / Embedding / Milvus / Neo4j / Retrieval)
│   ├── core/
│   │   ├── llm_provider.py       # LLM 抽象层: OllamaProvider + OpenAIProvider
│   │   ├── prompt.py             # Prompt 模板与分隔符常量
│   │   ├── retrieval.py          # Milvus + Neo4j 混合检索引擎
│   │   ├── reranker.py           # 实体感知两阶段重排序器
│   │   ├── utils.py              # 工具函数: embedding / query generation / entity extraction
│   │   └── generate_queries.py   # 语料 → Milvus 查询入库流水线
│   ├── pipelines/
│   │   ├── config.yml            # 检索参数默认配置
│   │   ├── build_graph.py        # Milvus → Neo4j 知识图谱构建
│   │   ├── qag_retrieval.py      # 主检索流水线: 子图 → 过滤 → 树 → 分层摘要
│   │   ├── qag_response.py       # (预留)
│   │   └── test/
│   │       └── test_qag_retrieval.py
│   ├── webui/
│   │   ├── app.py                # FastAPI 后端 (SSE 流式推送)
│   │   └── static/index.html     # vis-network.js 前端可视化
│   ├── api/                      # (预留)
│   └── tools/                    # (预留)
├── data/
│   ├── mini-bioasq_qag_deepseekv3.2.csv
│   ├── mini-bioasq_qag_qwen3_1.7b.csv
│   └── mini_bioasq_qag_glm_z1_9b.csv
├── docker-compose.yaml
├── bioasq_openai_qag.yaml        # 实验配置
└── .env                          # 环境变量 (已 gitignore)
```

## 检索策略详解

### 召回阶段 (`qag_recall_documents`)

| 策略 | 说明 |
|---|---|
| 文档向量检索 | Milvus `chunk_collection` 语义检索 |
| 相似查询召回 | Milvus `query_collection` 找到相似问题及其关联文档 |
| 关键词过滤 | KeyBERT 提取关键词，正则匹配过滤（已禁用） |
| 图多跳遍历 | Neo4j `RELATED` / `GENERATE_BY` 边多跳扩展 |

### 重排序 (`EntityReranker`)

融合评分: `v3 = alpha * v1 + (1 - alpha) * v2`

- **v1** — 问题-文档余弦相似度
- **v2** — 实体匹配得分（spaCy NER + 嵌入交叉相似度）
- **alpha** — 默认 0.5

### 分层摘要

按图遍历层级从叶子到根逐层调用 `Hierarchical_Summary` Prompt，上一层摘要作为下一层的上下文输入，最终生成回答。

## 配置优先级

`.env` 环境变量 > `pipelines/config.yml` > `config.py` 硬编码默认值

## 实验结果

`data/` 目录下预置了三种 LLM 生成的 QAG 结果：

| 文件 | 模型 |
|---|---|
| `mini-bioasq_qag_deepseekv3.2.csv` | DeepSeek-V3.2 |
| `mini-bioasq_qag_qwen3_1.7b.csv` | Qwen3 1.7B |
| `mini_bioasq_qag_glm_z1_9b.csv` | GLM-Z1 9B |
