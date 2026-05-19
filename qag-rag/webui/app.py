"""
QAG-RAG Web UI 后端。

FastAPI 服务：
- POST /api/retrieve（SSE 流式）：分阶段推送子图、树、答案
- 静态文件：托管前端可视化页面
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

# 使 qag-rag/ 目录可 import（ pipelines/、core/ 等变为顶层包）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pipelines.qag_retrieval import (
    filter_retrieval_tree,
    hierarchical_summarize,
    load_pipeline_config,
    retrieve_subgraph,
    subgraph_to_tree,
)
from core.llm_provider import get_llm_provider, get_embedding_provider


app = FastAPI(title="QAG-RAG Web UI", version="1.0")

# 静态文件
static_dir = Path(__file__).resolve().parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ---------------------------------------------------------------------------
# API 模型
# ---------------------------------------------------------------------------

class RetrieveRequest(BaseModel):
    question: str
    config: str = "config.yml"
    query_top_k: Optional[int] = None
    graph_depth: Optional[int] = None
    doc_top_k: Optional[int] = None
    related_threshold: Optional[float] = None
    filter_enabled: bool = True
    filter_threshold: Optional[float] = None


# ---------------------------------------------------------------------------
# API 端点
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = static_dir / "index.html"
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>QAG-RAG Web UI</h1><p>index.html not found</p>")


def _sse_event(data: dict) -> str:
    """构造 SSE 事件字符串。"""
    return f"event: {data['type']}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/api/retrieve")
async def api_retrieve(req: RetrieveRequest):
    """SSE 流式端点：分阶段推送子图、树、答案。"""

    async def event_stream():
        t0 = time.time()

        try:
            yield _sse_event({"type": "log", "msg": f"Loading config: {req.config}", "elapsed": round(time.time() - t0, 1)})
            cfg = load_pipeline_config(req.config)
            llm_cfg = cfg["llm"]
            emb_cfg = cfg["embedding"]
            milvus_cfg = cfg["milvus"]
            neo4j_cfg = cfg["neo4j"]
            ret_cfg = cfg["retrieval"]

            # 应用前端传入的检索参数覆盖
            if req.query_top_k is not None:
                ret_cfg.query_top_k = req.query_top_k
            if req.graph_depth is not None:
                ret_cfg.graph_depth = req.graph_depth
            if req.doc_top_k is not None:
                ret_cfg.doc_top_k = req.doc_top_k
            if req.related_threshold is not None:
                ret_cfg.related_threshold = req.related_threshold

            yield _sse_event({"type": "log", "msg": f"Retrieval params: query_top_k={ret_cfg.query_top_k}, "
                           f"graph_depth={ret_cfg.graph_depth}, "
                           f"doc_top_k={ret_cfg.doc_top_k}, "
                           f"related_threshold={ret_cfg.related_threshold}", "elapsed": round(time.time() - t0, 1)})

            yield _sse_event({"type": "log", "msg": f"LLM: {llm_cfg.provider} / {llm_cfg.model}", "elapsed": round(time.time() - t0, 1)})
            yield _sse_event({"type": "log", "msg": f"Embedding: {emb_cfg.provider} / {emb_cfg.model}", "elapsed": round(time.time() - t0, 1)})

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

            # Step 1: Subgraph retrieval
            yield _sse_event({"type": "log", "msg": "Starting subgraph retrieval...", "elapsed": round(time.time() - t0, 1)})
            subgraph = retrieve_subgraph(
                question=req.question,
                llm_provider=llm_provider,
                embedding_provider=embedding_provider,
                milvus_config=milvus_cfg,
                neo4j_config=neo4j_cfg,
                retrieval_config=ret_cfg,
            )
            yield _sse_event({"type": "log", "msg": f"Subgraph complete: {len(subgraph['root_queries'])} root queries, "
                           f"{len(subgraph['nodes']['Query'])} Query nodes, "
                           f"{len(subgraph['nodes']['Doc'])} Doc nodes", "elapsed": round(time.time() - t0, 1)})

            # Step 1.5: Feature-based filtering (论文 Section 3.3)
            if req.filter_enabled:
                threshold = req.filter_threshold if req.filter_threshold is not None else ret_cfg.filter_threshold
                yield _sse_event({"type": "log", "msg": f"Feature filter: threshold={threshold}", "elapsed": round(time.time() - t0, 1)})
                subgraph = filter_retrieval_tree(
                    subgraph, target_question=req.question,
                    embedding_provider=embedding_provider,
                    threshold=threshold,
                )
                fi = subgraph.get('_filter_info', {})
                yield _sse_event({"type": "log", "msg": f"Filter complete: kept {fi.get('kept_docs', '?')}/{fi.get('original_docs', '?')} Doc "
                               f"(pruned {fi.get('pruned_docs', '?')} Doc, {fi.get('pruned_queries', 0)} isolated Query)", "elapsed": round(time.time() - t0, 1)})

            # 推送子图
            yield _sse_event({
                "type": "subgraph",
                "subgraph": subgraph,
                "elapsed": round(time.time() - t0, 1),
            })

            # Step 2: Tree conversion
            yield _sse_event({"type": "log", "msg": "Converting to tree structure...", "elapsed": round(time.time() - t0, 1)})
            trees = subgraph_to_tree(
                subgraph, target_question=req.question, max_depth=ret_cfg.graph_depth - 1
            )
            yield _sse_event({"type": "log", "msg": f"Tree conversion complete: {len(trees)} trees", "elapsed": round(time.time() - t0, 1)})

            # 推送树
            yield _sse_event({
                "type": "tree",
                "tree": trees,
                "graph_depth": ret_cfg.graph_depth,
                "elapsed": round(time.time() - t0, 1),
            })

            # Step 3: Hierarchical summarization (stream each layer result)
            yield _sse_event({"type": "log", "msg": "Starting hierarchical summarization...", "elapsed": round(time.time() - t0, 1)})

            queue: asyncio.Queue = asyncio.Queue()

            def on_layer_summary(d, s):
                queue.put_nowait({
                    "type": "log",
                    "msg": f"[depth={d}] {s[:500]}...",
                    "elapsed": round(time.time() - t0, 1),
                })

            async def run_summarize():
                answer = hierarchical_summarize(
                    trees=trees,
                    target_question=req.question,
                    llm_provider=llm_provider,
                    on_layer_summary=on_layer_summary,
                )
                await queue.put({
                    "type": "answer",
                    "answer": answer,
                    "elapsed": round(time.time() - t0, 1),
                })

            task = asyncio.create_task(run_summarize())
            answer = None
            while True:
                event = await queue.get()
                yield _sse_event(event)
                if event["type"] == "answer":
                    answer = event["answer"]
                    break

            yield _sse_event({"type": "log", "msg": "Summarization complete", "elapsed": round(time.time() - t0, 1)})
            yield _sse_event({"type": "done", "elapsed": round(time.time() - t0, 1)})

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            yield _sse_event({"type": "log", "msg": f"Error: {e}", "elapsed": round(time.time() - t0, 1)})
            yield _sse_event({"type": "error", "error": str(e), "detail": tb, "elapsed": round(time.time() - t0, 1)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# 启动入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
