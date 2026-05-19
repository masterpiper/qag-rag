"""
QAG-RAG 检索流水线单元测试与集成测试。

运行方式：
    cd qag-rag/pipelines
    python -m unittest test.test_qag_retrieval

需要 Milvus + Neo4j 运行的测试会自动跳过（如未连接）。
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# 将 qag-rag/ 加入路径，使 pipelines/、core/ 可 import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# 若环境未安装 ollama，先 mock 掉，避免 core.llm_provider 导入失败
if 'ollama' not in sys.modules:
    ollama_mock = MagicMock()
    sys.modules['ollama'] = ollama_mock

from pipelines.qag_retrieval import (
    hierarchical_summarize,
    load_pipeline_config,
    retrieve_subgraph,
    subgraph_to_tree,
)
from core.llm_provider import BaseLLMProvider


# ---------------------------------------------------------------------------
# Mock Provider
# ---------------------------------------------------------------------------

class MockLLMProvider(BaseLLMProvider):
    """用于测试的 Mock LLM Provider。"""

    def __init__(self, responses=None):
        self.responses = responses or {}

    def generate(self, prompt: str, model=None, **kwargs) -> str:
        # 根据 prompt 内容返回固定摘要，便于断言
        if 'Hierarchical_Summary' in prompt or 'Dataset' in prompt:
            return "This is a mock summary for testing."
        return self.responses.get(prompt, "mock response")

    def embed(self, text: str, model=None) -> list[float]:
        # 返回固定维度 1024 的零向量
        return [0.0] * 1024


# ---------------------------------------------------------------------------
# 单元测试
# ---------------------------------------------------------------------------

class TestLoadPipelineConfig(unittest.TestCase):
    """测试配置加载。"""

    def test_load_default_config(self):
        cfg = load_pipeline_config("config.yml")
        self.assertIn('llm', cfg)
        self.assertIn('embedding', cfg)
        self.assertIn('retrieval', cfg)
        self.assertEqual(cfg['retrieval'].graph_depth, 5)  # config.yml 中已改为 5


class TestSubgraphToTree(unittest.TestCase):
    """测试子图转树。"""

    def test_simple_subgraph(self):
        subgraph = {
            "root_queries": [
                {"question": "q1", "distance": 0.9}
            ],
            "nodes": {
                "Query": [
                    {"question": "q1"},
                    {"question": "q2"},
                ],
                "Doc": [
                    {"id": "d1", "text": "doc1"},
                    {"id": "d2", "text": "doc2"},
                ]
            },
            "relationships": [
                {
                    "source": {"label": "Query", "question": "q1"},
                    "target": {"label": "Query", "question": "q2"},
                    "type": "RELATED",
                    "properties": {"sim": 0.85}
                },
                {
                    "source": {"label": "Query", "question": "q1"},
                    "target": {"label": "Doc", "id": "d1"},
                    "type": "GENERATE_BY",
                    "properties": {}
                },
                {
                    "source": {"label": "Query", "question": "q2"},
                    "target": {"label": "Doc", "id": "d2"},
                    "type": "GENERATE_BY",
                    "properties": {}
                }
            ]
        }
        trees = subgraph_to_tree(subgraph)
        self.assertEqual(len(trees), 1)
        root = trees[0]
        self.assertEqual(root["question"], "q1")
        self.assertEqual(len(root["docs"]), 1)
        self.assertEqual(root["docs"][0]["id"], "d1")
        self.assertEqual(len(root["children"]), 1)
        self.assertEqual(root["children"][0]["question"], "q2")

    def test_empty_subgraph(self):
        subgraph = {
            "root_queries": [],
            "nodes": {"Query": [], "Doc": []},
            "relationships": []
        }
        trees = subgraph_to_tree(subgraph)
        self.assertEqual(trees, [])


class TestHierarchicalSummarize(unittest.TestCase):
    """测试自底向上汇总。"""

    def test_single_tree(self):
        tree = {
            "question": "root_q",
            "docs": [{"id": "d1", "text": "text1"}],
            "children": []
        }
        llm = MockLLMProvider()
        answer = hierarchical_summarize([tree], "target question", llm)
        self.assertIn("mock summary", answer.lower())

    def test_multi_tree_merge(self):
        trees = [
            {"question": "q1", "docs": [{"id": "d1", "text": "a"}], "children": []},
            {"question": "q2", "docs": [{"id": "d2", "text": "b"}], "children": []},
        ]
        llm = MockLLMProvider()
        answer = hierarchical_summarize(trees, "target question", llm)
        self.assertIn("mock summary", answer.lower())

    def test_empty_trees(self):
        llm = MockLLMProvider()
        answer = hierarchical_summarize([], "target question", llm)
        self.assertIn("No relevant information", answer)


# ---------------------------------------------------------------------------
# 集成测试（需要 Milvus + Neo4j）
# ---------------------------------------------------------------------------

class TestRetrieveSubgraphIntegration(unittest.TestCase):
    """测试 retrieve_subgraph，需要 Milvus 和 Neo4j 服务。"""

    @classmethod
    def setUpClass(cls):
        # 尝试连接 Milvus 和 Neo4j，任一失败则跳过全部集成测试
        cls.skip_integration = False
        cls.reason = ""

        try:
            from pymilvus import MilvusClient
            client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")
            client.list_collections()
        except Exception as e:
            cls.skip_integration = True
            cls.reason = f"Milvus not available: {e}"
            return

        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "neo4j123"))
            driver.verify_connectivity()
            driver.close()
        except Exception as e:
            cls.skip_integration = True
            cls.reason = f"Neo4j not available: {e}"

    def setUp(self):
        if self.skip_integration:
            self.skipTest(self.reason)

    def test_retrieve_subgraph(self):
        llm = MockLLMProvider()
        emb = MockLLMProvider()
        cfg = load_pipeline_config("config.yml")

        subgraph = retrieve_subgraph(
            question="What are the Yamanaka factors?",
            llm_provider=llm,
            embedding_provider=emb,
            milvus_config=cfg["milvus"],
            neo4j_config=cfg["neo4j"],
            retrieval_config=cfg["retrieval"],
        )

        self.assertIn("root_queries", subgraph)
        self.assertIn("nodes", subgraph)
        self.assertIn("relationships", subgraph)
        self.assertIsInstance(subgraph["root_queries"], list)
        self.assertIsInstance(subgraph["nodes"]["Query"], list)
        self.assertIsInstance(subgraph["nodes"]["Doc"], list)
        self.assertIsInstance(subgraph["relationships"], list)


if __name__ == "__main__":
    unittest.main()
