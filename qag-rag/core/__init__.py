"""
QAG-RAG 核心模块。

包含 LLM 提供者、工具函数、Prompt 模板和检索功能。
"""

from .llm_provider import BaseLLMProvider, OllamaProvider, OpenAIProvider, get_llm_provider, get_embedding_provider
from .utils import get_embedding, query_generation, entity_extractor, tuple_extractor, query_summary
from .prompt import PROMPT
from .retrieval import QueryDocumentRetrieval

__all__ = [
    # LLM Provider
    "BaseLLMProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "get_llm_provider",
    "get_embedding_provider",
    # Utils
    "get_embedding",
    "query_generation",
    "entity_extractor",
    "tuple_extractor",
    "query_summary",
    # Prompt
    "PROMPT",
    # Retrieval
    "QueryDocumentRetrieval",
]
