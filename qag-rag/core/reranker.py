"""
重排序模块 - 基于向量相似度和实体语义匹配的两阶段重排序

融合向量相似度和实体语义匹配进行文档重排序：
1. 计算文档与问题的余弦相似度 → v1
2. 提取问题和文档的实体 → 实体嵌入矩阵
3. 计算实体匹配分数 → v2
4. v3 = α * v1 + (1-α) * v2
5. 按 v3 降序重排
"""

import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
import spacy

from .llm_provider import BaseLLMProvider


class EntityReranker:
    """
    实体感知重排序器。
    
    融合向量相似度和实体语义匹配进行文档重排序。
    """
    
    def __init__(
        self,
        embedding_provider: BaseLLMProvider,
        alpha: float = 0.5,
        ner_model: str = "en_core_web_sm",
    ):
        """
        Args:
            embedding_provider: 嵌入提供者
            alpha: v1 权重 (0~1), 默认 0.5
            ner_model: spaCy NER 模型名称
        """
        self.embedding_provider = embedding_provider
        self.alpha = alpha
        try:
            self.ner_model = spacy.load(ner_model)
        except OSError:
            # 如果模型未安装，尝试自动下载
            print(f"spaCy 模型 '{ner_model}' 未找到，正在下载...")
            spacy.cli.download(ner_model)
            self.ner_model = spacy.load(ner_model)
    
    def extract_entities(self, text: str) -> List[str]:
        """
        使用 spaCy NER 提取实体文本列表。
        
        Args:
            text: 输入文本
            
        Returns:
            实体文本列表
        """
        if not text or text.lower() in ["nan", "none", ""]:
            return []
        
        doc = self.ner_model(text)
        return [ent.text for ent in doc.ents]
    
    def _compute_entity_score(
        self,
        question_entities: List[str],
        doc_entities: List[str]
    ) -> float:
        """
        计算实体匹配分数。
        
        Args:
            question_entities: 问题实体列表
            doc_entities: 文档实体列表
        
        Returns:
            实体匹配分数 (0~1)
        """
        if not question_entities or not doc_entities:
            return 0.0
        
        # 实体嵌入 → 矩阵
        M_q = np.array([self.embedding_provider.embed(e) for e in question_entities])
        M_d = np.array([self.embedding_provider.embed(e) for e in doc_entities])
        
        # M_q × M_d.T → m×k 相似度矩阵
        sim_matrix = M_q @ M_d.T
        
        # 矩阵所有元素的平均值作为最终分数
        return float(np.mean(sim_matrix))
    
    def rerank(
        self,
        question: str,
        docs: List[Dict[str, Any]],
        top_k: int = 5,
        content_field: str = "content",
        embedding_field: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        对文档列表进行重排序。
        
        Args:
            question: 问题文本
            docs: 文档列表，每个文档为字典
            top_k: 返回的文档数量
            content_field: 文档内容字段名
            embedding_field: 文档嵌入字段名 (如 None 则重新计算)
        
        Returns:
            重排序后的文档列表
        """
        if not docs:
            return []
        
        # 步骤 1: 计算向量相似度 v1
        q_emb = np.array(self.embedding_provider.embed(question))
        v1 = []
        for doc in docs:
            if embedding_field and embedding_field in doc:
                d_emb = np.array(doc[embedding_field])
            else:
                d_emb = np.array(self.embedding_provider.embed(doc[content_field]))
            score = cosine_similarity([q_emb], [d_emb])[0][0]
            v1.append(score)
        
        # 步骤 2: 实体识别
        q_entities = self.extract_entities(question)
        doc_entities = [self.extract_entities(doc[content_field]) for doc in docs]
        
        # 步骤 3 & 4: 计算实体匹配分数 v2
        v2 = []
        for d_ents in doc_entities:
            s2 = self._compute_entity_score(q_entities, d_ents)
            v2.append(s2)
        
        # 步骤 5: 融合排序 v3 = α * v1 + (1-α) * v2
        v3 = [self.alpha * s1 + (1 - self.alpha) * s2 
              for s1, s2 in zip(v1, v2)]
        
        # 按 v3 降序重排
        sorted_indices = np.argsort(v3)[::-1][:top_k]
        return [docs[i] for i in sorted_indices]
    
    def rerank_with_scores(
        self,
        question: str,
        docs: List[Dict[str, Any]],
        top_k: int = 5,
        content_field: str = "content",
        embedding_field: Optional[str] = None,
    ) -> tuple[List[Dict[str, Any]], Dict[str, List[float]]]:
        """
        对文档列表进行重排序，并返回各阶段分数。
        
        Args:
            question: 问题文本
            docs: 文档列表
            top_k: 返回的文档数量
            content_field: 文档内容字段名
            embedding_field: 文档嵌入字段名
        
        Returns:
            (重排序后的文档列表, 分数字典 {v1: [...], v2: [...], v3: [...]})
        """
        if not docs:
            return [], {"v1": [], "v2": [], "v3": []}
        
        # 步骤 1: 计算向量相似度 v1
        q_emb = np.array(self.embedding_provider.embed(question))
        v1 = []
        for doc in docs:
            if embedding_field and embedding_field in doc:
                d_emb = np.array(doc[embedding_field])
            else:
                d_emb = np.array(self.embedding_provider.embed(doc[content_field]))
            score = cosine_similarity([q_emb], [d_emb])[0][0]
            v1.append(score)
        
        # 步骤 2: 实体识别
        q_entities = self.extract_entities(question)
        doc_entities = [self.extract_entities(doc[content_field]) for doc in docs]
        
        # 步骤 3 & 4: 计算实体匹配分数 v2
        v2 = []
        for d_ents in doc_entities:
            s2 = self._compute_entity_score(q_entities, d_ents)
            v2.append(s2)
        
        # 步骤 5: 融合排序 v3 = α * v1 + (1-α) * v2
        v3 = [self.alpha * s1 + (1 - self.alpha) * s2 
              for s1, s2 in zip(v1, v2)]
        
        # 按 v3 降序重排
        sorted_indices = np.argsort(v3)[::-1][:top_k]
        reranked_docs = [docs[i] for i in sorted_indices]
        
        scores = {
            "v1": [v1[i] for i in sorted_indices],
            "v2": [v2[i] for i in sorted_indices],
            "v3": [v3[i] for i in sorted_indices],
        }
        
        return reranked_docs, scores
