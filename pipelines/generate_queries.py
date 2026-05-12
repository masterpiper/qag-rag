"""
QAG-RAG 主流程：查询生成与 Milvus 插入。

处理 BioASQ 语料库，生成子问题并存储到 Milvus 向量数据库中。
支持 Ollama 和 OpenAI 两种 LLM 后端。
"""

import os
import sys
import ast
import uuid
from tqdm import tqdm
from datasets import load_from_disk
from pymilvus import DataType, MilvusClient
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 导入本地模块
import sys
sys.path.insert(0, '..')

from config import (
    milvus_config,
    retrieval_config,
    llm_config,
    embedding_config,
)
from core.llm_provider import get_llm_provider, get_embedding_provider
from core.utils import query_generation, get_embedding
from core.prompt import PROMPT

# 常量
COLLECTION_NAME = "bioasq_query"
RECOVER = 17357 + 22 - 1 + 10132 + 3735 + 11 + 3164 + 2405 + 3 + 1269 + 19 + 16 + 2089
MODE = "fix"  # "normal" 或 "fix"


def create_milvus_collection(client: MilvusClient, collection_name: str, dimension: int):
    """
    创建 Milvus 集合。
    
    Args:
        client: Milvus 客户端
        collection_name: 集合名称
        dimension: 向量维度
    """
    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, max_length=65535)
    schema.add_field(field_name="question", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=dimension)
    schema.add_field(field_name="doc_ids", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="related_questions", datatype=DataType.VARCHAR, max_length=65535)
    
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="id",
        index_name="id_idx",
        index_type="AUTOINDEX",
    )
    index_params.add_index(
        field_name="embedding",
        index_name="cosine_idx",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )
    
    if client.has_collection(collection_name):
        print(f"集合 '{collection_name}' 已存在。")
    else:
        client.create_collection(
            collection_name=collection_name,
            dimension=dimension,
            schema=schema,
            index_params=index_params,
        )
        print(f"集合 '{collection_name}' 创建成功。")


def process_passage(
    passage: dict,
    client: MilvusClient,
    llm_provider,
    embedding_provider,
    collection_name: str,
) -> bool:
    """
    处理单个语料段落。
    
    Args:
        passage: 语料段落字典
        client: Milvus 客户端
        llm_provider: LLM 提供者
        embedding_provider: 嵌入提供者
        collection_name: 集合名称
        
    Returns:
        处理是否成功
    """
    text = passage['passage']
    
    # 跳过无效文本
    if text.lower() in ["nan", "", "none"] or not text or not text.split():
        return True
    
    # 生成子问题
    q_list = []
    retry = 0
    while not q_list and retry < 3:
        retry += 1
        try:
            q_list = query_generation(text=text, provider=llm_provider)
        except Exception as e:
            print(f"生成问题失败 (ID: {passage['id']}): {e}")
            continue
    
    if not q_list:
        with open("./error_log.txt", "a") as f:
            f.write(f"Failed ID: {passage['id']}\n")
        return False
    
    # 处理每个生成的问题
    for q in q_list:
        if not q:
            continue
            
        # 生成嵌入向量
        q_embedding = embedding_provider.embed(q)
        
        # 搜索相似问题
        res = client.search(
            collection_name=collection_name,
            limit=5,
            data=[q_embedding],
            output_fields=["id", "question", "doc_ids", "related_questions", "embedding"],
        )
        
        if_insert_q = False
        
        if res and res[0]:
            for r in res[0]:
                distance = r['distance']
                
                # 合并相似问题
                if distance > retrieval_config.merge_threshold:
                    doc_ids = ast.literal_eval(r['entity']['doc_ids'])
                    
                    if passage['id'] in doc_ids or len(doc_ids) > retrieval_config.limite_related_docs:
                        continue
                    
                    doc_ids.append(passage['id'])
                    client.upsert(
                        collection_name=collection_name,
                        data={"id": r['entity']['id'], "doc_ids": str(doc_ids)},
                        partial_update=True
                    )
                    if_insert_q = True
                    
                # 添加相关问题
                elif distance <= retrieval_config.merge_threshold and distance > retrieval_config.related_threshold:
                    new_id = uuid.uuid4().int & ((1 << 63) - 1)
                    client.insert(
                        collection_name=collection_name,
                        data={
                            "id": new_id,
                            "question": q,
                            "embedding": q_embedding,
                            "doc_ids": f"[{passage['id']}]",
                            "related_questions": f"[{r['id']}]",
                        }
                    )
                    
                    related_questions = ast.literal_eval(r['entity']['related_questions'])
                    
                    if new_id not in related_questions and len(related_questions) <= retrieval_config.limite_num_q:
                        related_questions.append(new_id)
                        client.upsert(
                            collection_name=collection_name,
                            data={
                                "id": r['entity']['id'],
                                "related_questions": str(related_questions),
                            },
                            partial_update=True
                        )
                    if_insert_q = True
            
            # 如果没有合并或关联，直接插入
            if not if_insert_q:
                client.insert(
                    collection_name=collection_name,
                    data=[{
                        "id": uuid.uuid4().int % ((1 << 63) - 1),
                        "question": q,
                        "embedding": q_embedding,
                        "doc_ids": f"[{passage['id']}]",
                        "related_questions": "[]",
                    }]
                )
        else:
            # 没有找到相似问题，直接插入
            client.insert(
                collection_name=collection_name,
                data=[{
                    "id": uuid.uuid4().int % ((1 << 63) - 1),
                    "question": q,
                    "embedding": q_embedding,
                    "doc_ids": f"[{passage['id']}]",
                    "related_questions": "[]",
                }]
            )
    
    return True


def run_normal_mode(ds_corpus, client: MilvusClient, llm_provider, embedding_provider):
    """运行正常模式：处理所有语料。"""
    for i in tqdm(range(RECOVER, len(ds_corpus['passages']))):
        passage = ds_corpus['passages'][i]
        process_passage(
            passage=passage,
            client=client,
            llm_provider=llm_provider,
            embedding_provider=embedding_provider,
            collection_name=COLLECTION_NAME,
        )


def run_fix_mode(ds_corpus, client: MilvusClient, llm_provider, embedding_provider):
    """运行修复模式：处理已知失败的 ID。"""
    error_ids = [
        12446, 16052, 20982, 27792, 29290, 29813, 30252, 32012, 
        34191, 34622, 35164, 36595, 37871, 37878, 38346, 39066, 
        39198, 39315, 39801, 39840
    ]
    
    # 打印第一个错误 ID 的文本
    if error_ids:
        first_id = error_ids[0]
        print(f"{first_id}: {ds_corpus['passages'][first_id]['passage']}")
    
    for i in tqdm(error_ids):
        passage = ds_corpus['passages'][i]
        process_passage(
            passage=passage,
            client=client,
            llm_provider=llm_provider,
            embedding_provider=embedding_provider,
            collection_name=COLLECTION_NAME,
        )


def main():
    """主函数。"""
    print("=" * 60)
    print("QAG-RAG: 查询生成与 Milvus 插入")
    print("=" * 60)
    
    # 打印配置信息
    print(f"\nLLM 提供者：{llm_config.provider} ({llm_config.model})")
    print(f"Embedding 提供者：{embedding_config.provider} ({embedding_config.model})")
    print(f"Milvus: {milvus_config.uri}/{milvus_config.db_name}")
    print(f"运行模式：{MODE}")
    print("=" * 60)
    
    # 加载数据集
    print("\n加载数据集...")
    ds_corpus = load_from_disk("../data/rag_mini_bioasq_corpus")
    print(f"数据集加载完成，共 {len(ds_corpus['passages'])} 条记录")
    print(f"示例：{ds_corpus['passages'][0]}")
    
    # 初始化 Milvus 客户端
    print("\n初始化 Milvus 客户端...")
    client = MilvusClient(
        uri=milvus_config.uri,
        token=milvus_config.token,
        db_name=milvus_config.db_name
    )
    
    # 创建集合
    print("\n创建/检查 Milvus 集合...")
    create_milvus_collection(client, COLLECTION_NAME, embedding_config.dimension)
    
    # 初始化 LLM 提供者
    print("\n初始化 LLM 提供者...")
    llm_provider = get_llm_provider(
        provider_type=llm_config.provider,
        model=llm_config.model,
        api_key=llm_config.api_key,
        base_url=llm_config.base_url,
        host=llm_config.ollama_host,
    )
    
    embedding_provider = get_embedding_provider(
        provider_type=embedding_config.provider,
        model=embedding_config.model,
        api_key=embedding_config.api_key,
        base_url=embedding_config.base_url,
        host=embedding_config.ollama_host,
    )
    
    # 运行主流程
    print("\n开始处理语料...")
    if MODE != "fix":
        run_normal_mode(ds_corpus, client, llm_provider, embedding_provider)
    else:
        run_fix_mode(ds_corpus, client, llm_provider, embedding_provider)
    
    print("\n处理完成！")


if __name__ == "__main__":
    main()
