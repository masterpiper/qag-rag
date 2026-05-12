"""
工具函数模块。

提供通用的工具函数，包括：
- 文本嵌入生成
- 查询生成
- 实体提取
- 关系三元组提取
- 查询摘要
"""

import re
from typing import Optional
from .prompt import PROMPT
from .llm_provider import BaseLLMProvider, OllamaProvider, OpenAIProvider


def get_embedding(
    text: str, 
    provider: BaseLLMProvider,
    model: Optional[str] = None
) -> list[float]:
    """
    生成文本嵌入向量。
    
    Args:
        text: 输入文本
        provider: LLM 提供者实例
        model: 模型名称（可选）
        
    Returns:
        嵌入向量列表
    """
    return provider.embed(text, model)


def query_generation(
    text: str, 
    provider: BaseLLMProvider,
    model: Optional[str] = None
) -> list[str]:
    """
    从文本生成子问题列表。
    
    Args:
        text: 输入文本
        provider: LLM 提供者实例
        model: 模型名称（可选）
        
    Returns:
        生成的问题列表
    """
    if text.lower() == "nan" or text == "" or text is None or text.lower() == "none" or text.split() == "":
        return []

    # 构建提示词
    query_gen_prompt = PROMPT["query_generation"].format(
        delimiter_start=PROMPT["Q_DELIMITER_START"],
        text=text,
        delimiter_end=PROMPT["Q_DELIMITER_END"]
    )
    
    # 生成响应
    if isinstance(provider, OllamaProvider):
        response_text = provider.generate(
            prompt=query_gen_prompt,
            model=model,
            options={"num_predict": 2048}
        )
    else:
        # 对于 OpenAI 提供者，跳过 extra_body 参数（某些模型不支持 enable_thinking）
        response_text = provider.generate(
            prompt=query_gen_prompt,
            model=model,
            max_tokens=2048,
            skip_extra_body=True
        )
    
    # 使用正则表达式提取问题
    re_str = f"{PROMPT['Q_DELIMITER_START']}(.*?){PROMPT['Q_DELIMITER_END']}"
    matches = re.findall(re_str, response_text, re.DOTALL)
    
    return list(set(matches))


def entity_extractor(
    text: str, 
    provider: BaseLLMProvider,
    model: Optional[str] = None
) -> list[tuple[str, str, str]]:
    """
    从文本中提取实体。
    
    Args:
        text: 输入文本
        provider: LLM 提供者实例
        model: 模型名称（可选）
        
    Returns:
        实体列表，每个实体为 (name, type, description) 元组
    """
    if text.lower() == "nan" or text == "" or text is None or text.lower() == "none" or text.split() == "":
        return []

    # 构建提示词
    extract_prompt = PROMPT["entity_extraction"].format(
        e_delimiter_start=PROMPT["E_DELIMITER_START"],
        e_delimiter_end=PROMPT["E_DELIMITER_END"],
        n_delimiter_start=PROMPT["N_DELIMITER_START"],
        n_delimiter_end=PROMPT["N_DELIMITER_END"],
        t_delimiter_start=PROMPT["T_DELIMITER_START"],
        t_delimiter_end=PROMPT["T_DELIMITER_END"],
        d_delimiter_start=PROMPT["D_DELIMITER_START"],
        d_delimiter_end=PROMPT["D_DELIMITER_END"],
        text=text
    )

    # 生成响应
    if isinstance(provider, OllamaProvider):
        response_text = provider.generate(
            prompt=extract_prompt,
            model=model,
            think=False,
            options={"thinking": False}
        )
    else:
        # 对于 OpenAI 提供者，跳过 extra_body 参数（某些模型不支持 enable_thinking）
        response_text = provider.generate(
            prompt=extract_prompt,
            model=model,
            temperature=0,
            skip_extra_body=True
        )

    # 使用正则表达式提取实体
    entity_re = f"{PROMPT['E_DELIMITER_START']}(.*?){PROMPT['E_DELIMITER_END']}"
    matches = re.findall(entity_re, response_text, re.DOTALL)
    
    name_re = f"{PROMPT['N_DELIMITER_START']}(.*?){PROMPT['N_DELIMITER_END']}"
    type_re = f"{PROMPT['T_DELIMITER_START']}(.*?){PROMPT['T_DELIMITER_END']}"
    description_re = f"{PROMPT['D_DELIMITER_START']}(.*?){PROMPT['D_DELIMITER_END']}"
    
    results = []
    for match in matches:
        names = re.findall(name_re, match, re.DOTALL)
        types = re.findall(type_re, match, re.DOTALL)
        descriptions = re.findall(description_re, match, re.DOTALL)
        if names and types and descriptions:
            results.append((names[0], types[0], descriptions[0]))
    
    return results


def tuple_extractor(
    text: str, 
    provider: BaseLLMProvider,
    model: Optional[str] = None
) -> list[tuple[str, str, str, str]]:
    """
    从文本中提取关系三元组。
    
    Args:
        text: 输入文本
        provider: LLM 提供者实例
        model: 模型名称（可选）
        
    Returns:
        三元组列表，每个三元组为 (head, relation, tail, description) 元组
    """
    if text.lower() == "nan" or text == "" or text is None or text.lower() == "none" or text.split() == "":
        return []

    # 构建提示词
    extract_prompt = PROMPT["tuple_extraction"].format(
        tuple_delimiter_start=PROMPT["TUPLE_DELIMITER_START"],
        tuple_delimiter_end=PROMPT["TUPLE_DELIMITER_END"],
        e_delimiter_start=PROMPT["E_DELIMITER_START"],
        e_delimiter_end=PROMPT["E_DELIMITER_END"],
        r_delimiter_start=PROMPT["R_DELIMITER_START"],
        r_delimiter_end=PROMPT["R_DELIMITER_END"],
        d_delimiter_start=PROMPT["D_DELIMITER_START"],
        d_delimiter_end=PROMPT["D_DELIMITER_END"],
        text=text
    )
    
    # 生成响应
    if isinstance(provider, OllamaProvider):
        response_text = provider.generate(
            prompt=extract_prompt,
            model=model,
            think=False,
            options={"thinking": False}
        )
    else:
        # 对于 OpenAI 提供者，跳过 extra_body 参数（某些模型不支持 enable_thinking）
        response_text = provider.generate(
            prompt=extract_prompt,
            model=model,
            temperature=0,
            skip_extra_body=True
        )

    # 使用正则表达式提取三元组
    tuple_re = f"{PROMPT['TUPLE_DELIMITER_START']}(.*?){PROMPT['TUPLE_DELIMITER_END']}"
    matches = re.findall(tuple_re, response_text, re.DOTALL)
    
    entity_re = f"{PROMPT['E_DELIMITER_START']}(.*?){PROMPT['E_DELIMITER_END']}"
    relation_re = f"{PROMPT['R_DELIMITER_START']}(.*?){PROMPT['R_DELIMITER_END']}"
    description_re = f"{PROMPT['D_DELIMITER_START']}(.*?){PROMPT['D_DELIMITER_END']}"
    
    results = []
    for match in matches:
        entities = re.findall(entity_re, match, re.DOTALL)
        if len(entities) == 2:
            relations = re.findall(relation_re, match, re.DOTALL)
            descriptions = re.findall(description_re, match, re.DOTALL)
            if relations and descriptions:
                results.append((entities[0], relations[0], entities[1], descriptions[0]))
    
    return results


def query_summary(
    query_list: list[str], 
    provider: BaseLLMProvider,
    model: Optional[str] = None
) -> str:
    """
    总结问题列表为单个问题。
    
    Args:
        query_list: 问题列表
        provider: LLM 提供者实例
        model: 模型名称（可选）
        
    Returns:
        总结后的问题
    """
    # 拼接问题
    questions = ",".join(query_list)
    
    prompt = PROMPT['query_summary'].format(
        delimiter_start=PROMPT['Q_DELIMITER_START'],
        delimiter_end=PROMPT['Q_DELIMITER_END'],
        questions=questions
    )
    
    # 生成响应
    if isinstance(provider, OllamaProvider):
        response_text = provider.generate(
            prompt=prompt,
            model=model,
            think=False,
            options={"thinking": False}
        )
    else:
        # 对于 OpenAI 提供者，跳过 extra_body 参数（某些模型不支持 enable_thinking）
        response_text = provider.generate(
            prompt=prompt,
            model=model,
            temperature=0,
            skip_extra_body=True
        )

    # 提取总结的问题
    q_re = f"{PROMPT['Q_DELIMITER_START']}(.*?){PROMPT['Q_DELIMITER_END']}"
    matches = re.findall(q_re, response_text, re.DOTALL)
    
    return matches[0] if matches else ""
