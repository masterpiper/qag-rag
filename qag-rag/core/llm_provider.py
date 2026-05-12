"""
LLM 提供者抽象层。

提供统一的接口支持 Ollama 和 OpenAI 两种 LLM 后端，包括：
- 文本生成 (generate)
- 向量嵌入 (embed)
"""

from abc import ABC, abstractmethod
from typing import Optional
import ollama
from openai import OpenAI


class BaseLLMProvider(ABC):
    """LLM 提供者抽象基类。"""
    
    @abstractmethod
    def generate(self, prompt: str, model: Optional[str] = None, **kwargs) -> str:
        """
        生成文本响应。
        
        Args:
            prompt: 输入提示词
            model: 模型名称（可选，使用默认模型如果未指定）
            **kwargs: 其他参数
            
        Returns:
            生成的文本响应
        """
        pass
    
    @abstractmethod
    def embed(self, text: str, model: Optional[str] = None) -> list[float]:
        """
        生成文本嵌入向量。
        
        Args:
            text: 输入文本
            model: 模型名称（可选，使用默认模型如果未指定）
            
        Returns:
            嵌入向量列表
        """
        pass


class OllamaProvider(BaseLLMProvider):
    """Ollama LLM 提供者。"""
    
    def __init__(self, host: str = "http://localhost:11434", model: str = "qwen3:1.7b"):
        """
        初始化 Ollama 提供者。
        
        Args:
            host: Ollama 服务器地址
            model: 默认模型名称
        """
        self.host = host
        self.model = model
        # 设置 Ollama 客户端的主机
        ollama.host = host
    
    def generate(self, prompt: str, model: Optional[str] = None, **kwargs) -> str:
        """
        使用 Ollama 生成文本。
        
        Args:
            prompt: 输入提示词
            model: 模型名称（可选）
            think: 是否启用思考模式
            options: 其他选项
            
        Returns:
            生成的文本
        """
        model = model or self.model
        think = kwargs.get("think", False)
        options = kwargs.get("options", {})
        
        response = ollama.generate(
            model=model,
            prompt=prompt,
            think=think,
            options=options,
        )
        return response["response"]
    
    def embed(self, text: str, model: Optional[str] = None) -> list[float]:
        """
        使用 Ollama 生成嵌入向量。
        
        Args:
            text: 输入文本
            model: 模型名称（可选）
            
        Returns:
            嵌入向量
        """
        model = model or self.model
        
        response = ollama.embed(
            model=model,
            input=text,
        )
        return response["embeddings"][0]


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM 提供者。"""
    
    def __init__(
        self, 
        api_key: str, 
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
    ):
        """
        初始化 OpenAI 提供者。
        
        Args:
            api_key: OpenAI API Key
            model: 默认 LLM 模型名称
            base_url: API Base URL（可选，用于兼容其他 OpenAI 兼容的 API）
            embedding_model: 默认嵌入模型名称
        """
        self.model = model
        self.embedding_model = embedding_model
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
    
    def generate(self, prompt: str, model: Optional[str] = None, **kwargs) -> str:
        """
        使用 OpenAI 生成文本。

        Args:
            prompt: 输入提示词
            model: 模型名称（可选）
            temperature: 温度参数
            max_tokens: 最大生成 token 数

        Returns:
            生成的文本
        """
        model = model or self.model
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 2048)

        # 构建请求参数
        request_params = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # 添加 extra_body 参数（仅当模型支持 thinking 时使用）
        # 注意：某些模型（如 GLM-Z1）不支持 enable_thinking 参数
        # 如果遇到错误，可以设置 skip_extra_body=True 来跳过
        if not kwargs.get("skip_extra_body", False):
            request_params["extra_body"] = {
                "enable_thinking": False
            }

        response = self.client.chat.completions.create(**request_params)
        return response.choices[0].message.content
    
    def embed(self, text: str, model: Optional[str] = None) -> list[float]:
        """
        使用 OpenAI 生成嵌入向量。
        
        Args:
            text: 输入文本
            model: 模型名称（可选）
            
        Returns:
            嵌入向量
        """
        model = model or self.embedding_model
        
        response = self.client.embeddings.create(
            model=model,
            input=text,
        )
        return response.data[0].embedding


def get_llm_provider(provider_type: str, **kwargs) -> BaseLLMProvider:
    """
    工厂函数：获取 LLM 提供者实例。
    
    Args:
        provider_type: 提供者类型 ("ollama" 或 "openai")
        **kwargs: 提供者特定的参数
        
    Returns:
        LLM 提供者实例
        
    Raises:
        ValueError: 不支持的提供者类型
    """
    if provider_type.lower() == "ollama":
        return OllamaProvider(
            host=kwargs.get("host", "http://localhost:11434"),
            model=kwargs.get("model", "qwen3:1.7b"),
        )
    elif provider_type.lower() == "openai":
        return OpenAIProvider(
            api_key=kwargs.get("api_key"),
            model=kwargs.get("model", "gpt-4o-mini"),
            base_url=kwargs.get("base_url"),
            embedding_model=kwargs.get("embedding_model", "text-embedding-3-small"),
        )
    else:
        raise ValueError(f"不支持的 LLM 提供者类型：{provider_type}")


def get_embedding_provider(provider_type: str, **kwargs) -> BaseLLMProvider:
    """
    工厂函数：获取嵌入提供者实例。

    Args:
        provider_type: 提供者类型 ("ollama" 或 "openai")
        **kwargs: 提供者特定的参数，其中 model 会被用作 embedding_model

    Returns:
        嵌入提供者实例

    Raises:
        ValueError: 不支持的提供者类型
    """
    # 嵌入提供者与 LLM 提供者使用相同的类
    # 注意：将 kwargs 中的 model 参数复制为 embedding_model
    if "model" in kwargs:
        kwargs["embedding_model"] = kwargs["model"]
    return get_llm_provider(provider_type, **kwargs)
