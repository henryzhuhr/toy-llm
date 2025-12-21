"""
https://help.aliyun.com/zh/model-studio/use-bailian-in-langchain


uv add langchain-community dashscope
"""

import os
from typing import Optional

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


def get_default_apikey() -> str:
    return os.getenv("DASHSCOPE_API_KEY", "")


def get_aliyun_chatmodel(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
    apikey: Optional[str] = None,
) -> BaseChatModel:
    if apikey is None:
        apikey = get_default_apikey()
    chatLLM = ChatOpenAI(
        api_key=SecretStr(apikey),
        base_url=base_url,
        model=model,  # 此处以qwen-plus为例，您可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        # other params...
    )
    return chatLLM


def get_aliyun_embeddings(
    model="text-embedding-v4",
    apikey: Optional[str] = None,
) -> Embeddings:
    if apikey is None:
        apikey = get_default_apikey()
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v4",
        dashscope_api_key=apikey,
        # other params...
    )
    return embeddings
