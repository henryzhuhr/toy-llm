"""
Build a semantic search engine with LangChain
https://docs.langchain.com/oss/python/langchain/knowledge-base
"""

import os
from typing import Any, Protocol, overload

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger


class Logger(Protocol):
    @overload
    def info(__self, __message: str, *args: Any, **kwargs: Any) -> None: ...  # noqa: N805
    @overload
    def info(__self, __message: Any) -> None: ...  # noqa: N805
    @overload
    def error(__self, __message: str, *args: Any, **kwargs: Any) -> None: ...  # noqa: N805
    @overload
    def error(__self, __message: Any) -> None: ...  # noqa: N805


def main(log: Logger):
    file_path = os.path.expandvars("$HOME/Downloads/nke-10k-2023.pdf")
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    log.info(f"Number of documents: {len(docs)}", len(docs))
    log.info(f"First 100 chars: {docs[0].page_content[:100]}")
    log.info(f"Metadata: {docs[0].metadata}")

    # 对文本进行拆分
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    all_splits = text_splitter.split_documents(docs)

    # 加载嵌入模型
    # https://ollama.com/library/qwen3-embedding
    # ollama pull qwen3-embedding:0.6b
    embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")
    # 创建向量存储
    vector_store = InMemoryVectorStore(embeddings)

    # 将拆分后的文本转换为向量
    vector_1 = embeddings.embed_query(all_splits[0].page_content)
    vector_2 = embeddings.embed_query(all_splits[1].page_content)

    assert len(vector_1) == len(vector_2)
    log.info(f"Generated vectors of length {len(vector_1)}\n")
    log.info(f"{vector_1[:10]}")

    ids = vector_store.add_documents(documents=all_splits)

    # 进行相似度搜索
    results = vector_store.similarity_search(
        "How many distribution centers does Nike have in the US?"
    )

    log.info(f"{repr(results[0])}")


if __name__ == "__main__":
    main(logger)
