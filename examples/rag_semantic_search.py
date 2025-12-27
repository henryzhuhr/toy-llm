"""
Build a semantic search engine with LangChain
https://docs.langchain.com/oss/python/langchain/knowledge-base
"""

import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.runnables import chain
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from src.log.interface import Logger


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
    # ollama pull qwen3-embedding:4b
    embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")
    # 创建向量存储
    vector_store = InMemoryVectorStore(embeddings)

    # 将拆分后的文本转换为向量
    vector_1 = embeddings.embed_query(all_splits[0].page_content)
    vector_2 = embeddings.embed_query(all_splits[1].page_content)

    assert len(vector_1) == len(vector_2)
    log.info(f"Generated vectors of length {len(vector_1)}, {vector_1[:10]}")

    ids = vector_store.add_documents(documents=all_splits)

    # 进行相似度搜索
    results = vector_store.similarity_search(
        "How many distribution centers does Nike have in the US?"
    )

    log.info(f"similarity search: {repr(results[0])}")

    # 进行相似度搜索并返回分数
    results = vector_store.similarity_search_with_score(
        "What was Nike's revenue in 2023?"
    )
    similarity_doc, score = results[0]
    log.info(f"Score: {score} similarity search: {repr(similarity_doc)}")

    # 通过向量进行相似度搜索
    embedding = embeddings.embed_query("How were Nike's margins impacted in 2023?")
    results = vector_store.similarity_search_by_vector(embedding)

    """
    4. Retrievers
    """

    """方式1: 使用 chain 装饰器创建检索器"""

    @chain
    def retriever(query: str) -> List[Document]:
        return vector_store.similarity_search(query, k=1)

    retriever.batch(
        [
            "How many distribution centers does Nike have in the US?",
            "When was Nike incorporated?",
        ],
    )

    """方式2: 通过 VectorStore 创建检索器"""
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1},
    )

    retriever.batch(
        [
            "How many distribution centers does Nike have in the US?",
            "When was Nike incorporated?",
        ],
    )


if __name__ == "__main__":
    main(logger)
