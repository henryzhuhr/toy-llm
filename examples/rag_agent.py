"""
Build a RAG agent with LangChain
https://docs.langchain.com/oss/python/langchain/rag

Split markdown:
https://docs.langchain.com/oss/python/integrations/splitters/markdown_header_metadata_splitter
"""

import os
from typing import List

from bs4.filter import SoupStrainer
from langchain.agents import create_agent
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph.state import CompiledStateGraph
from loguru import logger
from pydantic import BaseModel, Field

from src.log.interface import Logger


class RetrieveContextTool(BaseTool):
    name: str = "retrieve_context"
    description: str = "Retrieve relevant information from the indexed document to help answer a query."

    vector_store: InMemoryVectorStore = Field(..., exclude=True)  # 不序列化到 LLM
    k: int = 2

    def _run(self, query: str) -> str:
        retrieved_docs = self.vector_store.similarity_search(query, k=self.k)
        serialized = "\n\n".join(
            f"Source: {doc.metadata}\nContent: {doc.page_content}"
            for doc in retrieved_docs
        )
        return serialized

    # 如果你需要返回 artifact（如原始 Document 对象），可重写 invoke 或使用 response_format
    # 但 BaseTool 默认只返回 str（或 JSON serializable）。若需 artifact，建议在 agent 层处理


class RAGAgentConfig(BaseModel):
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    """Ollama API base URL."""

    embeddings_model: str = "qwen3-embedding:0.6b"
    """Ollama embeddings model name."""

    chat_model: str = "qwen3:0.6b"
    """Ollama chat model name."""


class RAGAgent:
    config: RAGAgentConfig

    agent: CompiledStateGraph

    embeddings: OllamaEmbeddings
    vector_store: InMemoryVectorStore

    def __init__(self, config: RAGAgentConfig):
        self.config = config
        # 初始化嵌入模型
        self.embeddings = OllamaEmbeddings(model=self.config.embeddings_model)
        # 初始化向量存储
        self.vector_store = InMemoryVectorStore(self.embeddings)

        retrieve_tool = RetrieveContextTool(vector_store=self.vector_store)

        tools = [retrieve_tool]
        prompt = SystemMessage(
            content=(
                # "You are an AI assistant that helps people find information "
                # "about building agents with LangChain."
                "您是一个AI助手，帮助人们寻找有关使用LangChain构建代理的信息。尽可能使用中文回答。"
            ),
        )
        chat_model = ChatOllama(base_url=config.base_url, model=config.chat_model)
        self.agent = create_agent(chat_model, tools, system_prompt=prompt)

    def add_documents(self, documents: List[Document]):
        document_ids = self.vector_store.add_documents(documents)
        return document_ids

    def run(self, query: str):
        for event in self.agent.stream(
            {"messages": [HumanMessage(content=query)]},
            stream_mode="values",
        ):
            message: AnyMessage = event["messages"][-1]
            if isinstance(message, AIMessage):
                logger.info(f"🤖 Assistant: {message}")
            elif isinstance(message, ToolMessage):
                logger.info(f"🔨 Tool Result: {message}")
            else:
                logger.info(f"💌 Message({type(message)}):  {message}")


class AgentDocumentLoader:
    @staticmethod
    def load(log: Logger) -> List[Document]:
        # Only keep post title, headers, and content from the full HTML.
        bs4_strainer = SoupStrainer(
            class_=("post-title", "post-header", "post-content")
        )
        loader = WebBaseLoader(
            web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
            bs_kwargs={"parse_only": bs4_strainer},
        )
        docs = loader.load()
        log.info(
            f"Loaded {len(docs)} documents.Total characters: {len(docs[0].page_content)}"
        )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # chunk size (characters)
            chunk_overlap=200,  # chunk overlap (characters)
            add_start_index=True,  # track index in original document
        )
        all_splits = text_splitter.split_documents(docs)
        log.info(f"Split blog post into {len(all_splits)} sub-documents.")
        return all_splits


def main(log: Logger):
    config = RAGAgentConfig()
    log.info(f"{config.model_dump_json()}")
    agent = RAGAgent(config)

    docs = AgentDocumentLoader.load(log)

    document_ids = agent.add_documents(docs)
    log.info(f"Added {len(document_ids)} documents to vector store: {document_ids[:5]}")

    query = "What are the key aspects of building an agent with LangChain?"
    query = "使用LangChain构建代理的关键方面是什么？，使用中文回答"
    log.info(f"Running RAG agent with query: {query}")
    agent.run(query)


if __name__ == "__main__":
    main(logger)
