"""
Build a RAG agent with LangChain
https://docs.langchain.com/oss/python/langchain/rag

Split markdown:
https://docs.langchain.com/oss/python/integrations/splitters/markdown_header_metadata_splitter
"""

import os
from typing import List

from bs4.filter import SoupStrainer
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from pydantic import BaseModel

from src.log.interface import Logger


class RAGAgentConfig(BaseModel):
    embeddings_model: str = "qwen3-embedding:0.6b"


class RAGAgent:
    config: RAGAgentConfig

    agent_executor: AgentExecutor

    embeddings: OllamaEmbeddings
    vector_store: InMemoryVectorStore

    def __init__(self, config: RAGAgentConfig):
        self.config = config
        # 初始化嵌入模型
        self.embeddings = OllamaEmbeddings(model=self.config.embeddings_model)
        # 初始化向量存储
        self.vector_store = InMemoryVectorStore(self.embeddings)

        @tool(response_format="content_and_artifact")
        def retrieve_context(query: str):
            """Retrieve information to help answer a query."""
            retrieved_docs = self.vector_store.similarity_search(query, k=2)
            serialized = "\n\n".join(
                (f"Source: {doc.metadata}\nContent: {doc.page_content}")
                for doc in retrieved_docs
            )
            return serialized, retrieved_docs

        tools = [retrieve_context]
        # If desired, specify custom instructions
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=(
                        "You are an AI assistant that helps people find information "
                        "about building agents with LangChain."
                    ),
                ),
                HumanMessage(
                    content=(
                        "Use the following context to answer the question:\n"
                        "{retrieve_context}\n\n"
                        "Question: {input}"
                    ),
                ),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
                # ("system", "You are a helpful assistant"),
                # ("placeholder", "{chat_history}"),
                # ("human", "{input}"),
            ]
        )
        base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model_name = os.getenv("OLLAMA_MODEL_NAME", "qwen3:4b")
        chat_model = ChatOllama(base_url=base_url, model=model_name)
        agent = create_tool_calling_agent(chat_model, tools, prompt=prompt)
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    def add_documents(self, documents: List[Document]):
        document_ids = self.vector_store.add_documents(documents)
        return document_ids

    def run(self, query: str):
        result = self.agent_executor.invoke({"input": query})
        print(result)
        print(type(result))
        return result


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
    log.info(f"Running RAG agent with query: {query}")
    agent.run(query)


if __name__ == "__main__":
    main(logger)
