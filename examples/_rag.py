"""
https://github.com/datawhalechina/handy-ollama/blob/main/docs/C7/7.%20使用%20DeepSeek%20R1%20和%20Ollama%20实现本地%20RAG%20应用.md
"""
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


RAG_TEMPLATE = """
您是问答任务的助手。使用以下检索到的上下文来回答问题。如果您不知道答案，只需说不知道。

<context>
{context}
</context>

回答以下问题:

{question}"""


def main():
    print("Welcome to the Ollama Chatbot!")

    headers = {
        "User-Agent": "",
        "accept": "*/*",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Connection": "keep-alive",
        "Cookie": "",
        "Priority": "u=0, i",
    }
    loader = WebBaseLoader(
        web_path="https://zhuanlan.zhihu.com/p/22922535643",
        header_template=headers,
    )

    documents = loader.load()
    for doc in documents:
        # print("🟢", doc.page_content)
        print("🟢", doc.metadata)
        print("🟢📖", doc.page_content)

    print()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(documents)
    for doc in documents:
        # print("🟢", doc.page_content)
        print("🟡", doc.metadata)
        print("🟡📖", doc.page_content)

    local_embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vectorstore = Chroma.from_documents(
        documents=all_splits, embedding=local_embeddings
    )

    model = ChatOllama(
        model="qwen2.5:3b",
    )

    rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

    retriever = vectorstore.as_retriever()

    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | model
        | StrOutputParser()
    )

    questions = ["小行星的危害", "小行星的直径是多少", "如何评价美国的行为"]
    for question in questions:
        answer = qa_chain.invoke(question)
        print("🤖", answer)


# 将传入的文档转换成字符串的形式
def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":
    main()
