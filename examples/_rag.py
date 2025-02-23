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
from langchain_community.utils.user_agent import get_user_agent

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
        "User-Agent": get_user_agent(),
        "accept": "*/*",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Connection": "keep-alive",
        "Cookie": "__snaker__id=DnqiF4KmSbUJYUAp; SESSIONID=2zthsnPGtFtDRjZYNu28FNzpFR1Rwxk8l4RKTwXMiOt; JOID=Vl8QAEIjE9_iHW9KEiUtTAAyY00JAzL6yzNPazcNM_7HNEFqM9RTZIITbk4e-CPYunx9kZd_lc9JrZR21fWfvnQ=; osd=Wl8UCkwvE9voE2NKFi8jQAA2aUMFAzbwxT9Pbz0DP_7DPk9mM9BZao4TakQQ9CPcsHJxkZN1m8NJqZ542fWbtHo=; _xsrf=LxDYMY3yy09dN7aJscXebRQAaOMP5a7q; __zse_ck=004_Go5oO92Zg6nv6ysxhQaMKQGEJiDhE75/9BCxf7hpLJA2pxUnTRaMcKqoU9P2L9FwyBaEebVlvIp3c61/1WpiCczWaECqJH8hu2HhZ9os3453fgvM3BfiuCKHgdcBYmo5-lVspeovIXMkTj2FKF62Xcj847WqIjSfdfSV6LRGX+myIdrMLMFMu7pxS/IKAEKuwpr0hDTLDr9mE4HvQ0dBBtSBxDs/rL7YA+3z0lsQZest4JtSGE2ItaPiV2VjrBr1U; _zap=984192ff-9732-4724-a161-843c6cf32ff8; d_c0=A_BReyILCRqPTnzf1toYxP4BYiZDBPg7yxc=|1740063497; captcha_session_v2=2|1:0|10:1740064017|18:captcha_session_v2|88:VGxBNmU1Nmo1M24vS24xcFVSdGVqclkvY3o0RVpMRC8wQUNqWXZzTnU4Zkc4MUxiQStNQzVSRS9LOUIyVW9Kbw==|d7757d203d2a8d0401ae4b9abaab8edf09f0a14db30a5afef7f3e1c75481a5b0; Hm_lvt_98beee57fd2ef70ccdd5ca52b9740c49=1740063501; Hm_lpvt_98beee57fd2ef70ccdd5ca52b9740c49=1740064018; HMACCOUNT=88220C50536B79E0; gdxidpyhxdE=CBN475EojonH2KS%2Ffh2f4BB23pMg%2F4BM%5Czk1Kk6tloIoTRvV4xpuT8dxa%2Fc4vTrYpkZSMcdVLGpv0jTOzBQSLWALqGOhVeA%5CEPLnM6kram5mEgKi%5C9Tfh4MI2E7DlVUKX%2Fk6ux89ddbf5JR6je0Dx9g3YGKQblyYj4ThZBtjV4gU2y87%3A1740145932813; __snaker__id=COzRgNZwksi9FvYr; BEC=d6322fc1daba6406210e61eaa4ec5a7a",
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
        model="deepseek-r1:latest",
    )

    rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

    retriever = vectorstore.as_retriever()

    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | model
        | StrOutputParser()
    )

    questions = [
        "小行星的危害",
        "小行星的直径是多少",
        "如何评价美国的行为",
        "特朗普是谁",
        "2024年美国总统选举",
    ]
    for question in questions:
        answer = qa_chain.invoke(question)
        print("🤖", answer)
        print()

    return
    while True:
        question = input("请输入问题: ")
        if question == "exit":
            break
        answer = qa_chain.invoke(question)
        print("🤖", answer)
        print()


# 将传入的文档转换成字符串的形式
def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":
    main()
