"""
https://github.com/xuwenhao/geektime-ai-course/blob/main/17_langchain_agent.ipynb
"""

import os
import time
from datetime import datetime, timedelta
from typing import List, Optional

from langchain.agents import Tool
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel


class ModelConfig(BaseModel):
    # base_url: str = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model_name: str = os.getenv("OLLAMA_MODEL_NAME", "qwen3:1.7b")


def main():
    model_config = ModelConfig()

    # 初始化 ChatOllama 模型
    llm_model = ChatOllama(
        base_url=model_config.base_url,
        model=model_config.model_name,
    )

    # 初始化电商客服机器人
    ecommerce_agent = EcommerceAgent(model_config)

    # 添加记忆
    memory = MemorySaver()

    # 创建一个 REACT 代理
    agent_executor = create_react_agent(
        llm_model,
        tools=ecommerce_agent.tools,
        checkpointer=memory,
    )

    # 全局提示词
    messages: List[AnyMessage] = [
        SystemMessage(
            f"你是一个智能客服助手（工号 {int(time.time())}），你需要帮助用户回答一些问题。今天的日期是 {time.strftime('%Y-%m-%d')}。"
        ),
    ]

    for question in [
        "你是谁",
        "我有一张订单，订单号是 2022ABCDE，一直没有收到，能麻烦帮我查一下吗？",
        "快递多久能到",
        "什么快递发货",
        "请问你们的货，能送到新疆吗？大概需要几天？",
        "今天天气怎么样？",
        "我想买一件衣服，但是不知道哪个款式好看，你能帮我推荐一下吗？",
        "提供哪些类型的发票？",
        "收到货后怎么退货？",
        "刚才那个订单号是多少",
    ]:
        messages.append(HumanMessage(question))
        inputs = {"messages": messages}

        """
        如前所述，此代理是无状态的。这意味着它不会记住之前的交互。
        为了给它提供记忆，我们需要传递一个检查点器。
        在传递检查点器时，我们还需要在调用代理时传递一个线程_id（这样它就知道从哪个线程/对话中恢复）。
        """
        config = RunnableConfig(configurable={"thread_id": "abc123"})
        stream = agent_executor.stream(
            inputs,
            config,
            stream_mode="values",
        )
        assistant: Optional[AIMessage] = None

        for s in stream:
            message: AnyMessage = s["messages"][-1]

            if isinstance(message, HumanMessage):
                print(
                    f"🙋\033[01;34m【用户问题】{message.content}\033[0m",
                )
            elif isinstance(message, ToolMessage):
                print("🔧「调用结果」", message.content)
            elif isinstance(message, AIMessage):
                if message.tool_calls:
                    print("🤖🔧", message.tool_calls)
                    for tool_call in message.tool_calls:
                        print(f" - [{tool_call['name']}] {tool_call['args']}")
                else:
                    assistant = message  # .content
            else:
                print("❌", message.type, message)
        if assistant:
            print(
                f"🤖\033[01;32m【客服回答】{assistant.content}\033[0m",
                # f"🤖\033[01;32m【客服回答】{repr(assistant.content)}\033[0m",
            )
            messages.append(AIMessage(content=assistant.content))
        print()
        # input("next:")
    return


# 电商客服机器人
class EcommerceAgent:
    def __init__(self, model_config: ModelConfig):
        # 示例使用
        ecommerce_functions = EcommerceFunctions(model_config)

        # 定义一些工具
        self.tools = [
            Tool(
                name="搜索订单",
                func=EcommerceFunctions.search_order,
                description="当您需要回答有关客户订单的问题时很有用",
            ),
            Tool(
                name="推荐产品",
                func=EcommerceFunctions.recommend_product,
                description="当您需要回答有关产品推荐的问题时很有用",
            ),
            Tool(
                name="常见问题解答",
                func=ecommerce_functions.faq,
                description="当您需要回答有关购物政策的问题时很有用，例如退换货政策、配送政策、快递物流信息等。",
            ),
            Tool(
                name="今日天气查询",
                func=ecommerce_functions.weather,
                description="当您需要回答有关天气的问题时很有用，例如今天的天气情况。",
            ),
        ]


class FQATools:
    """
    用于处理电商常见问题的工具类
    通过加载 FAQ 文档，使用 OllamaEmbeddings 和 FAISS 创建向量存储，
    """

    def __init__(self, model_config: ModelConfig, fqa_file="./data/ecommerce_faq.txt"):
        # 加载FAQ文档
        loader = TextLoader(fqa_file)
        documents = loader.load()

        # 切分文档
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10,
            chunk_overlap=0,
            separators=["\n\n"],  # 自定义切分
        )
        texts = text_splitter.split_documents(documents)

        # 使用 FAISS 创建向量存储
        embeddings = OllamaEmbeddings(
            base_url=model_config.base_url, model=model_config.model_name
        )
        docsearch = FAISS.from_documents(texts, embeddings)

        # 初始化 ChatOllama 模型
        llm_model = ChatOllama(
            base_url=model_config.base_url, model=model_config.model_name
        )

        # 创建简单的 RAG 检索问答
        self.qa = RetrievalQA.from_chain_type(
            llm=llm_model,
            retriever=docsearch.as_retriever(),  # 传入retriever
            verbose=False,
        )


class EcommerceFunctions:
    def __init__(self, model_config: ModelConfig, fqa_file="./data/ecommerce_faq.txt"):
        self.fqa_tools = FQATools(model_config, fqa_file)

    # 模拟问关于订单
    @staticmethod
    def search_order(input: str) -> str:
        print(f"【工具调用】模拟订单查询, input={input}")
        current_datetime = datetime.now()
        cddate = current_datetime.strftime("%Y-%m-%d")
        cddate_add_7d = (current_datetime + timedelta(days=7)).strftime("%Y-%m-%d")
        return f"订单状态：已发货；发货日期：{cddate}；预计送达时间：{cddate_add_7d}"

    # 模拟问关于推荐产品
    @staticmethod
    def recommend_product(input: str) -> str:
        print(f"【工具调用】模拟产品推荐, input={input}")
        return "蓝色格子衫"

    # 自由问答
    def faq(self, input: str) -> str:
        print(f"【工具调用】模拟常见问题解答, input={input}")
        return self.fqa_tools.qa.invoke(input)  # type: ignore

    # 模拟今天天气的查询
    def weather(self, input: str) -> str:
        print(f"【工具调用】模拟天气查询, input={input}")
        location = "深圳市"  # 模拟获取当前用户位置
        current_datetime = datetime.now()
        cddate = current_datetime.strftime("%Y-%m-%d")
        return f"今天是 {cddate}，天气晴，{location}的气温 25°C，有小雨。"


if __name__ == "__main__":
    main()
