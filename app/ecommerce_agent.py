"""
https://github.com/xuwenhao/geektime-ai-course/blob/main/17_langchain_agent.ipynb
"""

import os
import time

from langchain.agents import Tool, initialize_agent
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel


class ModelConfig(BaseModel):
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    model_name: str = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5:3b")


def main():
    model_config = ModelConfig()
    llm_model = ChatOllama(  # 初始化 ChatOllama 模型
        base_url=model_config.base_url, model=model_config.model_name
    )

    ecommerce_agent = EcommerceAgent(model_config)

    memory = MemorySaver()
    agent_executor = create_react_agent(
        llm_model,
        tools=ecommerce_agent.tools,
        checkpointer=memory,
    )

    messages = [
        # (
        #     "user",
        #     "我有一张订单，订单号是 2022ABCDE，一直没有收到，能麻烦帮我查一下吗？",
        # )
        SystemMessage(
            f"你是一个智能客服助手（工号 {time.time()}），你需要帮助用户回答一些问题。"
        ),
    ]

    for question in [
        "我有一张订单，订单号是 2022ABCDE，一直没有收到，能麻烦帮我查一下吗？",
        "物流时效是多久？",
        # "请问你们的货，能送到三亚吗？大概需要几天？",
        # "今天天气怎么样？",
        # "我想买一件衣服，但是不知道哪个款式好看，你能帮我推荐一下吗？",
        "提供哪些类型的发票？",
        # "优惠券有使用限制吗？",
        "你是谁",
        "美国大选是怎么进行的",
        "刚才那个订单号是多少",
    ]:
        messages.append(HumanMessage(question))
        inputs = {"messages": messages}

        """
        如前所述，此代理是无状态的。这意味着它不会记住之前的交互。
        为了给它提供记忆，我们需要传递一个检查点器。
        在传递检查点器时，我们还需要在调用代理时传递一个线程_id（这样它就知道从哪个线程/对话中恢复）。
        """
        config = {"configurable": {"thread_id": "abc123"}}
        stream = agent_executor.stream(
            inputs,
            config,
            stream_mode="values",
        )
        assistant: str = None

        for s in stream:
            message: BaseMessage = s["messages"][-1]

            if isinstance(message, HumanMessage):
                print("🙋【用户问题】", message.content)
            elif isinstance(message, ToolMessage):
                print("🔧【调用结果】", message.content)
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
            print("🤖【客服回答】", repr(assistant.content))
            messages.append(("assistant", assistant.content))
        print()
    return

    # 指定使用tools，llm，agent则是zero-shot"零样本分类"，不给案例自己推理
    # 而 react description，指的是根据你对于 Tool 的描述（description）进行推理（Reasoning）并采取行动（Action）
    agent = initialize_agent(
        ecommerce_agent.tools,
        llm_model,
        agent="zero-shot-react-description",
        verbose=False,
    )


# 电商客服机器人
class EcommerceAgent:
    def __init__(self, model_config: ModelConfig):
        # 示例使用
        ecommerce_functions = EcommerceFunctions(model_config)

        # 创建了一个 Tool 对象的数组，把这三个函数分别封装在了三个 Tool 对象里面
        # 并且定义了描述，这个 description 就是告诉 AI，这个 Tool 是干什么用的，会根据描述做出选择
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
                description="当您需要回答有关购物政策的问题时很有用，例如退换货政策、配送政策等。",
            ),
        ]


class FQATools:
    def __init__(self, model_config: ModelConfig, fqa_file="./data/ecommerce_faq.txt"):
        # 通过 RetrievalQA 让Tool支持问答
        loader = TextLoader(fqa_file)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10,
            chunk_overlap=0,
            separators=["\n\n"],  # 自定义切分
        )
        texts = text_splitter.split_documents(documents)
        embeddings = OllamaEmbeddings(
            base_url=model_config.base_url, model=model_config.model_name
        )
        docsearch = FAISS.from_documents(texts, embeddings)
        llm_model = ChatOllama(  # 初始化 ChatOllama 模型
            base_url=model_config.base_url, model=model_config.model_name
        )
        self.qa = RetrievalQA.from_chain_type(
            llm=llm_model,
            retriever=docsearch.as_retriever(),  # 传入retriever
            verbose=False,
        )


class EcommerceFunctions:
    def __init__(self, fqa_model="qwen2.5:3b", fqa_file="./data/ecommerce_faq.txt"):
        self.fqa_tools = FQATools(fqa_model, fqa_file)

    # 模拟问关于订单
    @staticmethod
    def search_order(input: str) -> str:
        return "订单状态：已发货；发货日期：2023-09-15；预计送达时间：2023-09-18"

    # 模拟问关于推荐产品
    @staticmethod
    def recommend_product(input: str) -> str:
        return f"红色连衣裙({input})"

    def faq(self, input: str) -> str:
        """ "useful for when you need to answer questions about shopping policies, like return policy, shipping policy, etc."""
        return self.fqa_tools.qa.invoke(input)


if __name__ == "__main__":
    main()
