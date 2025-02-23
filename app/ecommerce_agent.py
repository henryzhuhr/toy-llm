"""
https://github.com/xuwenhao/geektime-ai-course/blob/main/17_langchain_agent.ipynb
"""

from langchain_ollama import ChatOllama, OllamaEmbeddings

from langchain.agents import initialize_agent, Tool
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.retrieval_qa.base import RetrievalQA
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langgraph.checkpoint.memory import MemorySaver

MULTIPLE_CHOICE = """
请针对 >>> 和 <<< 中间的用户问题，选择一个合适的工具去回答它的问题。只要用A、B、C的选项字母告诉我答案。
如果你觉得都不合适，就选D。
>>>{question}<<<
我们有的工具包括：
A. 一个能够查询商品信息，为用户进行商品导购的工具
B. 一个能够查询订单信息，获得最新的订单情况的工具
C. 一个能够搜索商家的退换货政策、运费、物流时长、支付渠道、覆盖国家的工具
D. 都不合适
"""

# 电商客服代理
# E-commerce customer service agent
# ecommerce_agent


def main():
    llm_model = ChatOllama(model="qwen2.5:7b")  # 初始化 ChatOllama 模型

    ecommerce_agent = EcommerceAgent()

    memory = MemorySaver()
    agent_executor = create_react_agent(
        llm_model,
        tools=ecommerce_agent.tools,
        # checkpointer=memory,
    )

    messages = [
        # (
        #     "user",
        #     "我有一张订单，订单号是 2022ABCDE，一直没有收到，能麻烦帮我查一下吗？",
        # )
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
        messages.append(("user", question))
        inputs = {"messages": messages}
        
        """
        如前所述，此代理是无状态的。这意味着它不会记住之前的交互。
        为了给它提供记忆，我们需要传递一个检查点器。
        在传递检查点器时，我们还需要在调用代理时传递一个线程_id（这样它就知道从哪个线程/对话中恢复）。
        """
        config = {"configurable": {"thread_id": "abc123"}}
        stream = agent_executor.stream(
            inputs,
            # config,
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
                        print(f" - [{tool_call["name"]}] {tool_call["args"]}")
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
    def __init__(self):
        # 示例使用
        ecommerce_functions = EcommerceFunctions()

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
    def __init__(self, model="qwen2.5:3b", fqa_file="./data/ecommerce_faq.txt"):
        # 通过 RetrievalQA 让Tool支持问答
        loader = TextLoader(fqa_file)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10, chunk_overlap=0, separators=["\n\n"]  # 自定义切分
        )
        texts = text_splitter.split_documents(documents)
        embeddings = OllamaEmbeddings(model=model)
        docsearch = FAISS.from_documents(texts, embeddings)
        llm_model = ChatOllama(model=model)  # 初始化 ChatOllama 模型
        self.qa = RetrievalQA.from_chain_type(
            llm=llm_model,
            retriever=docsearch.as_retriever(),  # 传入retriever
            verbose=False,
        )


class EcommerceFunctions:
    def __init__(
        self, fqa_model="qwen2.5:3b", fqa_file="./data/ecommerce_faq.txt"
    ):
        self.fqa_tools = FQATools(fqa_model, fqa_file)

    # 模拟问关于订单
    @staticmethod
    def search_order(input: str) -> str:
        return (
            "订单状态：已发货；发货日期：2023-09-15；预计送达时间：2023-09-18"
        )

    # 模拟问关于推荐产品
    @staticmethod
    def recommend_product(input: str) -> str:
        return f"红色连衣裙({input})"

    def faq(self, input: str) -> str:
        """ "useful for when you need to answer questions about shopping policies, like return policy, shipping policy, etc."""
        return self.fqa_tools.qa.invoke(input)


if __name__ == "__main__":
    main()
