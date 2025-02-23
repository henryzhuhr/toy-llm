from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.agents import initialize_agent, Tool
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import SpacyTextSplitter, RecursiveCharacterTextSplitter
from langchain.chains.retrieval_qa.base import VectorDBQA


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


def main():
    llm_model = ChatOllama(model="qwen2.5:7b")  # 初始化 ChatOllama 模型

    multiple_choice_prompt = PromptTemplate(
        template=MULTIPLE_CHOICE, input_variables=["question"]
    )
    choice_chain = LLMChain(
        llm=llm_model, prompt=multiple_choice_prompt, output_key="answer"
    )

    # 通过VectorDBQA让Tool支持问答
    loader = TextLoader("./data/ecommerce_faq.txt")
    documents = loader.load()
    # print(documents)
    # text_splitter = SpacyTextSplitter(chunk_size=256, pipeline="zh_core_web_sm") # pip install spacy && python -m spacy download zh_core_web_sm
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10, chunk_overlap=0, separators=["\n\n"]  # 自定义切分
    )
    texts = text_splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model="qwen2.5:3b")
    docsearch = FAISS.from_documents(texts, embeddings)
    faq_chain = VectorDBQA.from_chain_type(
        llm=llm_model, vectorstore=docsearch, verbose=True
    )

    # 模拟问答

    question = "我想买一件衣服，但是不知道哪个款式好看，你能帮我推荐一下吗？"
    print("🤖", choice_chain(question))

    question = "我有一张订单，订单号是 2022ABCDE，一直没有收到，能麻烦帮我查一下吗？"
    print("🤖", choice_chain(question))

    question = "请问你们的货，能送到三亚吗？大概需要几天？"
    print("🤖", choice_chain(question))

    question = "今天天气怎么样？"
    print("🤖", choice_chain(question))

    # 指定使用tools，llm，agent则是zero-shot"零样本分类"，不给案例自己推理
    # 而 react description，指的是根据你对于 Tool 的描述（description）进行推理（Reasoning）并采取行动（Action）
    agent = initialize_agent(
        tools, llm_model, agent="zero-shot-react-description", verbose=True
    )

    question = "我想买一件衣服，但是不知道哪个款式好看，你能帮我推荐一下吗？"
    result = agent.run(question)
    print("🤖", result)

    question = "请问你们的货，能送到三亚吗？大概需要几天？"
    result = agent.run(question)
    print("🤖", result)


# 模拟问关于订单
def search_order(input: str) -> str:
    return "订单状态：已发货；发货日期：2023-09-15；预计送达时间：2023-09-18"


# 模拟问关于推荐产品
def recommend_product(input: str) -> str:
    return f"红色连衣裙({input})"


# 模拟问电商faq
def faq(input: str) -> str:
    return "7天无理由退货"


# 创建了一个 Tool 对象的数组，把这三个函数分别封装在了三个 Tool 对象里面
# 并且定义了描述，这个 description 就是告诉 AI，这个 Tool 是干什么用的，会根据描述做出选择
tools = [
    Tool(
        name="搜索订单",
        func=search_order,
        description="当您需要回答有关客户订单的问题时很有用",
    ),
    Tool(
        name="推荐产品",
        func=recommend_product,
        description="当您需要回答有关产品推荐的问题时很有用",
    ),
    Tool(
        name="常见问题解答",
        func=faq,
        description="当您需要回答有关购物政策的问题时很有用，例如退换货政策、配送政策等。",
    ),
]


if __name__ == "__main__":
    main()
