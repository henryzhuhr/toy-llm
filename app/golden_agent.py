from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.chains.llm_requests import LLMRequestsChain
from langchain.chains.llm import LLMChain

TEMPLATE = """在 >>> 和 <<< 之间是网页的返回的HTML内容。
网页是实时金价的信息
请抽取参数请求的信息。

>>> {requests_result} <<<
请使用如下的JSON格式返回数据
{{
"国内黄金价格":"111.60",
"纽约期货国际金价":"1111.00",
"伦敦现货黄金价格":"1111.00",
}}
Extracted:"""


def main():
    llm_model = ChatOllama(model="qwen2.5:7b")  # 初始化 ChatOllama 模型

    prompt = PromptTemplate(
        input_variables=["requests_result"],
        template=TEMPLATE,
    )

    # 网页获取 chain
    chain = LLMRequestsChain(
        llm_chain=LLMChain(
            llm=llm_model,
            prompt=prompt,
            verbose=True,
        )
    )
    inputs = {"url": "https://www.huilvbiao.com/gold/30days"}

    response = chain.invoke(inputs)
    print(response["output"])


if __name__ == "__main__":
    main()
