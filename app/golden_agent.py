import os

from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chains.llm_requests import LLMRequestsChain
from langchain_ollama import ChatOllama, OllamaEmbeddings

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
    base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    model_name = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5:3b")
    llm_model = ChatOllama(base_url=base_url, model=model_name)  # 初始化 ChatOllama 模型

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
