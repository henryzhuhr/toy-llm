import json
import os
import pandas as pd
from ollama import Client

# 定义文件路径
CSV_FILE = "/Users/henryzhu/Downloads/benchmark-事件列表.csv"

TOOLS = [

]

SYSTEM_PROMPT = """在大型语言模型（LLM）的应用中，函数调用技术是一种重要的能力，它允许模型与外部工具或API进行交互。通过这种方式，模型可以执行超出其训练数据范围的任务，例如查询实时信息、操作数据库、调用特定功能等。这种技术极大地扩展了LLM的应用场景和实用性。
函数调用的基本原理
定义函数接口 ：首先需要定义一个或多个函数接口，这些接口描述了函数的名称、输入参数以及输出结果。通常这些接口会以JSON Schema的形式提供给模型。
模型生成调用请求 ：当用户提出一个问题或任务时，如果该任务涉及到外部函数调用，模型会根据预定义的函数接口生成相应的调用请求。这个请求通常是一个包含函数名和参数的结构化数据（如JSON对象）。
现在我提供给你JSON Schema定义的接口，请根据这个接口帮我想一下，用户可能会问哪些问题，尽可能覆盖不同的参数组合
"""


def main():

    client = Client(
        # host="http://ollama-server:11434",
        host="http://localhost:11434",
        headers={"x-some-header": "some-value"},
    )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {"role": "user", "content": "深圳的天气怎么样？"},
    ]
    print(messages)

    tools = TOOLS

    model_name = "qwen2.5:3b"

    for tool in tools:
        function_name = tool.get("function").get("name")
        print("🔵", function_name)


if __name__ == "__main__":
    main()
