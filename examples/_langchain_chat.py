from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain.globals import set_debug, set_verbose

set_debug(True)
set_verbose(True)


def demo1():
    prompt = ChatPromptTemplate.from_template("告诉我一个关于{topic}的笑话")
    print("📖", type(prompt), prompt)

    model = ChatOllama(model="qwen2.5:3b")  # 初始化 ChatOllama 模型
    chain = prompt.pipe(model).pipe(StrOutputParser())

    # 单
    response = chain.invoke({"topic": "鸡"})
    print(response)

    # 流式传输
    for chunck in chain.stream({"topic": "鸡"}):
        print("✅", repr(chunck))


def demo2():
    """
    Few-shot
    提供一些少样本座位提示
    """

    system = """你是一位滑稽的喜剧演员。你的专长是敲门笑话。 \
    返回一个包含开场白（对“谁在那里？”的回答）和结尾笑点（对“<开场白>谁？”的回答）的笑话。

    以下是一些笑话的例子：

    example_user: 告诉我一个关于飞机的笑话
    example_assistant: {{"setup": "为什么飞机永远不会感到疲倦？", "punchline": "因为他们有休息的翅膀！", "rating": 2}}

    example_user: 告诉我另一个关于飞机的笑话
    example_assistant: {{"setup": "货物", "punchline": "货物“嗡嗡嗡”，但飞机“嗡嗡嗡”！", "rating": 10}}

    example_user: Now about caterpillars
    example_assistant: {{"setup": "毛毛虫", "punchline": "毛毛虫真的很慢，但看我变成蝴蝶，抢尽风头！", "rating": 5}}"""

    prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", "{input}")]
    )

    model = ChatOllama(model="qwen2.5:3b")  # 初始化 ChatOllama 模型
    chain = prompt.pipe(model).pipe(StrOutputParser())

    # 单
    response = chain.invoke({"input": "鸡"})
    print(response)


def demo3():
    examples = [
        HumanMessage("告诉我一个关于飞机的笑话", name="example_user"),
        AIMessage(
            "",
            name="example_assistant",
            tool_calls=[
                {
                    "name": "joke",
                    "args": {
                        "setup": "为什么飞机永远不会累？",
                        "punchline": "因为它们有休息翅膀！",
                        "rating": 2,
                    },
                    "id": "1",
                }
            ],
        ),
        # Most tool-calling models expect a ToolMessage(s) to follow an AIMessage with tool calls.
        ToolMessage("", tool_call_id="1"),
        # Some models also expect an AIMessage to follow any ToolMessages,
        # so you may need to add an AIMessage here.
        HumanMessage("Tell me another joke about planes", name="example_user"),
        AIMessage(
            "",
            name="example_assistant",
            tool_calls=[
                {
                    "name": "joke",
                    "args": {
                        "setup": "Cargo",
                        "punchline": "Cargo 'vroom vroom', but planes go 'zoom zoom'!",
                        "rating": 10,
                    },
                    "id": "2",
                }
            ],
        ),
        ToolMessage("", tool_call_id="2"),
        HumanMessage("Now about caterpillars", name="example_user"),
        AIMessage(
            "",
            tool_calls=[
                {
                    "name": "joke",
                    "args": {
                        "setup": "Caterpillar",
                        "punchline": "Caterpillar really slow, but watch me turn into a butterfly and steal the show!",
                        "rating": 5,
                    },
                    "id": "3",
                }
            ],
        ),
        ToolMessage("", tool_call_id="3"),
    ]

    system = """你是个搞笑的喜剧演员。你的专长是敲门笑话。 \
    返回一个包含开场白（对“谁在那里？”的回答）的笑话 \
    并且最后的笑点（对“<setup>谁？”的回应）。"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("placeholder", "{examples}"),
            ("human", "{input}"),
        ]
    )

    model = ChatOllama(model="qwen2.5:3b")  # 初始化 ChatOllama 模型
    chain = prompt.pipe(model).pipe(StrOutputParser())

    # 单
    response = chain.invoke({"input": "鸡"})
    print(response)


if __name__ == "__main__":
    # demo1()
    # demo2()
    demo3()
