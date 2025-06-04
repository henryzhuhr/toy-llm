import os
from typing import Annotated, List, Optional, Union

from langchain_community.tools import CopyFileTool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

memory = MemorySaver()


class State(TypedDict):
    # Messages have the type "list".
    # The `add_messages` function in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


class ChatBotNode:
    def __init__(self, tools: List[BaseTool] = []):
        self.llm = ChatOllama(
            base_url="http://host.docker.internal:11434", model="qwen2.5:3b"
        )  # 初始化 ChatOllama 模型
        self.llm = self.llm.bind_tools(tools)

    def chatbot(self, state: State):
        return {"messages": [self.llm.invoke(state["messages"])]}


def stream_graph_updates(graph: CompiledGraph, user_input: str):
    config = {"configurable": {"thread_id": "1"}}
    # The config is the **second positional argument** to stream() or invoke()!
    events = graph.stream(
        {"messages": [("user", user_input)]}, config, stream_mode="values"
    )

    assistant: Optional[Union[AIMessage, HumanMessage, ToolMessage]] = None
    for event in events:
        assistant = event["messages"][-1]
        # print(
        #     "🤖 Assistant:", type(event["messages"][-1]), event["messages"][-1]
        # )
    print("🤖 Assistant:", repr(assistant.content))
    if len(assistant.tool_calls) > 0:
        print("🔧 Tool:", assistant.tool_calls)
    print()


def main():
    graph_builder = StateGraph(State)

    # 工具
    copy_file_tool = CopyFileTool(root_dir="/tmp/tmprdvsw3tg")
    # 工具包: https://python.langchain.ac.cn/docs/integrations/tools/
    tools = [
        copy_file_tool,
    ]

    # 初始化节点
    chatbot = ChatBotNode(tools)

    graph_builder.add_node("chatbot", chatbot.chatbot)
    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    graph_builder.add_edge("chatbot", END)

    graph = graph_builder.compile(
        checkpointer=memory,
        # This is new!
        interrupt_before=["tools"],
        # Note: can also interrupt __after__ tools, if desired.
        # interrupt_after=["tools"]
    )

    try:
        graph_img = graph.get_graph().draw_mermaid_png()
        # 保存图片
        os.makedirs("tmp", exist_ok=True)
        with open("tmp/graph.png", "wb") as f:
            f.write(graph_img)
    except Exception:
        # This requires some extra dependencies and is optional
        pass

    user_inputs = [
        "你是谁",
        "我是马冬梅",
        "帮我记住我的车停在 C 区 3 号车位",
        "我是谁",
        "我在哪里停车了",
        "请帮我复制文件到系统 `/tmp` 目录",
        "帮我从 json 中获取 key 为 `name` 的值，json 内容为 `{'name': '马冬梅'}`",
    ]
    user_input_iter = iter(user_inputs)
    while True:
        try:
            user_input = next(user_input_iter)
            print("🙋 User: " + user_input)
            if user_input is None:
                break
            # user_input = input("User: ")
            # if user_input.lower() in ["quit", "exit", "q"]:
            #     print("Goodbye!")
            #     break

            stream_graph_updates(graph, user_input)
        except:
            # # fallback if input() is not available
            # user_input = "What do you know about LangGraph?"
            # print("User: " + user_input)
            # stream_graph_updates(graph, user_input)
            break


if __name__ == "__main__":
    main()
