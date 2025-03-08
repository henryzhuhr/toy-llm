import asyncio
import json
import os
from abc import ABC, abstractmethod
from json import tool
from typing import Annotated, List, Sequence, TypedDict, Union

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.constants import END
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from loguru import logger
from ollama import ResponseError

from modules.prompt import agent_prompt
from modules.tools.baidu_search import BaiduSearchTool


async def main():

    tools = [
        BaiduSearchTool(max_results=5),
    ]
    tools_by_name = {tool.name: tool for tool in tools}

    prompt = agent_prompt

    try:
        llm = ChatOllama(
            base_url="http://host.docker.internal:11434",
            # base_url="http://localhost:11434",
            model="qwen2.5:7b",
        )  # 初始化 ChatOllama 模型
        llm.invoke([HumanMessage("你好")])
    except ResponseError as e:
        logger.error(f"🤖 ChatOllama 初始化失败: {e}")
        return

    model_node = LLMNode(llm, prompt)
    tool_node = ToolNode(tools_by_name)

    # Define a new graph
    workflow = StateGraph(AgentState)

    # Define the two nodes we will cycle between
    workflow.add_node("agent", model_node.run)
    workflow.add_node("tools", tool_node.run)

    # Set the entrypoint as `agent`
    # This means that this node is the first one called
    workflow.set_entry_point("agent")

    # We now add a conditional edge
    workflow.add_conditional_edges(
        # First, we define the start node. We use `agent`.
        # This means these are the edges taken after the `agent` node is called.
        "agent",
        # Next, we pass in the function that will determine which node is called next.
        should_continue,
        # Finally we pass in a mapping.
        # The keys are strings, and the values are other nodes.
        # END is a special node marking that the graph should finish.
        # What will happen is we will call `should_continue`, and then the output of that
        # will be matched against the keys in this mapping.
        # Based on which one it matches, that node will then be called.
        {
            # If `tools`, then we call the tool node.
            "continue": "tools",
            # Otherwise we finish.
            "end": END,
        },
    )

    # We now add a normal edge from `tools` to `agent`.
    # This means that after `tools` is called, `agent` node is called next.
    workflow.add_edge("tools", "agent")

    # Now we can compile and visualize our graph
    graph = workflow.compile()

    try:
        graph_img = graph.get_graph().draw_mermaid_png()
        # 保存图片
        os.makedirs("tmp", exist_ok=True)
        with open("tmp/graph.png", "wb") as f:
            f.write(graph_img)
    except Exception:
        # This requires some extra dependencies and is optional
        pass

    inputs = {
        "messages": [
            HumanMessage("谁是美网的冠军？"),
        ]
    }
    for stream in graph.stream(inputs, stream_mode="values"):
        message = stream["messages"][-1]
        print(message)


class AgentState(TypedDict):
    """The state of the agent."""

    # add_messages is a reducer
    # See https://github.langchain.ac.cn/langgraph/concepts/low_level/#reducers
    messages: Annotated[Sequence[BaseMessage], add_messages]


class BaseNode(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def run(self, state: AgentState):
        raise NotImplementedError


class ToolNode(BaseNode):
    def __init__(self, tools_by_name):
        self.tools_by_name = tools_by_name

    def run(self, state: AgentState):
        outputs = []
        for tool_call in state["messages"][-1].tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


class LLMNode(BaseNode):
    def __init__(
        self,
        model: Union[ChatOllama, BaseChatModel],
        system_prompt: PromptTemplate = None,
    ):
        self.model = model
        # if system_prompt is None:
        #     self.system_prompt = SystemMessage("你是一个乐于助人的助手。")
        # else:
        #     self.system_prompt = SystemMessage(system_prompt)
        self.system_prompt = SystemMessage("你是一个乐于助人的助手。")
        print("system_prompt", type(system_prompt))

    def run(self, state: AgentState):
        response = self.model.invoke([self.system_prompt] + state["messages"])
        return {"messages": [response]}


# Define the conditional edge that determines whether to continue or not
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


if __name__ == "__main__":
    asyncio.run(main())
