import asyncio
import datetime
import json
import os
from abc import ABC, abstractmethod
from typing import Annotated, List, Sequence, TypedDict, Union

from langchain import hub
from langchain.agents import create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
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
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.constants import END
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from loguru import logger
from ollama import ResponseError

from modules.prompt import agent_prompt
from modules.tools.baidu_search import BaiduSearchTool

SYSTEM_PROMPT = SystemMessage("你是一个乐于助人的助手。")


class AgentState(TypedDict):
    """代理的状态。"""

    # add_messages is a reducer
    # See https://github.langchain.ac.cn/langgraph/concepts/low_level/#reducers
    messages: Annotated[Sequence[BaseMessage], add_messages]
    intermediate_steps: int


async def main():
    tools = [
        BaiduSearchTool(max_results=10),
        # TavilySearchResults(max_results=1),
    ]

    prompt = agent_prompt
    # prompt = hub.pull("hwchase17/structured-chat-agent")
    logger.info(f"🤖 prompt: {prompt}")

    base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    model_name = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5:3b")
    llm = ChatOllama(base_url=base_url, model=model_name)
    llm = llm.bind_tools(tools)

    # model_node = LLMNode(llm, prompt)
    llm_agent = create_react_agent(llm, tools, prompt)
    tool_node = ToolNode(tools)

    # Define a new graph
    workflow = StateGraph(AgentState)

    # Define the two nodes we will cycle between
    workflow.add_node("agent", llm_agent)
    workflow.add_node(tool_node.name, tool_node)

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
            "continue": tool_node.name,
            # Otherwise we finish.
            "end": END,
        },
    )

    # We now add a normal edge from `tools` to `agent`.
    # This means that after `tools` is called, `agent` node is called next.
    workflow.add_edge(tool_node.name, "agent")

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
            SystemMessage(
                # f"""
                # 当前时间为{current_time()}。请使用中文回答。
                # 如果你不清楚答案，或者不确定答案，请使用可用使用的工具进行回答，你可以使用如下工具 {[t.name for t in tools]}
                # """
                f"当前时间为{current_time()}。请使用中文回答。"
                f"如果你不清楚答案，或者不确定答案，请使用可用使用的工具进行回答，你可以使用如下工具 {[t.name for t in tools]}"
            ),
            HumanMessage("2024年美国大选结果"),
            # HumanMessage("深圳天气如何"),
        ],
        "intermediate_steps": 10,
    }
    for stream in graph.stream(inputs, stream_mode="values"):
        message: Union[BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage]
        message = stream["messages"][-1]

        MESSAGE_ICON = {
            SystemMessage: "🧪",
            HumanMessage: "🙋",
            AIMessage: "🤖",
            ToolMessage: "🛠️",
        }

        logger.info(
            f"{MESSAGE_ICON.get(type(message), ' ')} [{message.type}] message: {message}"
        )

        if isinstance(message, AIMessage):
            logger.warning(f"🤖 [response] {message.content}")


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


class LLMNode(BaseNode):
    def __init__(
        self,
        model: Union[ChatOllama, BaseChatModel],
        system_prompt: PromptTemplate = None,
    ):
        self.model = model

        # if isinstance(system_prompt, PromptTemplate):
        #     system_prompt = system_prompt.render()
        print(type(system_prompt))
        print(system_prompt)
        exit()
        # if system_prompt is None:
        #     self.system_prompt = SystemMessage("你是一个乐于助人的助手。")
        # else:
        #     self.system_prompt = SystemMessage(system_prompt)
        self.system_prompt = SystemMessage("你是一个乐于助人的助手。")
        print("system_prompt", type(system_prompt))

    def run(self, state: AgentState, config: RunnableConfig):
        """Call the LLM powering our "agent".

        This function prepares the prompt, initializes the model, and processes the response.

        Args:
            state (State): 当前对话的状态。
            config (RunnableConfig): 模型运行配置。

        Returns:
            dict: A dictionary containing the model's response message.
        """

        logger.info(f"【调试】 [state] {state}")

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


def current_time():
    # 返回当前时间
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    asyncio.run(main())
