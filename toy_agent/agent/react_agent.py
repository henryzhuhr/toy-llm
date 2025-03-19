import os
import time
from datetime import datetime
from typing import List, Literal, Union

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.tools.base import BaseTool
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.managed import IsLastStep, RemainingSteps
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.tool_node import ToolNode
from loguru import logger
from pydantic import BaseModel
from typing_extensions import Annotated

from modules.tools.baidu_search import BaiduSearchTool
from toy_agent import AgentState, Executor, Planner, Replanner
from toy_agent._state import AgentState, Plan
from toy_agent.agent._base import BaseNode
from toy_agent.agent.dispatcher import ReActAgent
from toy_agent.flow._base import BaseFlow
from toy_agent.prompt import PROMPTS


class ReActAgentState(BaseModel):
    """The state of the agent."""

    messages: Annotated[List[BaseMessage], add_messages]

    is_last_step: IsLastStep

    remaining_steps: RemainingSteps


class ReActAgent(BaseNode):
    name: str = "ReActAgent"

    llm: BaseChatModel = None
    tools: List[BaseTool] = None

    def __init__(self, llm: BaseChatModel, tools: List[BaseTool] = None):
        super().__init__()
        self.llm = llm
        self.tools = tools or []

        if tools:
            self.llm = self.llm.bind_tools(tools)

    async def __call__(self, state: ReActAgentState, config: RunnableConfig) -> ReActAgentState:
        logger.debug(f"[{self.name}]  state: {state}")
        logger.debug(f"[{self.name}] config: {config.keys()}")

        messages = state.messages

        return AgentState()


class ReActAgentFLow(BaseFlow):
    name: str = "plan_and_execute_agent"

    llm: BaseChatModel = None
    tools: List[BaseTool] = None

    def __init__(self, llm: BaseChatModel, tools: List[BaseTool] = None):
        super().__init__()

        self.llm = llm
        self.tools = tools or []
        self.tool_names = [t.name for t in tools]  # TODO: 检查是否有重复 tool 名称

        if tools:
            self.llm = self.llm.bind_tools(tools)

    def build_workflow(self):  # noqa: C901
        workflow = StateGraph(ReActAgentState)


if __name__ == "__main__":
    base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    model_name = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5:1.5b")
    llm = ChatOllama(base_url=base_url, model=model_name)
    flow = ReActAgentFLow(
        llm,
        tools=[
            TavilySearchResults(max_results=3),
            BaiduSearchTool(max_results=3),
        ],
    )
    flow.build_workflow()
