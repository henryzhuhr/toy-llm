import os
import time
from typing import Literal, Union

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.tool_node import ToolNode
from loguru import logger

from modules.tools.baidu_search import BaiduSearchTool
from toy_agent import AgentState, Executor, Planner, Replanner
from toy_agent.agent.dispatcher import ReActAgent
from toy_agent.flow._base import BaseFlow


class ReActAgentFLow(BaseFlow):
    name: str = "plan_and_execute_agent"

    def __init__(self):
        super().__init__()

    def build_workflow(self):  # noqa: C901
        workflow = StateGraph(AgentState)
        base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
        model_name = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5:1.5b")
        llm = ChatOllama(base_url=base_url, model=model_name)

        tools = [
            TavilySearchResults(max_results=3),
            BaiduSearchTool(max_results=3),
        ]
