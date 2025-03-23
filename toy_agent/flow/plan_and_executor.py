import os
import time
from collections import Counter
from typing import List, Literal, Union

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    ToolMessage,
)
from langchain_core.tools.base import BaseTool
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

# from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.tool_node import ToolNode
from loguru import logger

from toy_agent._state import PlanAndExecuteAgentState
from toy_agent.agent.dispatcher import ReActAgent
from toy_agent.agent.planner import Planner
from toy_agent.agent.processor import Processor
from toy_agent.agent.replanner import Replanner
from toy_agent.agent.tool_react import ToolCallReActAgent, ToolCallReActAgentState
from toy_agent.agent.tool_react_run import ToolReActExecutor
from toy_agent.flow import tool_react
from toy_agent.flow._base import BaseFlow
from toy_agent.flow.tool_react import ToolCallReActAgentFlow


class PlanAndExecutorFlow(BaseFlow):
    name: str = "plan_and_execute_agent"

    llm: Union[BaseChatModel, ChatOllama] = None
    tools: List[BaseTool] = None
    tool_names: List[str] = None

    def __init__(
        self, llm: Union[BaseChatModel, ChatOllama], tools: List[BaseTool] = None
    ):
        super().__init__()

        self.llm = llm
        self.tools = tools or []
        self.tool_names = [t.name for t in tools]
        self._validate_tools(self.tool_names)

        if len(tools) > 0:
            self.llm = self.llm.bind_tools(tools)

    def build_workflow(self, **kwargs):  # noqa: C901
        llm = self.llm
        tools = self.tools

        workflow = StateGraph(PlanAndExecuteAgentState)
        # --- add nodes ---
        planner = Planner(llm)
        workflow.add_node(planner.name, planner)
        workflow.set_entry_point(planner.name)

        tool_react_agent = ToolCallReActAgentFlow(llm, tools).build_workflow()
        tool_react_executor = ToolReActExecutor(tool_react_agent)
        workflow.add_node(
            tool_react_agent.name,
            tool_react_executor.get_arun(),
        )

        replanner = Replanner(llm)
        workflow.add_node(replanner.name, replanner)

        # --- add edges ---
        workflow.add_edge(START, planner.name)
        workflow.add_edge(planner.name, tool_react_agent.name)
        workflow.add_edge(tool_react_agent.name, replanner.name)

        def should_end(state: PlanAndExecuteAgentState):
            if state.response:
                return END
            else:
                return tool_react_executor.name

        workflow.add_conditional_edges(
            replanner.name,
            # Next, we pass in the function that will determine which node is called next.
            should_end,
            [tool_react_executor.name, END],
        )
        # --- compile ---
        compiled_state_graph = workflow.compile(
            debug=False,
            name="plan_and_execute_agent_test",
        )
        return compiled_state_graph

    @staticmethod
    async def _arun_tool_react(state: PlanAndExecuteAgentState, config):
        # async for event in tool_react_agent.astream(state, config=config):
        #     for k, v in event.items():
        #         pass
        return state

    @staticmethod
    def inject_tool_react_agent(graph: CompiledStateGraph):
        """
        装饰器：用于向静态方法注入 tool_react_agent。
        """

        def decorator(func):
            async def wrapper(state, config):
                logger.debug(f"[{graph.name}]  state: {state}")
                async for event in graph.astream(state, config=config):
                    for k, v in event.items():
                        pass  # 可以根据需要处理每个事件
                return await func(state, config)

            return wrapper

        return decorator
