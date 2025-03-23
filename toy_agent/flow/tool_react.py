from collections import Counter
from typing import List, Union

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
from pydantic import ConfigDict

from toy_agent._state import PlanAndExecuteAgentState
from toy_agent.agent.tool_react import ToolCallReActAgent, ToolCallReActAgentState
from toy_agent.agent.tool_react_run import ToolReActExecutor
from toy_agent.flow._base import BaseFlow


class ToolCallReActAgentFlow(BaseFlow):
    name: str = "tool_call_ReAct_agent"

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

    def build_workflow(self, **kwargs):
        workflow = StateGraph(ToolCallReActAgentState)

        # --- Add nodes ---
        react_agent_node = ToolCallReActAgent(self.llm, self.tools)
        workflow.add_node(react_agent_node.name, react_agent_node)

        tool_node = ToolNode(self.tools)
        workflow.add_node(tool_node.name, tool_node)

        # --- Add edges ---
        workflow.add_edge(START, react_agent_node.name)

        # We now add a conditional edge
        def should_continue(state: ToolCallReActAgentState) -> Union[str, list]:
            last_message = state.messages[-1]
            logger.debug(
                f"[should_continue] last_message: [{isinstance(last_message, AIMessage) and last_message.tool_calls}]"
                f"[{last_message.type}] {last_message}"
            )
            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                return tool_node.name
            else:
                return END

        should_continue_destinations = [tool_node.name, END]
        workflow.add_conditional_edges(
            react_agent_node.name,
            should_continue,
            path_map=should_continue_destinations,
        )

        should_return_direct = {t.name for t in self.tools if t.return_direct}

        def route_tool_responses(state: ToolCallReActAgentState):
            for m in reversed(state.messages):
                if not isinstance(m, ToolMessage):
                    break
                if m.name in should_return_direct:
                    return END
            return react_agent_node.name

        if should_return_direct:
            workflow.add_conditional_edges(
                tool_node.name,
                route_tool_responses,
                # path_map=[react_agent_node.name, END],
            )
        else:
            workflow.add_edge(tool_node.name, react_agent_node.name)
        # return workflow

        # def build_and_compile_workflow(
        #     self,
        #     # checkpointer: Checkpointer = None,
        #     # *,
        #     # store: Optional[BaseStore] = None,
        #     # interrupt_before: Optional[Union[All, list[str]]] = None,
        #     # interrupt_after: Optional[Union[All, list[str]]] = None,
        #     # debug: bool = False,
        #     # name: Optional[str] = None,
        #     **kwargs,
        # ):
        # workflow = self.build_workflow(**kwargs)
        compiled_state_graph = workflow.compile(
            # checkpointer=checkpointer,
            # store=store,
            # interrupt_before=interrupt_before,
            # interrupt_after=interrupt_after,
            # debug=debug,
            name=self.name,
        )

        return compiled_state_graph
