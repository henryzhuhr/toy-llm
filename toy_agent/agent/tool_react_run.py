from typing import List

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.state import CompiledStateGraph
from loguru import logger
from pydantic import ConfigDict

from toy_agent._state import PlanAndExecuteAgentState
from toy_agent.agent._base import BaseNode


class ToolReActExecutor(BaseNode):
    name: str = "ToolReActExecutor"

    graph: CompiledGraph = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, graph: CompiledGraph):
        super().__init__()
        self.graph = graph
        self.name = self.graph.name

    @staticmethod
    def inject_variables(graph: CompiledStateGraph):
        async def wrapper(state: PlanAndExecuteAgentState, config):
            logger.debug(f"[{graph.name}]  state: {state}")
            # logger.debug(f"[{graph.name}] config: {config}")

            plan = state.plan or []
            if len(plan) == 0:
                logger.warning(f"[{graph.name}] No plan found.")
                return state

            plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))

            task = plan[0]
            task_formatted = f"""For the following plan:
        {plan_str}\n\nYou are tasked with executing step {1}, {task}."""

            task_formatted = f"""你的任务是执行：{task}。"""
            agent_response = await graph.ainvoke(
                {"messages": [HumanMessage(task_formatted)]}
            )
            state.past_steps = [(task, agent_response["messages"][-1].content)]
            return state

        return wrapper

    def get_arun(self):
        """
        调用对象：运行 tool_react_agent 的逻辑。
        """
        return self.inject_variables(self.graph)
