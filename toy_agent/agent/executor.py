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
from loguru import logger
from pydantic import ConfigDict

from toy_agent._state import PlanAndExecuteAgentState
from toy_agent.agent._base import BaseNode


class Executor(BaseNode):
    name: str = "executor"
    agent_executor: CompiledGraph = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, agent_executor: CompiledGraph):
        super().__init__()
        self.agent_executor = agent_executor

    async def __call__(
        self, state: PlanAndExecuteAgentState, config
    ) -> PlanAndExecuteAgentState:
        logger.debug(f"[{Executor.name}]  state: {state}")
        logger.debug(f"[{Executor.name}] config: {config.keys()}")
        plan: List[str] = state.plan
        if not plan:  # Check if plan is empty
            return PlanAndExecuteAgentState(
                past_steps=[], response="No steps to execute in the plan"
            )

        plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
        task = plan[0]
        #     task_formatted = f"""For the following plan:
        # {plan_str}\n\nYou are tasked with executing step {1}, {task}."""
        task_formatted = f"""对于以下计划:\n{plan_str}\n你的任务是执行 {task}."""

        logger.info(f"[ execute step ] task_formatted: {task_formatted}")

        # agent_response = await agent_executor.ainvoke(
        #     {"messages": [HumanMessage(task_formatted)]}
        # )
        inputs = {"messages": [HumanMessage(task_formatted)]}
        for stream in Executor.agent_executor.stream(inputs, stream_mode="values"):
            message: AnyMessage = stream["messages"][-1]

            MESSAGE_ICON = {
                SystemMessage: "🧪",
                HumanMessage: "🙋",
                AIMessage: "🤖",
                ToolMessage: "🛠️",
            }

            logger.info(
                f"[{Executor.name}] message {MESSAGE_ICON.get(type(message), ' ')} [{message.type}] message: {message}"
            )
            agent_response: AIMessage = message
        # logger.info(f"[{self.name}] agent_response: {agent_response}")

        past_steps = state.past_steps + [(task, agent_response.content)]
        logger.info(f"[{Executor.name}] past_steps: {past_steps}")

        state.past_steps = past_steps
        return state
