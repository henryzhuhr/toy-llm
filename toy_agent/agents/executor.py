from typing import List

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph.graph import CompiledGraph
from loguru import logger
from pydantic import ConfigDict

from toy_agent._state import AgentState
from toy_agent.agents._base import BaseNode


class Executor(BaseNode):
    name: str = "executor"
    agent_executor: CompiledGraph = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, agent_executor: CompiledGraph):
        super().__init__()
        self.agent_executor = agent_executor

    async def __call__(self, state: AgentState, config) -> AgentState:
        logger.debug(f"[{self.name}]  state: {state}")
        logger.debug(f"[{self.name}] config: {config.keys()}")
        logger.info(f"[ execute step ] State: {state}")
        plan: List[str] = state.get("plan", [])
        if not plan:  # Check if plan is empty
            return AgentState(past_steps=[], response="No steps to execute in the plan")

        plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
        task = plan[0]
        #     task_formatted = f"""For the following plan:
        # {plan_str}\n\nYou are tasked with executing step {1}, {task}."""
        task_formatted = f"""对于以下计划:
    {plan_str}
    你的任务是执行 step {1}, {task}."""

        logger.info(f"[ execute step ] task_formatted: {task_formatted}")

        # agent_response = await agent_executor.ainvoke(
        #     {"messages": [HumanMessage(task_formatted)]}
        # )
        inputs = {"messages": [HumanMessage(task_formatted)]}
        for stream in self.agent_executor.stream(inputs, stream_mode="values"):
            message: AnyMessage = stream["messages"][-1]

            MESSAGE_ICON = {
                SystemMessage: "🧪",
                HumanMessage: "🙋",
                AIMessage: "🤖",
                ToolMessage: "🛠️",
            }

            logger.info(
                f"[ execute step ] {MESSAGE_ICON.get(type(message), ' ')} [{message.type}] message: {message}"
            )
            agent_response: AIMessage = message
        # logger.info(f"[ execute step ] agent_response: {agent_response}")

        past_steps = state.get("past_steps", []) + [(task, agent_response.content)]
        logger.info(f"[ execute step ] past_steps: {past_steps}")

        return AgentState(past_steps=past_steps)
