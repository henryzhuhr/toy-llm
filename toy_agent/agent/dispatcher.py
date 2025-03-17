from typing import List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools.base import BaseTool
from langgraph.graph.graph import CompiledGraph
from loguru import logger
from pydantic import ConfigDict, Field

from toy_agent._state import AgentState
from toy_agent.agent._base import BaseNode

DEFAULT_DISPATCHER_PROMPT = """对于以下计划:
{plan_str}
你的任务是执行第一步, {task}."""  # 你的任务是执行 step {1}, {task}."""


class Dispatcher(BaseNode):
    name: str = "dispatcher"
    dispatcher_prompt_template: str = Field(default=DEFAULT_DISPATCHER_PROMPT)
    llm: BaseChatModel = None
    tools: List[BaseTool] = None

    should_return_direct: set = None

    def __init__(
        self, llm: BaseChatModel, tools: List[BaseTool] = None, name: str = None
    ):
        super().__init__()
        self.name = name or self.name
        self.llm = llm
        if tools:
            self.llm = self.llm.bind_tools(tools)

        # If any of the tools are configured to return_directly after running,
        # our graph needs to check if these were called
        self.should_return_direct = {t.name for t in tools if t.return_direct}

    async def __call__(self, state: AgentState, config: RunnableConfig) -> AgentState:
        logger.debug(f"[{self.name}]  state: {state}")
        logger.debug(f"[{self.name}] config: {config.keys()}")

        plan: List[str] = state.get("plan", [])
        if not plan:  # Check if plan is empty
            return AgentState(past_steps=[], response="No steps to execute in the plan")

        plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))

        task = plan[0]
        #     task_formatted = f"""For the following plan:
        # {plan_str}\n\nYou are tasked with executing step {1}, {task}."""

        dispatcher_prompt = self.dispatcher_prompt_template.format(
            plan_str=plan_str, task=task
        )

        logger.info(f"[{self.name}] prompt: {dispatcher_prompt}")

        input = [HumanMessage(dispatcher_prompt)]

        response: AIMessage = await self.llm.ainvoke(input)

        logger.info(f"[{self.name}] response: {response}")
        # 如果没有 tool_calls 但是，content 有想要的结果，可以正则尝试一下

        needed = self._are_more_steps_needed(state, response)
        logger.info(f"[{self.name}] _are_more_steps_needed: {needed}")

        if needed:
            # 需要测一下，返回没有tool调用的情况
            return AgentState(
                messages=AIMessage(
                    id=response.id,
                    # content="Sorry, need more steps to process this request.",
                    content="抱歉，需要更多步骤来处理此请求。",
                )
            )

        return AgentState(
            messages=[response]
        )  # TODO: 是否更新 state，而不是返回新的 state

    def _are_more_steps_needed(self, state: AgentState, response: AnyMessage) -> bool:
        has_tool_calls = isinstance(response, AIMessage) and response.tool_calls
        all_tools_return_direct = (
            all(
                call["name"] in self.should_return_direct
                for call in response.tool_calls
            )
            if isinstance(response, AIMessage)
            else False
        )
        remaining_steps = state.get("remaining_steps", None)
        is_last_step = state.get("is_last_step", False)
        return (
            (remaining_steps is None and is_last_step and has_tool_calls)
            or (
                remaining_steps is not None
                and remaining_steps < 1
                and all_tools_return_direct
            )
            or (remaining_steps is not None and remaining_steps < 2 and has_tool_calls)
        )
