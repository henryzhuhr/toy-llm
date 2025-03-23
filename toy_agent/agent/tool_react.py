from typing import List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools.base import BaseTool
from langgraph.graph.message import add_messages
from langgraph.managed import IsLastStep, RemainingSteps
from loguru import logger
from pydantic import BaseModel, Field
from typing_extensions import Annotated

from toy_agent._state import ToolCallReActAgentState
from toy_agent.agent._base import BaseNode

template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of {tool_names}
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}"""


template_cn = """尽可能回答以下问题。您可以使用以下工具：{tools}。
使用以下格式：
```
Question: 您必须回答的输入问题
Thought: 你应该经常考虑该做什么
Action: 采取的行动，应该是以下之一：{tool_names}
Action Input: 动作的输入
Observation: 行动的结果
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: 我现在知道最终答案
Final Answer: 原始输入问题的最终答案


Begin!

Question: {input}"""


class ToolCallReActAgent(BaseNode):
    name: str = "ToolCallReActAgent"

    llm: BaseChatModel = None
    tools: List[BaseTool] = None

    def __init__(self, llm: BaseChatModel, tools: List[BaseTool] = None):
        super().__init__()
        self.llm = llm
        self.tools = tools or []

        if tools:
            self.llm = self.llm.bind_tools(tools)

    async def __call__(
        self, state: ToolCallReActAgentState, config: RunnableConfig
    ) -> ToolCallReActAgentState:
        logger.debug(f"[{self.name}]  state: {state}")
        logger.debug(f"[{self.name}] config: {config.keys()}")

        messages = state.messages

        if len(messages) == 0:
            state.is_last_step = True
            state.messages.append(
                AIMessage("Please provide the input question you need to answer.")
            )
            return state

        last_message = messages[-1]
        # _validate_chat_history(messages)
        # response = cast(AIMessage, model_runnable.invoke(state, config))

        if isinstance(last_message, ToolMessage):
            t_msg = last_message
            logger.debug(
                f"[{self.name}] last_message: [{t_msg.type}:{t_msg.name}] {t_msg}"
            )
            prompt = (
                f"你使用了“{t_msg.name}”工具，请对工具的返回内容进行总结。工具返回了 {t_msg.content}。"
                f"如果你需要更多的步骤来处理这个请求，请继续使用工具获取更多信息。"
                f"你可以使用的工具有：{[(t.name, t.description) for t in self.tools]}"
            )
            state.messages.append(HumanMessage(prompt))
        # else:
        # prompt = template_cn.format(
        #     tools=[(t.name, t.description) for t in self.tools],
        #     tool_names=[t.name for t in self.tools],
        #     input=last_message.content,
        # )
        # prompt = state.messages[-1]

        input = state.messages
        # input = [state.messages[-1]]
        logger.debug(f"[{self.name}] input: {input}")

        response = await self.llm.ainvoke(input)

        need = self._are_more_steps_needed(state, response)

        if need:
            state.messages.append(
                AIMessage(
                    id=response.id,
                    content="Sorry, need more steps to process this request.",
                )
            )
        state.messages.append(response)
        return state

    def _are_more_steps_needed(
        self, state: ToolCallReActAgentState, response: BaseMessage
    ) -> bool:
        has_tool_calls = isinstance(response, AIMessage) and response.tool_calls
        should_return_direct = {t.name for t in self.tools if t.return_direct}
        all_tools_return_direct = (
            all(call["name"] in should_return_direct for call in response.tool_calls)
            if isinstance(response, AIMessage)
            else False
        )
        remaining_steps = state.remaining_steps
        is_last_step = state.is_last_step

        if is_last_step and has_tool_calls:
            return True
        elif remaining_steps < 1:
            if all_tools_return_direct:
                return True
        elif remaining_steps < 2:
            if has_tool_calls:
                return True
        return False

        return (
            (remaining_steps is None and is_last_step and has_tool_calls)
            or (
                remaining_steps is not None
                and remaining_steps < 1
                and all_tools_return_direct
            )
            or (remaining_steps is not None and remaining_steps < 2 and has_tool_calls)
        )
