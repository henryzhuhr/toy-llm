import asyncio
import os
import time
from datetime import datetime
from typing import List, Literal, Union

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
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
from toy_agent.agent._base import BaseNode
from toy_agent.flow._base import BaseFlow


class ReActAgentState(BaseModel):
    """The state of the agent."""

    messages: Annotated[List[BaseMessage], add_messages]

    is_last_step: IsLastStep

    remaining_steps: RemainingSteps


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

    async def __call__(
        self, state: ReActAgentState, config: RunnableConfig
    ) -> ReActAgentState:
        logger.debug(f"[{self.name}]  state: {state}")
        logger.debug(f"[{self.name}] config: {config.keys()}")

        last_message = state.messages[-1]
        # _validate_chat_history(messages)
        # response = cast(AIMessage, model_runnable.invoke(state, config))

        if isinstance(last_message, ToolMessage):
            t_msg = last_message
            logger.error(
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
        self, state: ReActAgentState, response: BaseMessage
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


class ReActAgentFLow(BaseFlow):
    name: str = "plan_and_execute_agent"

    llm: BaseChatModel = None
    tools: List[BaseTool] = None
    tool_names: List[str] = None

    def __init__(self, llm: BaseChatModel, tools: List[BaseTool] = None):
        super().__init__()

        self.llm = llm
        self.tools = tools or []
        self.tool_names = [t.name for t in tools]  # TODO: 检查是否有重复 tool 名称

        if tools:
            self.llm = self.llm.bind_tools(tools)

    def build_workflow(self):  # noqa: C901
        workflow = StateGraph(ReActAgentState)

        react_agent_node = ReActAgent(self.llm, self.tools)
        workflow.add_node(react_agent_node.name, react_agent_node)
        workflow.set_entry_point(react_agent_node.name)

        tool_node = ToolNode(self.tools)
        workflow.add_node(tool_node.name, tool_node)

        # We now add a conditional edge
        def should_continue(state: ReActAgentState) -> Union[str, list]:
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

        def route_tool_responses(state: ReActAgentState):
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

        compiled_state_graph = workflow.compile(
            # checkpointer=checkpointer,
            # store=store,
            # interrupt_before=interrupt_before,
            # interrupt_after=interrupt_after,
            # debug=debug,
            # name=name,
        )

        try:
            graph = compiled_state_graph.get_graph(xray=True)
            graph_mermaid = graph.draw_mermaid()  # noqa: F841
            # print(graph_mermaid)
            graph_img = graph.draw_mermaid_png()
            os.makedirs(save_dir := "tmp", exist_ok=True)
            with open(f"{save_dir}/{compiled_state_graph.name}.png", "wb") as f:
                f.write(graph_img)
        except Exception as e:
            logger.warning(f"Failed to save graph image: {e}")

        return compiled_state_graph


async def _test():
    base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    model_name = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5:7b")
    llm = ChatOllama(base_url=base_url, model=model_name)
    flow = ReActAgentFLow(
        llm,
        tools=[
            TavilySearchResults(max_results=3),
            # BaiduSearchTool(max_results=10),
        ],
    )
    app = flow.build_workflow()
    config = {"recursion_limit": 50}
    start_state = ReActAgentState(
        messages=[HumanMessage("2024年澳大利亚公开赛男单冠军是谁")],
        is_last_step=False,
        remaining_steps=25,
    )
    async for event in app.astream(start_state, config=config):
        for k, v in event.items():
            logger.debug(f"[asteam:{k}] {v}")
            if k != "__end__":
                logger.info(v)


if __name__ == "__main__":
    asyncio.run(_test())
