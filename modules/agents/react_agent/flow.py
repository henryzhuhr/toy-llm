import datetime
import os
import time
from typing import ClassVar, List, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from loguru import logger
from pydantic import BaseModel, Field
from typing_extensions import Annotated

from modules.agents.react_agent._base import BaseAgent
from modules.tools.baidu_search import BaiduSearchTool

from .prompts import REACT_PROMPT_TEMPLATE


class AgentState(BaseModel):
    messages: Annotated[List[AnyMessage], add_messages] = Field(default_factory=list)

    is_last_step: IsLastStep = Field(default=False)


class LLM(BaseModel):
    SYSTEM_PROMPT: ClassVar[str] = """You are a helpful AI assistant.
System time: {system_time}"""

    name: str = "call_model"  # Node name

    llm: BaseChatModel = None  # 声明 llm 字段

    max_history_messages: int = Field(
        default=10, description="The maximum number of messages to keep in the history."
    )

    def __init__(self, tools: List = None):
        super().__init__()
        tools = tools or []
        base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
        model_name = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5:3b")

        # Initialize the model with tool binding. Change the model or add more tools here.
        self.llm = ChatOllama(base_url=base_url, model=model_name).bind_tools(tools)

    def __call__(self, state: AgentState) -> AgentState:
        """Call the LLM powering our "agent".

        This function prepares the prompt, initializes the model, and processes the response.

        Args:
            state (State): The current state of the conversation.
            config (RunnableConfig): Configuration for the model run.

        Returns:
            dict: A dictionary containing the model's response message.
        """
        # Format the system prompt. Customize this to change the agent's behavior.
        system_message = self.SYSTEM_PROMPT.format(
            system_time=datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
        )

        # Get the model's response
        history_messages = state.messages
        history_messages = history_messages[
            -self.max_history_messages :
        ]  # 历史消息，TODO: 应该让模型对历史对话进行总结
        response = cast(
            AIMessage,
            self.llm.invoke(
                [
                    # SystemMessage(system_message), # 是否加入系统消息
                    *history_messages,
                ]
            ),
        )

        # Append the model's response to the state's messages
        state.messages.append(response)

        # Handle the case when it's the last step and the model still wants to use a tool
        if state.is_last_step and response.tool_calls:
            return {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content="Sorry, I could not find an answer to your question in the specified number of steps.",
                    )
                ]
            }

        # Return the model's response as a list to be added to existing messages
        return state


class PlanningAgent(BaseAgent):
    active_plan_id: str = Field(default_factory=lambda: f"plan-{int(time.time())}")

    llm: LLM = None  # 声明 llm 字段

    name: str = "planning_agent"  # Node name
    tools: List[BaseTool] = Field(default_factory=list)

    def __init__(self, llm: LLM, tools: List[BaseTool] = None):
        super().__init__()
        self.llm = llm
        if tools is not None:
            self.tools = tools
            self.llm.llm = self.llm.llm.bind_tools(self.tools)
        else:
            self.tools = []

    def __call__(self, state: AgentState) -> AgentState:
        """Create an initial plan based on the request using the flow's LLM and PlanningTool."""
        logger.info(f"Creating initial plan with ID: {self.active_plan_id}")

        # Create a system message for plan creation
        system_message = SystemMessage(
            # "You are a planning assistant. Your task is to create a detailed plan with clear steps."
            # "您是一位规划助手。您的任务是制定一个详细的计划，并包含明确的步骤。"
            REACT_PROMPT_TEMPLATE.format(
                tools=self.tools,
                tool_names=[t.name for t in self.tools],
                input=state.messages[-1].content,
            )
        )

        # Create a user message with the request
        # user_message = HumanMessage(
        #     # f"Create a detailed plan to accomplish this task: {state.messages[-1].content}"
        #     f"制定一个详细的计划来完成这项任务: {state.messages[-1].content}。"
        #     f"为了顺利完成任务，你可以使用以下工具：{[t.name for t in self.tools]}。"
        #     f"但是在规划的时候请不要使用工具调用，只需要明确使用什么工具即可"
        # )
        user_message = HumanMessage(
            # "You are a planning assistant. Your task is to create a detailed plan with clear steps."
            # "您是一位规划助手。您的任务是制定一个详细的计划，并包含明确的步骤。"
            REACT_PROMPT_TEMPLATE.format(
                tools=self.tools,
                tool_names=[t.name for t in self.tools],
                input=state.messages[-1].content,
            )
        )

        # state.messages.append(system_message)
        state.messages.append(user_message)

        state = self.llm.__call__(state)
        return state


class PlanExecuteAgentFlow:
    def __init__(self):
        pass

    def __call__(self):
        pass


if __name__ == "__main__":
    pass
