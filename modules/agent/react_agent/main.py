import datetime
import os
import time
from inspect import signature
from typing import Any, ClassVar, List, Tuple, Union, cast

from langchain import hub
from langchain.agents import AgentExecutor, BaseSingleActionAgent, create_react_agent
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_core.agents import AgentAction, AgentFinish, AgentStep
from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    AsyncCallbackManagerForToolRun,
    BaseCallbackManager,
    CallbackManagerForChainRun,
    CallbackManagerForToolRun,
    Callbacks,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from loguru import logger
from pydantic import BaseModel, Field
from typing_extensions import Annotated

from modules.agent.react_agent._base import BaseAgent
from modules.agent.react_agent.prompts import REACT_PROMPT_TEMPLATE
from modules.tools.baidu_search import BaiduSearchTool

template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""


def render_text_description(tools: list[BaseTool]) -> str:
    """Render the tool name and description in plain text.

    Args:
        tools: The tools to render.

    Returns:
        The rendered text.

    Output will be in the format of:

    .. code-block:: markdown

        search: This tool is used for search
        calculator: This tool is used for math
    """
    descriptions = []
    for tool in tools:
        if hasattr(tool, "func") and tool.func:
            sig = signature(tool.func)
            description = f"{tool.name}{sig} - {tool.description}"
        else:
            description = f"{tool.name} - {tool.description}"

        descriptions.append(description)
    return "\n".join(descriptions)


class CustomReActAgent(BaseSingleActionAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def input_keys(self) -> List[str]:
        return ["input"]

    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        logger.error(f"intermediate_steps: {intermediate_steps}")
        return AgentAction()

    async def aplan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        logger.error("only for async")
        return AgentAction()


def main():
    tools = [BaiduSearchTool(max_results=5)]

    prompt = hub.pull("hwchase17/react")
    logger.info(f"prompt: [{type(prompt)}] {prompt}")

    base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    model_name = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5:3b")

    # Initialize the model with tool binding. Change the model or add more tools here.
    chat_model = ChatOllama(base_url=base_url, model=model_name)

    # agent = create_react_agent(chat_model, tools, prompt)

    system_prompt_template = PromptTemplate.from_template(REACT_PROMPT_TEMPLATE)
    system_prompt = system_prompt_template.partial(
        tools=render_text_description(tools),
        tool_names=[tool.name for tool in tools],
        # tool_names=", ".join([t.name for t in tools]),
    )
    logger.info(f"system_prompt: [] {system_prompt}")
    # logger.info(f"system_prompt: [] {system_prompt.format(input='hi')}")

    chat_model_with_stop = chat_model.bind(stop=["\nObservation"])
    # agent:Union[BaseSingleActionAgent, BaseMultiActionAgent, Runnable] = (
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
        }
        | system_prompt
        | chat_model_with_stop
        | ReActJsonSingleInputOutputParser()
    )
    agent = CustomReActAgent()
    # instantiate AgentExecutor
    agent_executor = AgentExecutor(agent=agent, tools=[], verbose=True)

    agent_executor.invoke({"input": "你是谁"})


if __name__ == "__main__":
    main()
