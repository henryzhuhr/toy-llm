import asyncio
import os
import re
from typing import Any, List, Tuple, Union

from langchain.agents import (
    AgentExecutor,
    AgentOutputParser,
    BaseSingleActionAgent,
    LLMSingleActionAgent,
    Tool,
)
from langchain.chains.llm import LLMChain
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish, OutputParserException
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph.state import CompiledStateGraph
from loguru import logger
from ollama import ResponseError
from pydantic import Field

from modules.prompt import agent_prompt
from modules.tools.baidu_search import BaiduSearchTool


class BaiduSearchAgent(BaseSingleActionAgent):
    """虚拟自定义代理。"""

    tool: BaiduSearchTool = Field(
        default_factory=lambda: BaiduSearchTool(max_results=5)
    )

    def __init__(self):
        super().__init__()

    @property
    def input_keys(self):
        return ["query"]

    def plan(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        raise NotImplementedError

    async def aplan(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        """根据输入决定要做什么。

        Args:
            intermediate_steps: LLM到目前为止采取的步骤以及观察结果
            **kwargs: 用户输入

        Returns:
            指定要使用的工具的行动。
        """
        logger.info(f"🤖 intermediate_steps: {intermediate_steps}")
        logger.info(f"🤖 agent 参数: {kwargs}")

        for action, observation in intermediate_steps:
            logger.info(f"🤖 [action] {action}  [observation] {observation}")

        return AgentAction(
            tool=self.tool.name,
            tool_input={"query": kwargs["query"], "max_results": 10},
            log="",
        )


async def main():

    tools = [BaiduSearchTool(max_results=5)]

    base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    # base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model_name = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5:3b")
    llm = ChatOllama(base_url=base_url, model=model_name)

    try:
        llm = ChatOllama(base_url=base_url, model=model_name)
        llm.invoke([HumanMessage("你好")])
    except ResponseError as e:
        logger.error(f"🤖 ChatOllama 初始化失败: {e}")
        return

    agent = BaiduSearchAgent()

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
    )
    await agent_executor.ainvoke("2023年加拿大有多少人口？")


if __name__ == "__main__":
    asyncio.run(main())
