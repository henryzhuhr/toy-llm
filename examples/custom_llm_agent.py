import os
import re
from typing import List, Union

from langchain.agents import (
    AgentExecutor,
    AgentOutputParser,
    BaseSingleActionAgent,
    LLMSingleActionAgent,
    Tool,
)
from langchain.chains.llm import LLMChain
from langchain.prompts import BaseChatPromptTemplate, StringPromptTemplate
from langchain.schema import (
    AgentAction,
    AgentFinish,
    HumanMessage,
    OutputParserException,
)
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph.state import CompiledStateGraph
from loguru import logger
from ollama import ResponseError
from pydantic import Field

from modules.prompt import agent_prompt
from modules.tools.baidu_search import BaiduSearchTool

# 设置基本模板
template = """Complete the objective as best you can. You have access to the following tools:

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

These were previous tasks you completed:



Begin!

Question: {input}
{agent_scratchpad}"""


# 设置一个提示模板
class CustomPromptTemplate(BaseChatPromptTemplate):
    # 要使用的模板
    template: str
    # 可用工具的列表
    tools: List[Tool]

    def format_messages(self, **kwargs) -> str:
        # 获取中间步骤（AgentAction，Observation元组）
        # 以特定方式格式化它们
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # 将agent_scratchpad变量设置为该值
        kwargs["agent_scratchpad"] = thoughts
        # 从提供的工具列表创建一个tools变量
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools]
        )
        # 为提供的工具创建一个工具名称列表
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]


class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # 检查代理是否应该结束
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # 返回值通常是一个带有单个`output`键的字典
                # 目前不建议尝试其他任何东西 :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # 解析出动作和动作输入
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # 返回动作和动作输入
        return AgentAction(
            tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output
        )


def main():
    tools = [BaiduSearchTool(max_results=5)]
    prompt = CustomPromptTemplate(
        template=template,
        tools=tools,
        # 这里省略了`agent_scratchpad`、`tools`和`tool_names`变量，因为这些是动态生成的
        # 这里包括了`intermediate_steps`变量，因为这是需要的
        input_variables=["input", "intermediate_steps"],
    )

    output_parser = CustomOutputParser()

    base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    # base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model_name = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5:3b")

    try:
        llm = ChatOllama(base_url=base_url, model=model_name)
        llm.invoke([HumanMessage("你好")])
    except ResponseError as e:
        logger.error(f"🤖 ChatOllama 初始化失败: {e}")
        return

    # LLM链由LLM和提示组成
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    tool_names = [tool.name for tool in tools]
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=tool_names,
    )

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True
    )

    agent_executor.run("Search for Leo DiCaprio's girlfriend on the internet.")


main()
