"""Default prompts used by the agent."""

from typing import List

SYSTEM_PROMPT = """You are a helpful AI assistant.

System time: {system_time}"""


REACT_PROMPT_TEMPLATE = """Answer the following questions as best you can. You have access to the following tools:

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

Question: {input}
Thought:{agent_scratchpad}
"""

REACT_PROMPT_TEMPLATE = """尽可能回答以下问题。您可以使用以下工具：

{tools}

使用以下格式:

Question: 您必须回答的输入问题
Thought: 你应该总是考虑要做什么
Action: 采取的行动，应该是 {tool_names} 中的一个
Action Input: 行动的输入
Observation: 行动的结果
... (Thought/Action/Action Input/Observation 可以重复 N 次)
Thought: 我现在知道最终的答案了
Final Answer: 原始输入问题的最终答案

开始！
Question: {input}
Thought:{agent_scratchpad}
"""
# Thought:{agent_scratchpad}

class PromptFactory:
    """Factory class to generate prompts for the agent."""

    @staticmethod
    def get_prompt(input: str, tools: List[str], agent_scratchpad: str) -> str:
        """Generate a prompt for the agent.

        Args:
            input (str): The input question.
            tools (List[str]): The list of tools available to the agent.
            agent_scratchpad (str): The agent's scratchpad.

        Returns:
            str: The generated prompt.
        """
        tool_names = ", ".join(tools)
        return REACT_PROMPT_TEMPLATE.format(
            input=input, tools=tool_names, agent_scratchpad=agent_scratchpad
        )
