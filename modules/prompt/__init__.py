from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder

# AGENT_PROMPT_TEMPLATE = r"""你是一个乐于助人的助手。"""

# agent_prompt = ChatPromptTemplate(
#     messages=
#         SystemMessage(AGENT_PROMPT_TEMPLATE),
#         # MessagesPlaceholder("{messages}"),
#     ,
#     validate_template=True,
# )

# PLACEHOLDER {messages}


class AGENT_PROMPT_TEMPLATE:

    template = '''Answer the following questions as best you can. You have access to the following tools:

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
Thought:{agent_scratchpad}'''

    template_cn = '''尽可能回答以下问题。您可以使用以下工具：

{tools}

使用以下格式：

Question: 您必须回答的输入问题
Thought: 你应该经常考虑该做什么
Action: 采取的行动，应该是以下之一：{tool_names}
Action Input: 动作的输入
Observation: 行动的结果
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: 我现在知道最终答案
Final Answer: 原始输入问题的最终答案

Begin!

Question: {input}
Thought:{agent_scratchpad}'''


agent_prompt = PromptTemplate.from_template(AGENT_PROMPT_TEMPLATE.template)


# 指令错误导致的问题: This is the error: Could not parse LLM output: https://github.com/langchain-ai/langchain/issues/1358#issuecomment-1972379648
