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


agent_prompt = PromptTemplate.from_template(AGENT_PROMPT_TEMPLATE.template)


# 指令错误导致的问题: This is the error: Could not parse LLM output: https://github.com/langchain-ai/langchain/issues/1358#issuecomment-1972379648
