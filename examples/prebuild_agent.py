import os
from datetime import datetime
from tabnanny import verbose
from typing import Annotated, TypedDict, Union

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from loguru import logger


def check_weather(location: str, at_time: datetime | None = None) -> str:
    '''Return the weather forecast for the specified location.'''
    return f"{location} 的天气预报是晴朗的。"


class CustomState(TypedDict):
    today: str
    messages: Annotated[list[BaseMessage], add_messages]
    is_last_step: str
    remaining_steps: str


def main():
    tools = [check_weather]

    base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    model_name = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5:3b")
    llm = ChatOllama(base_url=base_url, model=model_name)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Today is {today}"),
            ("placeholder", "{messages}"),
        ]
    )

    graph = create_react_agent(
        llm, tools=tools, state_schema=CustomState, state_modifier=prompt
    )
    inputs = {"messages": [HumanMessage("深圳天气如何")], "today": "July 16, 2004"}

    for s in graph.stream(inputs, stream_mode="values"):
        message: Union[AIMessage, HumanMessage, ToolMessage]
        message = s["messages"][-1]
        print(f"🤖 [{message.type}] {message}")
        print()


if __name__ == "__main__":
    main()
