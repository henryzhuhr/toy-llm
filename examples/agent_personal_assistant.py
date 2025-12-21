"""
Build a personal assistant with subagents
https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant

Split markdown:
https://docs.langchain.com/oss/python/integrations/splitters/markdown_header_metadata_splitter
"""

import os
from typing import List

from bs4.filter import SoupStrainer
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from pydantic import BaseModel, Field

from src.log.interface import Logger

# ============================================================================
# Define low-level API tools (stubbed)
# ============================================================================


@tool
def create_calendar_event(
    title: str,
    start_time: str,  # ISO format: "2024-01-15T14:00:00"
    end_time: str,  # ISO format: "2024-01-15T15:00:00"
    attendees: list[str],  # email addresses
    location: str = "",
) -> str:
    """Create a calendar event. Requires exact ISO datetime format."""
    return f"Event created: {title} from {start_time} to {end_time} with {len(attendees)} attendees"


@tool
def send_email(
    to: list[str],  # email addresses
    subject: str,
    body: str,
    cc: list[str] = [],
) -> str:
    """Send an email via email API. Requires properly formatted addresses."""
    # Stub: In practice, this would call SendGrid, Gmail API, etc.
    return f"Email sent to {', '.join(to)} - Subject: {subject}"


@tool
def get_available_time_slots(
    attendees: list[str],
    date: str,  # ISO format: "2024-01-15"
    duration_minutes: int,
) -> list[str]:
    """Check calendar availability for given attendees on a specific date."""
    # Stub: In practice, this would query calendar APIs
    return ["09:00", "14:00", "16:00"]


class PersonalAssistantConfig(BaseModel):
    embeddings_model: str = "qwen3-embedding:0.6b"
    base_url: str = Field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    model_name: str = Field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL_NAME", "qwen3:1.7b")
    )


class PersonalAssistant:
    config: PersonalAssistantConfig
    chat_model: BaseChatModel
    embeddings: OllamaEmbeddings
    vector_store: InMemoryVectorStore

    def __init__(self, config: PersonalAssistantConfig) -> None:
        self.config = config
        # 初始化聊天模型
        self.chat_model = ChatOllama(base_url=config.base_url, model=config.model_name)
        # 初始化嵌入模型
        self.embeddings = OllamaEmbeddings(model=self.config.embeddings_model)
        # 初始化向量存储
        self.vector_store = InMemoryVectorStore(self.embeddings)


class CalendarAgent:
    chat_model: BaseChatModel
    agent_executor: AgentExecutor
    SYSTEM_PROMPT = (
        "You are a calendar scheduling assistant. "
        "Parse natural language scheduling requests (e.g., 'next Tuesday at 2pm') "
        "into proper ISO datetime formats. "
        "Use get_available_time_slots to check availability when needed. "
        "Use create_calendar_event to schedule events. "
        "Always confirm what was scheduled in your final response."
    )

    def __init__(self, chat_model: BaseChatModel) -> None:
        self.chat_model = chat_model
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        "Use the tools to schedule the following request:\n{input}"
                    )
                ),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        tools = [create_calendar_event, get_available_time_slots]
        calendar_agent = create_tool_calling_agent(
            chat_model, tools=tools, prompt=prompt
        )
        self.agent_executor = AgentExecutor(
            agent=calendar_agent, tools=tools, verbose=False
        )

    def run(self, query: str):
        for chunk in self.agent_executor.stream({"input": query}):
            # chunk: dict like {"agent": AIMessage(...)} or {"action": ToolMessage(...)}
            for key, messages in chunk.items():
                messages: List[AnyMessage]
                logger.debug(f"{key}:{type(messages)}: {len(messages)}")
                break
                if isinstance(messages, (AIMessage, HumanMessage, ToolMessage)):
                    logger.info(f"type: {type(messages)} {messages}")
                else:
                    logger.debug(f"type: {type(messages)} {messages}")


class EmailAgent:
    SYSTEM_PROMPT = (
        "You are an email assistant. "
        "Compose professional emails based on natural language requests. "
        "Extract recipient information and craft appropriate subject lines and body text. "
        "Use send_email to send the message. "
        "Always confirm what was sent in your final response."
    )
    chat_model: BaseChatModel
    agent_executor: AgentExecutor

    def __init__(self, chat_model: BaseChatModel) -> None:
        self.chat_model = chat_model
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        "Use the tools to schedule the following request:\n{input}"
                    )
                ),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        tools = [create_calendar_event, get_available_time_slots]
        calendar_agent = create_tool_calling_agent(
            chat_model, tools=tools, prompt=prompt
        )
        self.agent_executor = AgentExecutor(
            agent=calendar_agent, tools=tools, verbose=True
        )

    def run(self, query: str):
        for chunk in self.agent_executor.stream({"input": query}):
            # chunk: dict like {"agent": AIMessage(...)} or {"action": ToolMessage(...)}
            for key, message in chunk.items():
                print(f" - type: {type(message)} {message}")
                if isinstance(message, (AIMessage, HumanMessage, ToolMessage)):
                    # message.pretty_print()
                    print(f" MESSAGE - type: {type(message)} {message}")


def main():
    config = PersonalAssistantConfig()
    assistant = PersonalAssistant(config=config)
    # Example usage:

    calendar_agent = CalendarAgent(assistant.chat_model)
    calendar_agent.run("Schedule a team meeting next Tuesday at 2pm for 1 hour")


if __name__ == "__main__":
    main()
