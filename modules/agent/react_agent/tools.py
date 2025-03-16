"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

from typing import Any, Callable, List, Optional, cast

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from pydantic import BaseModel, Field
from typing_extensions import Annotated

from .configuration import Configuration


class TavilyInput(BaseModel):
    """Input for the Tavily tool."""

    # query: str = Field(description="search query to look up")
    query: str = Field(description="搜索查询的关键词或输入，应为中文。")


async def search(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[list[dict[str, Any]]]:
    """Search for general web results.

    This function performs a search using the Tavily search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.
    """
    configuration = Configuration.from_runnable_config(config)
    wrapped = TavilySearchResults(
        description=(
            "一款针对全面、准确和可信结果进行优化的搜索引擎。 "
            "当您需要回答关于时事的问题时很有用。 "
            "输入应为搜索查询，请使用中文进行输入。"
        ),
        args_schema=TavilyInput,
        max_results=configuration.max_search_results,
    )
    result = await wrapped.ainvoke({"query": query})
    return cast(list[dict[str, Any]], result)


TOOLS: List[Callable[..., Any]] = [search]
