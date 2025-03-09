import asyncio
import os
from typing import List

# from langchain.agents import AgentExecutor, create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from loguru import logger

from modules.prompt import agent_prompt
from modules.tools.baidu_search import BaiduSearchTool


# Define the tools for the agent to use
@tool
def search(query: str):
    """Call to surf the web."""
    # This is a placeholder, but don't tell the LLM that...
    if "sf" in query.lower() or "san francisco" in query.lower():
        return "It's 60 degrees and foggy."
    return "It's 90 degrees and sunny."


@tool
def play_song_on_spotify(song: str):
    """Play a song on Spotify"""
    # Call the spotify API ...
    return f"Successfully played {song} on Spotify!"


@tool
def play_song_on_apple(song: str):
    """Play a song on Apple Music"""
    # Call the apple music API ...
    return f"Successfully played {song} on Apple Music!"


async def main():
    tools = [
        # TavilySearchResults(max_results=1),
        BaiduSearchTool(max_results=5),
    ]
    tools = [play_song_on_apple, play_song_on_spotify]

    base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    model_name = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5:3b")
    llm = ChatOllama(base_url=base_url, model=model_name)
    llm = llm.bind_tools(tools, parallel_tool_calls=False )
    logger.info(f"🤖 LLM 初始化成功. LLM 可用工具")

    prompt = agent_prompt

    # Initialize memory to persist state between graph runs
    checkpointer = MemorySaver()

    app = create_react_agent(llm, tools, checkpointer=checkpointer)

    # agent: CompiledStateGraph = create_react_agent(llm, tools, prompt)
    try:
        graph_img = app.get_graph().draw_mermaid_png()
        # 保存图片
        os.makedirs("tmp", exist_ok=True)
        with open("tmp/graph.png", "wb") as f:
            f.write(graph_img)
    except Exception:
        # This requires some extra dependencies and is optional
        pass

    # agent_executor = AgentExecutor(
    #     agent=agent,
    #     tools=tools,
    #     verbose=True,
    #     # handle_parsing_errors=True,
    # )

    # logger.info("🤖 AgentExecutor 初始化成功")
    # response = await agent_executor.ainvoke({"input": [HumanMessage("中国的国土面积")]})

    # messages: List[BaseMessage] = response.get("messages", [])
    # for message in messages:
    #     logger.info(f"🤖 message: [{message.type}] {message}")

    final_state = await app.ainvoke(
        {"messages": [{"role": "user", "content": "中国的国土面积"}]},
        config={"configurable": {"thread_id": 42}},
    )

    result = final_state["messages"][-1].content
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
