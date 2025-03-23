import asyncio
import os
import time
from datetime import datetime
from typing import List, Literal, Union

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_ollama import ChatOllama
from loguru import logger

from toy_agent.agent.tool_react import ToolCallReActAgentState
from toy_agent.flow.tool_react import ToolCallReActAgentFLow
from toy_agent.tools.baidu_search import BaiduSearchTool


async def main():
    base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    model_name = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5:7b")
    llm = ChatOllama(base_url=base_url, model=model_name)
    flow = ToolCallReActAgentFLow(
        llm,
        tools=[
            # TavilySearchResults(max_results=3),
            BaiduSearchTool(max_results=1),
        ],
    )
    app = flow.build_workflow()

    try:
        graph = app.get_graph(xray=True)
        # graph_mermaid = graph.draw_mermaid()  # noqa: F841
        # print(graph_mermaid)
        graph_img = graph.draw_mermaid_png()
        os.makedirs(save_dir := "tmp", exist_ok=True)
        with open(f"{save_dir}/{app.name}.png", "wb") as f:
            f.write(graph_img)
    except Exception as e:
        logger.warning(f"Failed to save graph image: {e}")

    config = {"recursion_limit": 50}
    start_state = ToolCallReActAgentState(
        messages=[HumanMessage("中国的国土面积")],
        is_last_step=False,
        remaining_steps=25,
    )
    async for event in app.astream(start_state, config=config):
        for k, v in event.items():
            # logger.debug(f"🤖 [asteam:{k}] {v}")
            messages: List[AnyMessage] = v.get("messages", [])
            last_message: AnyMessage = None
            if messages:
                last_message = messages[-1]
                logger.warning(f"🤖 [asteam:{k}] [{last_message.type}] {last_message}")
            # if k != "__end__":
            #     logger.info(v)


if __name__ == "__main__":
    asyncio.run(main())
