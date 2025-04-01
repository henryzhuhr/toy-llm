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

from toy_agent._state import PlanAndExecuteAgentState
from toy_agent.agent.tool_react import ToolCallReActAgentState
from toy_agent.flow.plan_and_executor import PlanAndExecutorFlow
from toy_agent.tools.baidu_search import BaiduSearchTool


async def main():
    base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    model_name = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5:3b")
    llm = ChatOllama(base_url=base_url, model=model_name)

    tools = [
        # TavilySearchResults(max_results=3),
        BaiduSearchTool(max_results=1),
    ]
    flow = PlanAndExecutorFlow(llm, tools=tools)
    app = flow.build_workflow()

    try:
        os.makedirs(save_dir := "tmp", exist_ok=True)
        graph = app.get_graph(xray=True)
        graph_mermaid = graph.draw_mermaid()  # noqa: F841
        with open(f"{save_dir}/{app.name}.md", "wb") as f:
            f.write(f"{datetime.now()}\n```mermaid\n{graph_mermaid}\n```".encode())
        # print(graph_mermaid)
        # graph_img = graph.draw_mermaid_png()
        # with open(f"{save_dir}/{app.name}.png", "wb") as f:
        #     f.write(graph_img)
    except Exception as e:
        logger.warning(f"Failed to save graph image: {e}")
    # return
    config = {"recursion_limit": 50}
    start_state = PlanAndExecuteAgentState(
        # input="武汉大学的第一任校长是哪里人",
        input="联合国安理会最近一次会议讨论了什么议题",
    )
    async for event in app.astream(start_state, config=config):
        for k, v in event.items():
            logger.info(f"🤖 [asteam:{k}] {v}")
            continue
            messages: List[AnyMessage] = v.get("messages", [])
            last_message: AnyMessage = None
            if messages:
                last_message = messages[-1]
                logger.warning(f"🤖 [asteam:{k}] [{last_message.type}] {last_message}")
            # if k != "__end__":
            #     logger.info(v)


if __name__ == "__main__":
    asyncio.run(main())
