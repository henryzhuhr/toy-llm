import asyncio
import os
import uuid
from datetime import datetime
from typing import List, Optional

from langchain_community.tools.tavily_search import TavilySearchResults  # noqa: F401
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from loguru import logger

from toy_agent._state import PlanAndExecuteAgentState
from toy_agent.flow.plan_and_executor import PlanAndExecutorFlow
from toy_agent.tools.baidu_search import BaiduSearchTool


async def main():
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model_name = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5:7b")
    llm = ChatOllama(base_url=base_url, model=model_name)

    tools = [
        TavilySearchResults(max_results=1),
        # BaiduSearchTool(max_results=1),
    ]
    flow = PlanAndExecutorFlow(llm, tools=tools)
    app = flow.build_workflow()

    try:
        os.makedirs(save_dir := "tmp", exist_ok=True)
        graph = app.get_graph(xray=True)
        graph_mermaid = graph.draw_mermaid()  # noqa: F841
        with open(f"{save_dir}/{app.name}.md", "wb") as f:
            f.write(f"{datetime.now()}\n```mermaid\n{graph_mermaid}\n```".encode())
    except Exception as e:
        logger.warning(f"Failed to save graph image: {e}")

    config = RunnableConfig(
        recursion_limit=50,
        configurable={
            "thread_id": uuid.uuid4().__str__(),
        },
    )

    start_state = PlanAndExecuteAgentState(
        input="深圳大学有几个校区，分别在哪",
    )
    async for event in app.astream(
        start_state,
        config=config,
        # subgraphs=True,  # https://langchain-ai.github.io/langgraph/how-tos/subgraphs-manage-state/#define-parent-graph
    ):
        # print(event)
        for k, v in event.items():
            v: dict
            logger.info(f"🤖 [asteam:{k}] {v}")
            messages: List[AnyMessage] = v.get("messages", [])
            last_message: Optional[AnyMessage] = None
            if messages:
                last_message = messages[-1]
                logger.warning(f"🤖 [asteam:{k}] [{last_message.type}] {last_message}")
            # if k != "__end__":
            #     print(f"🤖 \033[01;36m{v}\033[0m")
            if v.get("response"):
                print(f"\n✅ \033[01;35m{v.get('response')}\033[0m\n")
                break


if __name__ == "__main__":
    asyncio.run(main())
