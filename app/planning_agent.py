import asyncio
import os
from typing import List

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph.state import CompiledStateGraph
from loguru import logger
from ollama import ResponseError

from modules.prompt import agent_prompt
from modules.tools.baidu_search import BaiduSearchTool


async def main():

    tools = [
        BaiduSearchTool(max_results=5),
    ]

    prompt = agent_prompt

    try:
        llm = ChatOllama(
            base_url="http://host.docker.internal:11434",
            # base_url="http://localhost:11434",
            model="qwen2.5:3b",
        )  # 初始化 ChatOllama 模型
        llm.invoke([HumanMessage("你好")])
    except ResponseError as e:
        logger.error(f"🤖 ChatOllama 初始化失败: {e}")
        return

    agent: CompiledStateGraph = create_react_agent(llm, tools, prompt)
    try:
        graph_img = agent.get_graph().draw_mermaid_png()
        # 保存图片
        os.makedirs("tmp", exist_ok=True)
        with open("tmp/graph.png", "wb") as f:
            f.write(graph_img)
    except Exception:
        # This requires some extra dependencies and is optional
        pass
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        # handle_parsing_errors=True,
    )

    logger.info("🤖 AgentExecutor 初始化成功")
    response = await agent_executor.ainvoke(
        {
            "input": [
                HumanMessage("谁是美网的冠军？"),
            ]
        }
    )

    messages: List[BaseMessage] = response.get("messages", [])
    for message in messages:
        logger.info(f"🤖 message: [{message.type}] {message}")


if __name__ == "__main__":
    asyncio.run(main())
