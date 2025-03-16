from langchain_core.messages import AnyMessage
from loguru import logger

from modules.agent.react_agent.graph import graph


async def main():
    result = await graph.ainvoke(
        {"messages": [("user", "LangChain的创始人是谁？")]},
        {
            "configurable": {
                "system_prompt": "您是一个有用的AI助手。请使用中文回答。如果使用工具搜索到的结果是英文，请翻译成中文"
            }
        },
    )

    for msg in result["messages"]:
        msg: AnyMessage
        logger.info(f"🐣 ({type(msg)}) {msg}")
        print(msg.content)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
