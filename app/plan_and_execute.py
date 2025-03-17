import asyncio
import getpass
import os

from toy_agent.flow.factory import FlowFactory


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("TAVILY_API_KEY")


async def main():
    graph = FlowFactory.PLAN_AND_EXECUTOR.create()().build_workflow()
    graph = FlowFactory.PLAN_AND_EXECUTOR_TEST.create()().build_workflow()
    return

    config = {"recursion_limit": 10, "callbacks": []}
    inputs = {"input": "2024年澳大利亚公开赛男单冠军的家乡是哪里？"}
    async for event in graph.astream(inputs, config=config):
        for k, v in event.items():
            if k != "__end__":
                print(f"🤖 [外部输出] [{k}] {v}")


if __name__ == "__main__":
    asyncio.run(main())
