"""Tool for the Baidu search API."""

from typing import Dict, List, Optional, Tuple, Type, Union

from baidusearch.baidusearch import search
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from utils.logger import logger


class BaiduSearchInput(BaseModel):
    """Input for the Baidu Search tool."""

    query: str = Field(description="要查找的搜索查询")


# 自定义工具: https://python.langchain.ac.cn/docs/how_to/custom_tools/#subclass-basetool
class BaiduSearchTool(BaseTool):  # type: ignore[override, override]
    """百度搜索工具

    Args:
    - max_results (int): 最大结果数
    """

    name: str = "百度搜素"  # 必须定义
    description: str = (  # 必须定义
        "针对全面、准确和可信的结果进行了优化的搜索引擎。"
        "当您需要回答有关时事的问题时很有用。 "
        "输入应该是搜索查询。"
    )
    args_schema: Type[BaseModel] = BaiduSearchInput
    return_direct: bool = True

    # 可选参数
    max_results: int = 5  # 最大结果数

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Union[List[Dict[str, str]], str], Dict]:
        logger.warning("use _arun instead of _run")
        logger.info(f"🔧 Tool [{self.name}] param: {query}")
        result = search(query, self.max_results)
        return result

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        result = search(query, self.max_results)
        return result


if __name__ == "__main__":
    import asyncio

    async def main():
        tool = BaiduSearchTool()
        # result = tool.invoke({"query": "中国的国土面积"})
        # print(result)
        result = await tool.ainvoke({"query": "中国的国土面积"})
        print(result)

    asyncio.run(main())
