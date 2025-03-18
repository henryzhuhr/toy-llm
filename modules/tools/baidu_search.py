"""Tool for the Baidu search API."""

from typing import Dict, List, Optional, Tuple, Type, Union

from baidusearch.baidusearch import search
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from loguru import logger
from pydantic import BaseModel, Field


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
        logger.info(f"🔧 Tool [{self.name}] param: [query]{query}")
        result = search(query, self.max_results)
        return result

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        logger.info(f"🔧 Tool [{self.name}] param: [query] {query}")
        result = search(query, self.max_results)
        mock_results = [
            {
                "title": "中国的国土面积 960万平方千米",
                "content": "中国位于亚洲东部，太平洋西岸。陆地总面积约960万平方千米，海域总面积约473万平方千米。中国陆地边界长度约2.2万千米，大陆海岸线长度约1.8万千米。海域分布着大小岛屿7600个，面积最大的是台湾岛，面积35759平方千米。目前中国有34个省级行政区，包括23个省、5个自治区、4个直辖市、2个特别行政区。北京是中国的首都。",
                "url": "https://www.gov.cn/guoqing/index.htm",
                "score": 0.66777676,
            },
        ]
        # return "百度搜索返回的结果，请根据输出进行分析", mock_results
        return {"results": result}


if __name__ == "__main__":
    import asyncio

    async def main():
        tool = BaiduSearchTool()
        # result = tool.invoke({"query": "中国的国土面积"})
        # print(result)
        result = await tool.ainvoke({"query": "中国的国土面积"})
        print(result)

    asyncio.run(main())
