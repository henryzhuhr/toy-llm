from math import pi
from typing import Union

from langchain_core.tools import BaseTool


class CircumferenceTool(BaseTool):
    name = "周长计算器"
    description = "使用此工具当您需要使用圆的半径来计算周长时"

    def _run(self, radius: Union[int, float]):
        return float(radius) * 2.0 * pi

    async def _arun(self, radius: int):
        raise NotImplementedError("此工具不支持异步")
