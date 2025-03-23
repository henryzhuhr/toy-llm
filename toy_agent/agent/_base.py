from abc import ABC

from pydantic import BaseModel, Field

# from toy_agent._state import BaseAgentState


class BaseNode(BaseModel, ABC):
    name: str = Field(..., description="Node name")

    # @staticmethod
    # async def __run__(self, state: BaseAgentState) -> BaseAgentState:
    #     raise NotImplementedError
