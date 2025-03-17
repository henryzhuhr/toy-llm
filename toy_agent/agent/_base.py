from abc import ABC

from pydantic import BaseModel, Field

from toy_agent._state import AgentState


class BaseNode(BaseModel, ABC):
    name: str = Field(..., description="Node name")

    @staticmethod
    def __run__(state: AgentState) -> AgentState:
        raise NotImplementedError
