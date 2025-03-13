from abc import abstractmethod
from typing import Optional

from pydantic import Field

from ._base import BaseAgent

# class ReActAgent(BaseAgent):
#     name: str
#     description: Optional[str] = None

#     system_prompt: Optional[str] = None
#     next_step_prompt: Optional[str] = None

#     # llm: Optional[LLM] = Field(default_factory=LLM)
#     # memory: Memory = Field(default_factory=Memory)
#     # state: AgentState = AgentState.IDLE

#     max_steps: int = 10
#     current_step: int = 0

#     @abstractmethod
#     async def think(self) -> bool:
#         """Process current state and decide next action"""

#     @abstractmethod
#     async def act(self) -> str:
#         """Execute decided actions"""

#     async def step(self) -> str:
#         """Execute a single step: think and act."""
#         should_act = await self.think()
#         if not should_act:
#             return "Thinking complete - no action needed"
#         return await self.act()


class ActionAgent(BaseAgent):
    """这是一个数据类，表示代理应采取的操作。它有一个 tool 属性（这是应该调用的工具的名称）和一个 tool_input 属性（该工具的输入）"""

    @abstractmethod
    async def act(self) -> str:
        """Execute decided actions"""


class ReActAgentFLow:
    def __init__(self):
        pass


if __name__ == "__main__":
    pass
