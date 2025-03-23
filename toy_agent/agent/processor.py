from langchain_core.language_models.chat_models import BaseChatModel
from loguru import logger

from toy_agent._state import PlanAndExecuteAgentState
from toy_agent.agent._base import BaseNode


class Processor(BaseNode):
    name: str = "processor"
    llm: BaseChatModel = None
    index: int = 0

    def __init__(self):
        super().__init__()
        self.index = 0

    async def __call__(
        self, state: PlanAndExecuteAgentState, config
    ) -> PlanAndExecuteAgentState:
        logger.debug(f"[{self.name}]  state: {state}")
        logger.debug(f"[{self.name}] config: {config.keys()}")

        # Update the plan and the past steps
        # If the current task is the same as the first task in the plan, then the task is completed

        if state.plan and state.plan[0] == state.current_task:
            state.past_steps.append(state.plan.pop(0))

        logger.debug(f"[{self.name}] state before return: {state}")
        return state
