import os
from datetime import datetime

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from loguru import logger

from toy_agent._state import AgentState, Plan
from toy_agent.agent._base import BaseNode
from toy_agent.prompt import PROMPTS


class Planner(BaseNode):
    name: str = "planner"
    llm: BaseChatModel = None

    def __init__(self, llm: BaseChatModel):
        super().__init__()
        self.llm = llm

    async def __call__(self, state: AgentState, config) -> AgentState:
        logger.debug(f"[{self.name}]  state: {state}")
        logger.debug(f"[{self.name}] config: {config.keys()}")

        system_prompt = SystemMessage(
            PROMPTS.DEFAULT_SYSTEMT_PROMPT.format(time=datetime.now())
        )
        human_prompt = HumanMessage(PROMPTS.PLAN_PROMPT)

        # messages = [system_prompt, human_prompt, HumanMessage(state["input"])]
        messages = [system_prompt, human_prompt, HumanMessage(state.input)]
        logger.debug(f"[{self.name}] messages: {messages}")

        structured_response = await self.llm.with_structured_output(
            Plan,
            method="json_schema",
            include_raw=True,
        ).ainvoke(messages)
        logger.debug(f"[{self.name}] structured response: {structured_response}")

        if structured_response.get("parsed"):
            plan: Plan = structured_response["parsed"]
            logger.info(
                f"[{self.name}] 🤖{AIMessage('').type} plan steps: {os.linesep}{
                    os.linesep.join(list(plan.steps))
                }"
            )
        else:
            # 解析失败
            raw_reponse: AIMessage = structured_response.get("raw", None)
            if raw_reponse:
                pass  # TODO: 使用模型原始的输出自定义解析
            else:
                # 模型没有返回任何内容
                logger.error("[ plan step ] No structured response found.")
                plan: Plan = Plan(steps=[])

        logger.debug(f"[ plan step ] Plan: {plan}")
        state.plan = plan.steps
        return state
