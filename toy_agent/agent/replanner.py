from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from loguru import logger

from toy_agent._state import Act, AgentState, Response
from toy_agent.agent._base import BaseNode
from toy_agent.prompt import PROMPTS


class Replanner(BaseNode):
    name: str = "replanner"
    llm: BaseChatModel = None

    def __init__(self, llm: BaseChatModel):
        super().__init__()
        self.llm = llm

    async def __call__(self, state: AgentState, config) -> AgentState:
        logger.debug(f"[{self.name}]  state: {state}")
        logger.debug(f"[{self.name}] config: {config.keys()}")

        human_prompt = HumanMessage(
            PROMPTS.REPLAN_PROMPT_TEMPLATE.format(
                input=state["input"],  # TODO:如果没有输出的情况
                plan=state.get("plan", []),
                past_steps=state.get("past_steps", []),
            )
        )

        messages = [human_prompt, HumanMessage(state["input"])]
        logger.debug(f"[{self.name}] messages: {messages}")

        structured_response = await self.llm.with_structured_output(
            Act,
            method="json_schema",
            include_raw=True,
        ).ainvoke(messages)
        logger.debug(f"[{self.name}] structured response: {structured_response}")

        if structured_response.get("parsed"):
            act: Act = structured_response["parsed"]
            ai_response: AIMessage = structured_response.get("raw")
            if ai_response:
                logger.debug(f"[ replan step ] 🤖 {ai_response.content}")
            # plan: Plan = structured_response["parsed"]
            # logger.info(
            #     f"[{self.name}] 🤖{AIMessage('').type} plan steps: {os.linesep}{
            #         os.linesep.join([s for s in plan.steps])
            #     }"
            # )
        else:
            # 解析失败
            raw_reponse: AIMessage = structured_response.get("raw", None)
            if raw_reponse:
                pass  # TODO: 使用模型原始的输出自定义解析
            else:
                # 模型没有返回任何内容
                logger.error("[ plan step ] No structured response found.")
                act: Act = Act(action=Response(response="模型输出有问题"))

        # Handle proper Act instance
        if isinstance(act.action, Response):
            return AgentState(response=act.action.response)
        else:
            return AgentState(plan=act.action.steps)
