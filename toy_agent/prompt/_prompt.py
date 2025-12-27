from enum import StrEnum

from langchain_core.prompts import ChatPromptTemplate

from ._planner_prompt import plan_agent_prompt
from ._replanner_prompt import replan_prompt_template
from ._system_prompt import default_system_prompt_template_with_time


class PROMPTS(StrEnum):
    # System Prompts
    DEFAULT_SYSTEMT_PROMPT = default_system_prompt_template_with_time

    # agent Prompts
    PLAN_PROMPT = plan_agent_prompt
    REPLAN_PROMPT_TEMPLATE = replan_prompt_template


class PromptFactory:
    @staticmethod
    def get_prompt(prompt_type: PROMPTS) -> ChatPromptTemplate:
        return prompt_type.value
