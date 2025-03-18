from enum import Enum

from toy_agent.flow.plan_and_executor import PlanAndExecutorFlow
from toy_agent.flow.plan_and_executor_with_lg_react import PlanAndExecuteWithLGReactFlow


class FlowFactory(Enum):
    PLAN_AND_EXECUTOR_WITH_LG_REACT = PlanAndExecuteWithLGReactFlow
    PLAN_AND_EXECUTOR = PlanAndExecutorFlow

    def create(self):
        """创建并返回绑定的具体类实例"""
        return self.value
