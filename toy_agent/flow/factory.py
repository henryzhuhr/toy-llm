from enum import Enum

from toy_agent.flow.plan_and_executor import PlanAndExecuteFlow
from toy_agent.flow.plan_and_executor_test import PlanAndExecuteTestFlow


class FlowFactory(Enum):
    PLAN_AND_EXECUTOR = PlanAndExecuteFlow
    PLAN_AND_EXECUTOR_TEST = PlanAndExecuteTestFlow

    def create(self):
        """创建并返回绑定的具体类实例"""
        return self.value
