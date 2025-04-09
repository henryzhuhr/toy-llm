from typing import Annotated, List, Tuple, Union

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.managed import IsLastStep, RemainingSteps
from pydantic import BaseModel, Field


class BaseAgentState(BaseModel):
    """Base class for agent state."""

    pass


class ReActAgentState(BaseModel):
    """The state of the agent."""

    messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)

    is_last_step: IsLastStep = Field(default=False)

    remaining_steps: RemainingSteps = Field(default=50)


class PlannerAgentState(BaseAgentState):
    input: str = Field(description="Input from user")
    # current_task: str = Field(default=None)
    plan: List[str] = Field(default_factory=list)
    # past_steps: Annotated[List[Tuple], operator.add] = Field(default_factory=list)
    past_steps: List[Tuple] = Field(default_factory=list)
    response: str = None


class PlanAndExecuteAgentState(ReActAgentState, PlannerAgentState):
    pass


class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )


class Response(BaseModel):
    """Response to user."""

    response: str


class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )
