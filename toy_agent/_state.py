import operator
from typing import Annotated, List, Sequence, Tuple, Union

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.managed import IsLastStep, RemainingSteps
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# class AgentState(BaseModel):
class AgentState(BaseModel):
    # messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)
    # is_last_step: IsLastStep = Field(default=False)
    # remaining_steps: RemainingSteps = Field(
    #     default=None,  # determined by the user, why set default to 25?
    #     description="Number of steps remaining",
    # )

    input: str = Field(description="Input from user")
    # current_task: str = Field(default=None)
    plan: List[str] = Field(default_factory=list)
    # past_steps: Annotated[List[Tuple], operator.add] = Field(default_factory=list)
    past_steps: List[Tuple] = Field(default_factory=list)
    response: str = None


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
