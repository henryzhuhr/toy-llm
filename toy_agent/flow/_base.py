from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class BaseFlow(ABC, BaseModel):
    name: str = Field(..., title="Name of the flow")

    @abstractmethod
    def build_workflow():
        raise NotImplementedError
