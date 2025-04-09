from abc import ABC, abstractmethod
from typing import Counter, List

from langgraph.graph.state import CompiledStateGraph


class BaseFlow(ABC):
    name: str = None

    @abstractmethod
    def build_workflow(self, **kwargs) -> CompiledStateGraph:
        raise NotImplementedError

    @staticmethod
    def _validate_tools(tool_names: List[str]):
        """Validate if the tool names are unique"""
        tool_name_count = Counter(tool_names)

        duplicate_tool_names = [n for n, c in tool_name_count.items() if c > 1]
        if len(duplicate_tool_names) > 0:
            raise ValueError(
                f"Duplicate tool names found: {duplicate_tool_names} in {tool_names}"
            )
