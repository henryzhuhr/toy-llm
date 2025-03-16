import os
from typing import List

from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama


class OllamLLM:
    def __init__(self, tools: List[BaseTool] = None):
        tools = tools or []
        base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
        model_name = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5:3b")
        self.llm = ChatOllama(
            base_url=base_url,
            model=model_name,
        ).bind_tools(tools)

    def get_ollam(self):
        return self.config.get("ollam", {})
