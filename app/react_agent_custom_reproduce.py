import asyncio
import datetime
import os
from typing import Dict, List, Literal, Union, cast

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from loguru import logger

from modules.agent.react_agent.state import InputState, State

# from modules.tools.baidu_search import BaiduSearchTool
print(END)


class LLMNode:
    SYSTEM_PROMPT = """You are a helpful AI assistant.

System time: {system_time}"""

    name: str = "call_model"  # Node name

    def __init__(self, tools: List):
        base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
        model_name = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5:7b")

        # Initialize the model with tool binding. Change the model or add more tools here.
        llm = ChatOllama(base_url=base_url, model=model_name)
        self.llm = llm.bind_tools(tools)

    def __call__(
        self, state: State, config: RunnableConfig
    ) -> Dict[str, List[AIMessage]]:
        """Call the LLM powering our "agent".

        This function prepares the prompt, initializes the model, and processes the response.

        Args:
            state (State): The current state of the conversation.
            config (RunnableConfig): Configuration for the model run.

        Returns:
            dict: A dictionary containing the model's response message.
        """
        # Format the system prompt. Customize this to change the agent's behavior.
        system_message = self.SYSTEM_PROMPT.format(
            system_time=datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
        )

        # Get the model's response
        response = cast(
            AIMessage,
            self.llm.invoke(
                [{"role": "system", "content": system_message}, *state.messages], config
            ),
        )

        # Handle the case when it's the last step and the model still wants to use a tool
        if state.is_last_step and response.tool_calls:
            return {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content="Sorry, I could not find an answer to your question in the specified number of steps.",
                    )
                ]
            }

        # Return the model's response as a list to be added to existing messages
        return {"messages": [response]}


def route_model_output(state: State) -> Literal[END, "tools"]:
    """Determine the next node based on the model's output.

    This function checks if the model's last message contains tool calls.

    Args:
        state (State): The current state of the conversation.

    Returns:
        str: The name of the next node to call ("__end__" or "tools").
    """
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    # If there is no tool call, then we finish
    if not last_message.tool_calls:
        return END
    # Otherwise we execute the requested actions
    return "tools"


async def main():
    tools = [
        # BaiduSearchTool(max_results=10),
        TavilySearchResults(max_results=10),
    ]

    # Define a new graph
    builder = StateGraph(
        State,
        input=InputState,
        # config_schema=react_agent.Configuration
    )

    # Define the two nodes we will cycle between
    llm_node = LLMNode(tools)
    builder.add_node(llm_node.name, llm_node)
    # builder.add_node("tools", ToolNode(tools))
    builder.add_node("tools", ToolNode(tools))

    # Set the entrypoint as `call_model`
    # This means that this node is the first one called
    builder.add_edge(START, llm_node.name)

    # Add a conditional edge to determine the next step after `call_model`
    builder.add_conditional_edges(
        llm_node.name,
        # After call_model finishes running, the next node(s) are scheduled
        # based on the output from route_model_output
        route_model_output,
    )

    # Add a normal edge from `tools` to `call_model`
    # This creates a cycle: after using tools, we always return to the model
    builder.add_edge("tools", llm_node.name)

    # Compile the builder into an executable graph
    # You can customize this by adding interrupt points for state updates
    graph = builder.compile(
        interrupt_before=[],  # Add node names here to update state before they're called
        interrupt_after=[],  # Add node names here to update state after they're called
    )
    graph.name = "ReAct Agent"  # This customizes the name in LangSmith

    try:
        graph_img = graph.get_graph().draw_mermaid_png()
        # 保存图片
        os.makedirs("tmp", exist_ok=True)
        with open("tmp/graph.png", "wb") as f:
            f.write(graph_img)
    except Exception:
        # This requires some extra dependencies and is optional
        pass

    inputs = {
        "messages": [
            HumanMessage("2024年美国大选结果"),
            # HumanMessage("深圳天气如何"),
        ]
    }
    for stream in graph.stream(inputs, stream_mode="values"):
        message: Union[BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage]
        message = stream["messages"][-1]

        MESSAGE_ICON = {
            SystemMessage: "🧪",
            HumanMessage: "🙋",
            AIMessage: "🤖",
            ToolMessage: "🛠️",
        }

        logger.info(
            f"{MESSAGE_ICON.get(type(message), ' ')} [{message.type}] message: {message}"
        )

        if isinstance(message, AIMessage):
            logger.warning(f"🤖 [response] {message.content}")


if __name__ == "__main__":
    asyncio.run(main())
