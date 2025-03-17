import os
from typing import Literal, Union

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.tool_node import ToolNode
from loguru import logger

from modules.tools.baidu_search import BaiduSearchTool
from toy_agent import AgentState, Executor, Planner, Replanner
from toy_agent.agent.dispatcher import Dispatcher
from toy_agent.flow._base import BaseFlow


class PlanAndExecuteTestFlow(BaseFlow):
    name: str = "plan_and_execute_agent"

    def __init__(self):
        super().__init__()

    def build_workflow(self):
        workflow = StateGraph(AgentState)
        base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
        model_name = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5:1.5b")
        llm = ChatOllama(base_url=base_url, model=model_name)

        tools = [
            TavilySearchResults(max_results=3),
            BaiduSearchTool(max_results=3),
        ]

        # --- add nodes ---
        planner = Planner(llm)
        workflow.add_node(planner.name, planner)

        # react_agent = ReActAgent(llm, tools=tools)
        # workflow.add_node(react_agent.name, react_agent)
        dispatcher = Dispatcher(llm, tools)
        dispather_start = "dispatcher_start"
        dispather_end = "dispatcher_end"
        workflow.add_node(dispather_start, lambda state: state)
        workflow.add_node(dispatcher.name, dispatcher)
        workflow.add_node(dispather_end, lambda state: state)

        logger.info(
            f"dispatcher should_return_direct: {dispatcher.should_return_direct}"
        )

        tool_node = ToolNode(tools)
        workflow.add_node(tool_node.name, tool_node)

        replanner = Replanner(llm)
        workflow.add_node(replanner.name, replanner)

        # --- add edges ---
        workflow.add_edge(START, planner.name)
        workflow.add_edge(planner.name, dispather_start)
        workflow.add_edge(dispather_start, dispatcher.name)

        # We now add a conditional edge
        # Define the function that determines whether to continue or not
        def should_continue(state: AgentState) -> Union[str, list]:
            messages = state.get("messages")
            last_message = messages[-1]
            # If there is no function call, then we finish
            if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
                # response_format: Optional[
                #     Union[
                #         StructuredResponseSchema, tuple[str, StructuredResponseSchema]
                #     ]
                # ]
                response_format = None  # 暂时禁用
                return (
                    dispather_end
                    if response_format is None
                    else "generate_structured_response"
                )
            # Otherwise if there is, we continue
            else:
                return tool_node.name
                # if version == "v1":
                #     return "tools"
                # elif version == "v2":
                #     tool_calls = [
                #         tool_node.inject_tool_args(call, state, store)  # type: ignore[arg-type]
                #         for call in last_message.tool_calls
                #     ]
                #     return [Send("tools", [tool_call]) for tool_call in tool_calls]

        workflow.add_conditional_edges(
            # First, we define the start node. We use `agent`.
            # This means these are the edges taken after the `agent` node is called.
            dispatcher.name,
            # Next, we pass in the function that will determine which node is called next.
            should_continue,
            path_map=[tool_node.name, dispather_end],
        )

        def route_tool_responses(state: AgentState):
            for m in reversed(state.get("messages")):
                if not isinstance(m, ToolMessage):
                    break
                if m.name in dispatcher.should_return_direct:
                    return dispather_end
            return dispatcher.name

        if dispatcher.should_return_direct:
            workflow.add_conditional_edges(tool_node.name, route_tool_responses)
            pass
        else:
            workflow.add_edge(tool_node.name, dispatcher.name)

        def should_replan_end(state: AgentState):
            # if state.response:
            if "response" in state and state["response"]:
                return END
            else:
                return dispatcher.name

        workflow.add_conditional_edges(
            replanner.name,
            # Next, we pass in the function that will determine which node is called next.
            should_replan_end,
            [dispather_start, END],
        )

        # --- compile ---
        compiled_state_graph = workflow.compile(
            debug=True,
            name="plan_and_execute_agent_test",
        )

        try:
            graph = compiled_state_graph.get_graph(xray=True)
            graph_mermaid = graph.draw_mermaid()  # noqa: F841
            graph_img = graph.draw_mermaid_png()
            os.makedirs(save_dir := "tmp", exist_ok=True)
            with open(f"{save_dir}/{compiled_state_graph.name}.png", "wb") as f:
                f.write(graph_img)
        except Exception as e:
            logger.warning(f"Failed to save graph image: {e}")

        return compiled_state_graph

