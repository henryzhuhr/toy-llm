import asyncio
import getpass
import os

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from loguru import logger

from toy_agent import AgentState, Executor, Planner, Replanner


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("TAVILY_API_KEY")


async def main():
    graph = build_workflow()
    return

    config = {"recursion_limit": 10, "callbacks": []}
    inputs = {"input": "2024年澳大利亚公开赛男单冠军的家乡是哪里？"}
    async for event in graph.astream(inputs, config=config):
        for k, v in event.items():
            if k != "__end__":
                print(v)


def build_workflow():
    workflow = StateGraph(AgentState)
    base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    model_name = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5:1.5b")
    llm = ChatOllama(base_url=base_url, model=model_name)

    tools = [TavilySearchResults(max_results=3)]

    # --- add nodes ---
    planner = Planner(llm)
    workflow.add_node(planner.name, planner)

    agent_executor = create_react_agent(llm, tools)
    executor = Executor(agent_executor)
    # TODO: subgraph cannot be expanded, related issue
    # - https://github.com/langchain-ai/langgraph/issues/2607
    workflow.add_node(executor.name, executor)  # TODO: subgraph cannot be expanded

    replanner = Replanner(llm)
    workflow.add_node(replanner.name, replanner)

    # --- add edges ---
    workflow.add_edge(START, planner.name)
    workflow.add_edge(planner.name, executor.name)
    workflow.add_edge(executor.name, replanner.name)

    # --- add conditional edges ---
    def should_end(state: AgentState):
        # if state.response:
        if "response" in state and state["response"]:
            return END
        else:
            return executor.name

    workflow.add_conditional_edges(
        replanner.name,
        # Next, we pass in the function that will determine which node is called next.
        should_end,
        [executor.name, END],
    )

    compiled_state_graph = workflow.compile(
        debug=True,
        name="plan_and_execute_agent",
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


if __name__ == "__main__":
    asyncio.run(main())
