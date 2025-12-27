import os

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from loguru import logger
from modules.tools.baidu_search import BaiduSearchTool

from toy_agent import AgentState, Executor, Planner, Replanner
from toy_agent.flow._base import BaseFlow


class PlanAndExecuteWithLGReactFlow(BaseFlow):
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

        agent_executor = create_react_agent(llm, tools)
        executor = Executor(agent_executor)
        workflow.add_node(
            executor.name, lambda state, config: executor.__call__(state, config)
        )

        # def execute_step(state: AgentState):
        #     plan = state.plan
        #     plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
        #     task = plan[0]
        #     task_formatted = f"""For the following plan:
        # {plan_str}\n\nYou are tasked with executing step {1}, {task}."""
        #     agent_response = await agent_executor.ainvoke(
        #         {"messages": [("user", task_formatted)]}
        #     )
        #     return {
        #         "past_steps": [(task, agent_response["messages"][-1].content)],
        #     }

        # workflow.add_node(executor.name, execute_step)

        replanner = Replanner(llm)
        workflow.add_node(replanner.name, replanner)

        # --- add edges ---
        workflow.add_edge(START, planner.name)
        workflow.add_edge(planner.name, executor.name)
        workflow.add_edge(executor.name, replanner.name)

        # --- add conditional edges ---
        def should_end(state: AgentState):
            if state.response:
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
            debug=False,
            name="plan_and_execute_agent",
        )

        try:
            graph = compiled_state_graph.get_graph(xray=True)
            graph_mermaid = graph.draw_mermaid()  # noqa: F841
            # print(graph_mermaid)
            graph_img = graph.draw_mermaid_png()
            os.makedirs(save_dir := "tmp", exist_ok=True)
            with open(f"{save_dir}/{compiled_state_graph.name}.png", "wb") as f:
                f.write(graph_img)
        except Exception as e:
            logger.warning(f"Failed to save graph image: {e}")

        return compiled_state_graph
