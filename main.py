from langchain_core.messages import HumanMessage

from modules.agents.plan_execute_agent.flow import LLM, AgentState
from modules.agents.react_agent.flow import PlanningAgent
from modules.tools.baidu_search import BaiduSearchTool

if __name__ == "__main__":
    tools = [BaiduSearchTool(max_results=10)]
    llm_with_tools = LLM(tools)
    state = AgentState(messages=[HumanMessage("2025年的美国总统是谁")])
    # state = llm.__call__(state)
    # print(f"🐣 [{type(state)}: {len(state.messages)}] {state.messages}")
    print()
    for msg in state.messages:
        print(f"🐣 [{msg.type}] {msg}")

    llm = LLM()
    planning_agent = PlanningAgent(llm, tools)
    state = planning_agent.__call__(state)
    # print(f"🐣 [{type(state)}: {len(state.messages)}] {state.messages}")
    print()
    for msg in state.messages:
        print(f"\n🐣 [{msg.type}] {msg} \n{msg.content}")
