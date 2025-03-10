import datetime
import operator
import os
from math import e
from typing import Annotated, List, Tuple, TypedDict, Union

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent
from loguru import logger
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from modules.tools.baidu_search import BaiduSearchTool

planner_prompt = ChatPromptTemplate.from_messages(
    [
        #         SystemMessage(
        #             """针对既定目标，制定一个简单的分步计划。
        # 此计划应包括个人任务，如果正确执行，将得出正确答案。不要添加任何多余的步骤。
        # 最后一步的结果应该是最终答案。确保每一步都有所需的所有信息——不要跳过步骤。"""
        #         ),
        (
            "system",
            """针对给定的目标，制定一个简单的分步计划。

此计划应包括个人任务，如果正确执行，将得出正确答案。不要添加任何多余的步骤。

最后一步的结果应该是最终答案。确保每一步都有所需的所有信息——不要跳过步骤。请使用中文。""",
        ),
        ("placeholder", "{messages}"),
    ]
)


replanner_prompt = ChatPromptTemplate.from_template(
    """针对给定的目标，制定一个简单的分步计划。
此计划应包括个人任务，如果正确执行，将得出正确答案。不要添加任何多余的步骤。
最后一步的结果应该是最终答案。确保每一步都有所需的所有信息——不要跳过步骤。

你的目标是这个：
{input}

你最初的计划是这样的：
{plan}

您目前已经完成了以下步骤：
{past_steps}

**说明**:
- 如果需要更多步骤才能实现目标，则返回一个包含剩余步骤的**计划**。
- 如果所有必要的步骤都已完成，根据收集到的信息向用户返回一个**响应**。
- **不要**在新计划中包括任何已经完成的步骤。
- Do **not** return an empty plan; if no further steps are needed, you **must** return a **Response**.
- Ensure your output is in the correct structured format as per the `Act` model.

**Remember**:
- The `Act` can be either a `Plan` or a `Response`.
- A `Plan` contains a list of steps that still need to be done.
- A `Response` contains the final answer to the user.

相应地更新您的计划。如果没有更多步骤需要执行并且您可以返回给用户，那么就那样回应。否则，填写计划。、
只添加仍需要完成的步骤到计划中。不要将已完成的步骤作为计划的一部分返回。请使用中文。"""
)


class PlanExecute(TypedDict):
    """定义状态
    现在，让我们先定义这个代理的跟踪状态。
    首先，我们需要跟踪当前计划。让我们用字符串列表来表示它。
    接下来，我们应该跟踪之前执行过的步骤。让我们用元组列表来表示（这些元组将包含步骤和结果）。
    最后，我们需要一些状态来表示最终响应以及原始输入。
    """

    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str


class Plan(BaseModel):
    """Plan to follow in future
    规划步骤
    现在让我们来思考创建规划步骤。这将使用函数调用创建一个计划。
    使用 Pydantic 与 LangChain
    """

    steps: List[str] = Field(
        default_factory=[], description="要遵循的不同步骤，应该按排序顺序"
    )


class Response(BaseModel):
    """Response to user."""

    response: str


class Act(BaseModel):
    """Action to perform."""

    # description="Action to perform. If you want to respond to user, use Response. "
    # "If you need to further use tools to get the answer, use Plan."
    action: Union[Response, Plan] = Field(
        description="要执行的行动. 如果您想回复用户，请使用回复。 "
        "如果您需要进一步使用工具来获取答案，请使用计划。"
    )


class PlannerNode:
    def __init__(self, llm: ChatOllama):
        self.llm = llm
        self.planner = planner_prompt | self.llm.with_structured_output(Plan)

    def run(self, state: PlanExecute):
        logger.info(f"🧠 Planning with state: {state}")
        plan: Plan = self.planner.invoke({"messages": [("user", state["input"])]})
        return {"plan": plan.steps}


class ExecutorNode:
    def __init__(self, graph: CompiledGraph):
        self.graph = graph

    def run(self, state: PlanExecute):
        plan = state["plan"]
        plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
        task = plan[0]

        task_formatted = f"""以下计划：
    {plan_str}\n\n您被分配执行 step {1}, {task}."""
        agent_response = self.graph.invoke({"messages": [("user", task_formatted)]})
        return {
            "past_steps": [(task, agent_response["messages"][-1].content)],
        }


class ReplannerNode:
    def __init__(self, llm: ChatOllama):
        self.llm = llm
        self.replanner = replanner_prompt | self.llm.with_structured_output(Act)

    def run(self, state: PlanExecute):
        logger.info(f"🧠 Replanning with state: {state}")
        output: Act = self.replanner.invoke(state)
        if isinstance(output.action, Response):
            return {"response": output.action.response}
        else:
            return {"plan": output.action.steps}


def should_end(state: PlanExecute):
    if "response" in state and state["response"]:
        return END
    else:
        return "agent"


def main():
    base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    model_name = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5:3b")
    llm = ChatOllama(base_url=base_url, model=model_name, temperature=0)
    # llm.invoke([HumanMessage("你好")])

    tools = [BaiduSearchTool(max_results=10)]
    # tools = [TavilySearchResults(max_results=3)]

    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prompt = f"""你是一个乐于助人的助手。
    如果你不清楚答案，请使用可能的工具帮助你完成任务
    你可以使用如下工具进行查看结果:{[tool.name for tool in tools]}
    当使用工具后，你需要对结果进行分析。如果你不知道如何继续，请告诉我。

    今天的日期是{current_date}，现在的时间是{current_time}，回答的问题的时候需要结合当前的时间。
    """
    graph = create_react_agent(llm, tools, prompt=prompt)

    workflow = StateGraph(PlanExecute)

    # Add the plan node
    plan_node = PlannerNode(llm)
    workflow.add_node("planner", plan_node.run)

    # Add the execution step
    def execute_step(state: PlanExecute):
        plan = state["plan"]
        plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
        task = plan[0]
        task_formatted = f"""For the following plan:
    {plan_str}\n\nYou are tasked with executing step {1}, {task}."""
        agent_response = graph.invoke({"messages": [("user", task_formatted)]})
        return {
            "past_steps": [(task, agent_response["messages"][-1].content)],
        }

    execute_node = ExecutorNode(graph)
    # workflow.add_node("agent", execute_node.run)
    workflow.add_node("agent", execute_step)

    # Add a replan node
    replan_step = ReplannerNode(llm)
    workflow.add_node("replan", replan_step.run)

    workflow.add_edge(START, "planner")

    # From plan we go to agent
    workflow.add_edge("planner", "agent")

    # From agent, we replan
    workflow.add_edge("agent", "replan")

    workflow.add_conditional_edges(
        "replan",
        # Next, we pass in the function that will determine which node is called next.
        should_end,
        ["agent", END],
    )

    # Finally, we compile it!
    # This compiles it into a LangChain Runnable,
    # meaning you can use it as you would any other runnable
    app = workflow.compile()

    try:
        graph_img = app.get_graph(xray=True).draw_mermaid_png()
        os.makedirs("tmp", exist_ok=True)
        with open("tmp/graph-app.png", "wb") as f:
            f.write(graph_img)
    except Exception:
        # This requires some extra dependencies and is optional
        pass

    # inputs = {"messages": [HumanMessage("去年美国大选的结果是什么？")]}
    # for stream in graph.stream(inputs, stream_mode="values"):
    #     message: Union[BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage]
    #     message = stream["messages"][-1]

    #     MESSAGE_ICON = {
    #         SystemMessage: "🧪",
    #         HumanMessage: "🙋",
    #         AIMessage: "🤖",
    #         ToolMessage: "🛠️",
    #     }

    #     logger.info(
    #         f"{MESSAGE_ICON.get(type(message), ' ')} [{message.type}] message: {message}"
    #     )

    config = {"recursion_limit": 50}
    inputs = {"input": "去年美国大选的结果是什么？"}
    for event in app.stream(inputs, config=config):
        for k, v in event.items():
            if k != "__end__":
                print(v)


if __name__ == "__main__":
    main()
