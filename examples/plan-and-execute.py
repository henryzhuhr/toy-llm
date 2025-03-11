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

planner_prompt = ChatPromptTemplate.from_template(
    """针对给定的目标，制定一个简单的分步计划。
此计划应包括个人任务，如果正确执行，将得出正确答案。不要添加任何多余的步骤。
最后一步的结果应该是最终答案。确保每一步都有所需的所有信息——不要跳过步骤。请使用中文。并且你需要按照指定的格式输出。

用户的输入是这样的：
{messages}
""",
)


planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.""",
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


根据上述的信息，您需要执行以下操作
- 如果你认为你已经有了答案，请回复用户，使用工具 `Response`。
- 如果你需要进一步的步骤，请填写一个新的计划，使用工具 `Plan`。只添加仍需要完成的步骤到计划中，不要将已完成的步骤作为计划的一部分返回。


**说明**:
- 如果需要更多步骤才能实现目标，则返回一个包含剩余步骤的 `Plan`。
- 如果所有必要的步骤都已完成，根据收集到的信息向用户返回一个 `Response`。
- **不要**在新计划中包括任何已经完成的步骤。
- **不要**返回一个空计划；如果没有进一步的步骤需要，你必须返回一个 `Response`。
- 确保您的输出按照 `Act` 模型采用正确的结构化格式.

**记住**:
- 该 `Act` 可以是 `Plan` 或 `Response`。
- 一个`Plan`包含仍需完成的步骤列表。他是一个工具
- 一个 `Response` 包含对用户的最终答案。
"""
)


replanner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
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
        default_factory=[],
        # description="规划步骤。要遵循的不同步骤，应该按排序顺序",
        description="different steps to follow, should be in sorted order",
    )


class Response(BaseModel):
    """Response to user."""

    response: str


class Act(BaseModel):
    """Action to perform."""

    # description="Action to perform. If you want to respond to user, use Response. "
    # "If you need to further use tools to get the answer, use Plan."
    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
        #         description=""""要执行的行动。
        # - 如果您已经有确定的答案，请使用回复 (`Response`) 告诉用户答案。
        # - 如果您不确定答案是否是正确的，需要进一步使用工具来获取答案，请使用规划步骤 (`Plan`)。"""
    )


class PlannerNode:
    def __init__(self, llm: ChatOllama):
        self.llm = llm
        self.planner_prompt = planner_prompt
        self.planner = planner_prompt | self.llm.with_structured_output(Plan)

    def __call__(self, state: PlanExecute):
        logger.info(f"🧠 Planning with state: {state}")
        inputs = self.planner_prompt.format_prompt(
            messages=[HumanMessage(state["input"])]
        ).to_messages()
        logger.error(f"inputs: {type(inputs)}")
        logger.error(f"inputs: {inputs}")
        structured_llm = self.llm.with_structured_output(Plan, include_raw=True)

        plan: Plan = structured_llm.invoke(inputs)
        logger.error(f"plan: {type(plan)}")
        logger.error(f"plan: {plan}")
        exit()
        plan: Plan = self.planner.invoke({"messages": [HumanMessage(state["input"])]})
        return {"plan": plan.steps}


class ExecutorNode:
    def __init__(self, graph: CompiledGraph):
        self.graph = graph

    def __call__(self, state: PlanExecute):
        logger.info(f"🚗 Executor with state: {state}")
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

    def __call__(self, state: PlanExecute):
        logger.info(f"🧠 Replanning with state: {state}")
        # output: Act = self.replanner.invoke(
        #     {
        #         "input": state["input"],
        #         "plan": state["plan"],
        #         "past_steps": state["past_steps"],
        #     }
        # )
        # logger.error(f"output: {output}")

        prompt = replanner_prompt.format_prompt(
            input=state["input"],
            plan=state["plan"],
            past_steps=state["past_steps"],
        ).to_messages()
        # logger.error(f"[prompt]: {prompt}")

        model_with_structure = self.llm.with_structured_output(Act, include_raw=True)
        output: Act = model_with_structure.invoke(prompt)

        logger.error(f"[output]: {output}")

        exit()

        # if isinstance(output.action, Response):
        #     return {"response": output.action.response}
        # else:
        #     return {"plan": output.action.steps}


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
    workflow.add_node("planner", plan_node)

    # Add the execution step
    execute_node = ExecutorNode(graph)
    workflow.add_node("agent", execute_node)

    # Add a replan node
    replan_step = ReplannerNode(llm)
    workflow.add_node("replan", replan_step)

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
        graph_img = app.get_graph(xray=8).draw_mermaid_png()
        os.makedirs("tmp", exist_ok=True)
        with open("tmp/graph.png", "wb") as f:
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
