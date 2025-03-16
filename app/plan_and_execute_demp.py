import datetime
import getpass
import operator
import os
from typing import Annotated, List, Tuple, Union

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# from langfuse.callback import CallbackHandler
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from loguru import logger
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("TAVILY_API_KEY")
# _set_env("LANGFUSE_PUBLIC_KEY")
# _set_env("LANGFUSE_SECRET_KEY")
# _set_env("LANGFUSE_HOST")


tools = [TavilySearchResults(max_results=3)]

# Choose the LLM that will drive the agent
base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
model_name = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5:3b")
llm = ChatOllama(base_url=base_url, model=model_name)
agent_executor = create_react_agent(llm, tools)


class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str


class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )


planner_prompt = ChatPromptTemplate.from_messages(
    [
        #         (
        #             "system",
        #             """For the given objective, come up with a simple step by step plan. \
        # This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
        # The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.""",
        #         ),
        SystemMessage(
            """针对既定目标，制定一个简单的分步计划。 \
此计划应包括个人任务，如果正确执行，将得出正确答案。不要添加任何多余的步骤。 \
最终步骤的结果应该是最终答案。确保每一步都有所需的所有信息 - 不要跳过步骤。"""
        ),
    ]
)


class Response(BaseModel):
    """Response to user."""

    response: str


class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
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


async def execute_step(state: PlanExecute):
    logger.info(f"[ execute step ] State: {state}")
    plan: List[str] = state.get("plan", [])
    if not plan:  # Check if plan is empty
        return PlanExecute(past_steps=[], response="No steps to execute in the plan")

    plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    #     task_formatted = f"""For the following plan:
    # {plan_str}\n\nYou are tasked with executing step {1}, {task}."""
    task_formatted = f"""对于以下计划:
{plan_str}
你的任务是执行 step {1}, {task}."""

    logger.info(f"[ execute step ] task_formatted: {task_formatted}")

    # agent_response = await agent_executor.ainvoke(
    #     {"messages": [HumanMessage(task_formatted)]}
    # )
    inputs = {"messages": [HumanMessage(task_formatted)]}
    for stream in agent_executor.stream(inputs, stream_mode="values"):
        message: AnyMessage = stream["messages"][-1]

        MESSAGE_ICON = {
            SystemMessage: "🧪",
            HumanMessage: "🙋",
            AIMessage: "🤖",
            ToolMessage: "🛠️",
        }

        logger.info(
            f"[ execute step ] {MESSAGE_ICON.get(type(message), ' ')} [{message.type}] message: {message}"
        )
        agent_response: AIMessage = message
    # logger.info(f"[ execute step ] agent_response: {agent_response}")

    past_steps = state.get("past_steps", []) + [(task, agent_response.content)]
    logger.info(f"[ execute step ] past_steps: {past_steps}")

    return PlanExecute(past_steps=past_steps)


async def plan_step(state: PlanExecute):
    system_prompt = f"您是一个有用的助手。请使用中文回答问题。当前的时间是：{datetime.datetime.now()}"

    human_prompt = HumanMessage(
        """针对既定目标，制定一个简单的分步计划。
        此计划应包括个人任务，如果正确执行，将得出正确答案。不要添加任何多余的步骤。
        确保每一步都有所需的所有信息，不要跳过步骤，在该步骤不允许使用任何工具。
        你的输出应该是一个列表。
        """
    )

    logger.info(f"[ plan step ] State: {state}")

    logger.info(f"[ plan step ] planner_prompt: {planner_prompt}")
    # prompt = planner_prompt.partial(messages=HumanMessage(state["input"]))

    messages = [system_prompt, human_prompt, HumanMessage(state["input"])]
    logger.info(f"[ plan step ] prompt: {messages}")

    structured_response = await llm.with_structured_output(
        Plan,
        method="json_schema",
        include_raw=True,
    ).ainvoke(messages)
    logger.info(f"[ plan step ] structured Response: {structured_response}")
    if structured_response.get("parsed"):
        plan: Plan = structured_response["parsed"]
    else:
        # 解析失败
        raw_reponse: AIMessage = structured_response.get("raw", None)
        if raw_reponse:
            pass  # TODO: 使用模型原始的输出自定义解析
        else:
            # 模型没有返回任何内容
            logger.error("[ plan step ] No structured response found.")
            plan: Plan = Plan(steps=[])

    logger.info(f"[ plan step ] Plan: {plan}")
    # return {"plan": plan.steps}
    return PlanExecute(plan=plan.steps)


async def replan_step(state: PlanExecute):
    logger.info(f"[ replan step ] State: {state}")

    replanner_prompt_template = ChatPromptTemplate.from_template(
        """针对既定目标，制定一个简单的分步计划。\
此计划应包括个人任务，如果正确执行，将得出正确答案。不要添加任何多余的步骤。\
最终步骤的结果应该是最终答案。确保每一步都有所需的所有信息 - 不要跳过步骤。

你的目标是这个：
{input}

你最初的计划是这样的：
{plan}

您目前已经完成了以下步骤：
{past_steps}

相应地更新您的计划。如果没有更多步骤需要执行并且您可以返回给用户，那么就那样回复。否则，填写计划。只添加仍需要完成的步骤到计划中。不要将已完成的步骤作为计划的一部分返回。
"""
    )

    replanner_prompt = replanner_prompt_template.invoke(
        dict(
            input=state["input"],
            plan=state.get("plan", []),
            past_steps=state.get("past_steps", []),
        )
    )
    logger.info(
        f"[ replan step ] replanner_prompt({type(replanner_prompt)}) {replanner_prompt}"
    )

    messages = replanner_prompt.to_messages()
    logger.info(f"[ replan step ] human_message({type(messages)}) {messages}")

    structured_response = await llm.with_structured_output(
        Act,
        method="json_schema",
        include_raw=True,
    ).ainvoke(messages)
    logger.info(f"[ plan step ] structured Response: {structured_response}")

    if structured_response.get("parsed"):
        act: Act = structured_response["parsed"]
        ai_response: AIMessage = structured_response.get("raw")
        if ai_response:
            logger.info(f"[ replan step ] 🤖 {ai_response.content}")
    else:
        # 解析失败
        raw_reponse: AIMessage = structured_response.get("raw", None)
        if raw_reponse:
            pass
        else:
            # 模型没有返回任何内容
            logger.error("[ replan step ] No structured response found.")
            act: Act = Act(action=Response(response="模型输出有问题"))

    # Handle proper Act instance
    if isinstance(act.action, Response):
        return PlanExecute(response=act.action.response)
    else:
        return PlanExecute(plan=act.action.steps)


def should_end(state: PlanExecute):
    if "response" in state and state["response"]:
        return END
    else:
        return "agent"


workflow = StateGraph(PlanExecute)

# Add the plan node
workflow.add_node("planner", plan_step)

# Add the execution step
workflow.add_node("agent", execute_step)

# Add a replan node
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

graph_img = app.get_graph(xray=True).draw_mermaid_png()
os.makedirs("tmp", exist_ok=True)
with open("tmp/graph.png", "wb") as f:
    f.write(graph_img)
# langfuse_handler = CallbackHandler()


async def call_agent(input: str):
    print("=== calling agent ===")
    # config = {"recursion_limit": 50, "callbacks": [langfuse_handler]}
    config = {"recursion_limit": 10, "callbacks": []}
    inputs = {"input": input}
    async for event in app.astream(inputs, config=config):
        for k, v in event.items():
            if k != "__end__":
                print(v)


if __name__ == "__main__":
    import asyncio
    # from langchain_core.messages import HumanMessage
    # inputs = {"input": "what is the hometown of the mens 2024 Australia open winner?"}
    # for s in app.stream(inputs,
    #                     config={"callbacks": [langfuse_handler]}):
    #     print(s)

    asyncio.run(call_agent("2024年澳大利亚公开赛男单冠军的家乡是哪里？"))
