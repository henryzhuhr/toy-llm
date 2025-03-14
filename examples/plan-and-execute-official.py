import getpass
import operator
import os
from typing import Annotated, List, Literal, Tuple, Union

from langchain import hub
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
# from langfuse.callback import CallbackHandler
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
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

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/structured-chat-agent")
prompt.pretty_print()

# Choose the LLM that will drive the agent
base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
model_name = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5:3b")
llm = ChatOllama(base_url=base_url, model=model_name, temperature=0)
agent_executor = create_react_agent(llm, tools, state_modifier=prompt)

# res = agent_executor.invoke({"messages": [("user", "who is the winnner of the us open")]})

# print(res)


class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str


class Plan(BaseModel):
    """Plan to follow in future"""

    type: Literal["Plan"] = "Plan"
    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
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

"""
"placeholder" is essentially a temporary holder for content that will be replaced when the template is actually used. You could also use "user" instead of "placeholder" if the messages will always be from the user's perspective.
"""

planner = planner_prompt | ChatOllama(
    base_url=base_url, model=model_name, temperature=0
).with_structured_output(
    Plan
)  # function calling because we are using structured output

# res = planner.invoke(
#     {
#         "messages": [
#             ("user", "what is the hometown of the current Australia open winner?")
#         ]
#     }
# )

# print(res)


class Response(BaseModel):
    """Response to user."""

    type: Literal["Response"] = "Response"
    response: str


class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        discriminator="type",
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan.",
    )

    @classmethod
    def from_response(cls, response_text: str):
        return cls(action=Response(type="Response", response=response_text))


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


replanner = replanner_prompt | ChatOllama(
    base_url=base_url, model=model_name, temperature=0
).with_structured_output(Act)


async def execute_step(state: PlanExecute):
    plan = state["plan"]
    if not plan:  # Check if plan is empty
        return {"past_steps": [], "response": "No steps to execute in the plan."}

    plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
    print("=== plan str ===")
    print(plan_str)
    print("=== end of plan str ===")
    task = plan[0]
    task_formatted = f"""For the following plan:
{plan_str}\n\nYou are tasked with executing step {1}, {task}."""
    agent_response = await agent_executor.ainvoke(
        {"messages": [("user", task_formatted)]}
    )
    return {
        "past_steps": [(task, agent_response["messages"][-1].content)],
    }


async def plan_step(state: PlanExecute):
    plan = await planner.ainvoke({"messages": [("user", state["input"])]})
    return {"plan": plan.steps}


async def replan_step(state: PlanExecute):
    try:
        output = await replanner.ainvoke(state)
        print("=== replanner output ===")
        print(output)
        print("=== end of replanner output ===")

        # Handle dict response
        if isinstance(output, dict) and "Response" in output:
            return {"response": output["Response"]}

        # Handle proper Act instance
        if isinstance(output.action, Response):
            return {"response": output.action.response}
        else:
            return {"plan": output.action.steps}
    except Exception as e:
        print(f"Error in replan_step: {e}")
        return {
            "response": str(
                output.get("Response", "An error occurred processing the response.")
            )
        }


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

# langfuse_handler = CallbackHandler()


async def call_agent(input: str):
    print("=== calling agent ===")
    # config = {"recursion_limit": 50, "callbacks": [langfuse_handler]}
    config = {"recursion_limit": 50, "callbacks": []}
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

    asyncio.run(
        call_agent("what is the hometown of the mens 2024 Australia open winner?")
    )
