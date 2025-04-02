# %%
import asyncio
import json
import operator
import os
from typing import Annotated, List, Literal, Tuple, Union

import numpy as np
from IPython.display import Image, display
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import ToolMessage, convert_to_messages
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.types import Command
from langgraph_supervisor import create_supervisor
from pydantic import BaseModel, Field
from typing_extensions import Literal, TypedDict

# %%
os.chdir("..")

with open("secrets.json") as f:
    secrets = json.loads(f.read())
with open("config.json") as f:
    config = json.loads(f.read())
os.environ["LANGSMITH_API_KEY"] = secrets.get("langsmith").get("api_key")
os.environ["TAVILY_API_KEY"] = secrets.get("tavily").get("api_key")


# %%

llm = ChatOllama(
    model="qwen2.5:14b",
    temperature=0,
)


# %%
class PlanExecute(TypedDict):
    input: str
    plan_format_instructions: str
    act_format_instructions: str

    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str


class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )


class Response(BaseModel):
    """Response to user."""

    response: str


class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description=(
            "Action to perform. "
            "If you want to respond to user, use Response. "
            "If you need to further use tools to get the answer, use Plan."
        )
    )


plan_json_parser = JsonOutputParser(pydantic_object=Plan)
act_json_parser = JsonOutputParser(pydantic_object=Act)
# %%


planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "For the given objective, come up with a simple step by step plan. "
            "This plan should involve individual tasks, "
            "that if executed correctly will yield the correct answer. "
            "Do not add any superfluous steps. "
            "Wrap strings by double quotes instead of single quotes. "
            "The result of the final step should be the final answer. "
            "Make sure that each step has all the information needed - do not skip steps. "
            "{plan_format_instructions}",
        ),
        ("human", "{messages}"),
    ]
)


replanner_prompt = ChatPromptTemplate.from_template(
    "For the given objective, come up with a simple step by step plan. "
    "This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. "
    "The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps. "
    "Your objective was this: "
    "{input} "
    "Your original plan was this: "
    "{plan} "
    "You have currently done the follow steps: "
    "{past_steps} "
    "Update your plan accordingly. "
    "If no more steps are needed and you can return to the user, then respond with that. "
    "Otherwise, fill out the plan. "
    "Only add steps to the plan that still NEED to be done. "
    "Do not return previously done steps as part of the plan. "
    "{act_format_instructions}",
)

# %%
agent_executor = create_react_agent(
    model=llm,
    tools=[TavilySearchResults(max_results=5)],
    prompt=(
        "You are a web search assistant, "
        "Always search the internet before answering questions."
    ),
)
planner = planner_prompt | llm | plan_json_parser
replanner = replanner_prompt | llm | act_json_parser


# %%
def plan_step(state: PlanExecute):
    plan = planner.invoke(
        {
            "messages": [{"role": "human", "content": state["input"]}],
            "plan_format_instructions": plan_json_parser.get_format_instructions(),
        }
    )
    return {"plan": plan.get("steps", [])}


def replan_step(state: PlanExecute):
    output = replanner.invoke(
        {
            "input": state["input"],
            "plan": state["plan"],
            "past_steps": state["past_steps"],
            "act_format_instructions": act_json_parser.get_format_instructions(),
        }
    )

    if "response" in output.get("action").keys():
        return {"response": output.get("action", {}).get("response", "")}
    else:
        return {"plan": output.get("action", {}).get("steps", [])}


def should_end(state: PlanExecute):
    if "response" in state and state["response"]:
        return END
    else:
        return "agent"


def execute_step(state: PlanExecute):
    plan = state["plan"]
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    task_formatted = f"""
    For the following plan:
    {plan_str}\n\nYou are tasked with executing step {1}, {task}.
    """
    agent_response = agent_executor.invoke(
        {"messages": [{"role": "human", "content": task_formatted}]}
    )
    return {
        "past_steps": [(task, agent_response["messages"][-1].content)],
    }


# %%
from langgraph.graph import START, StateGraph

workflow = StateGraph(PlanExecute)

workflow.add_node("planner", plan_step)
workflow.add_node("replan", replan_step)
workflow.add_node("agent", execute_step)

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
# %%
display(Image(app.get_graph(xray=True).draw_mermaid_png()))


# %%
question = "what is the hometown of the mens 2024 Australia open winner?"

for event in app.stream({"input": question}):
    for k, v in event.items():
        if k != "__end__":
            print("-" * 20 + k + "-" * 20)
            print(v)


# %%
