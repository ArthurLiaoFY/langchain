# %%
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
    tools=[TavilySearchResults(max_results=3)],
    prompt="You are a helpful assistant.",
)
planner = planner_prompt | llm | plan_json_parser
replanner = replanner_prompt | llm | act_json_parser


# %%
async def execute_step(state: PlanExecute):
    plan = state["plan"]
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    task_formatted = f"""
    For the following plan:
    {plan_str}\n\nYou are tasked with executing step {1}, {task}.
    """
    agent_response = await agent_executor.ainvoke(
        {"messages": [("human", task_formatted)]}
    )
    return {
        "past_steps": [(task, agent_response["messages"][-1].content)],
    }
