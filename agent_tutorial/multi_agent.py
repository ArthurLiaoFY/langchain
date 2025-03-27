from typing import Literal

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command

model = ChatOllama(
    model="qwen2.5:14b",
    temperature=0,
)


def agent_1(state: MessagesState) -> Command[Literal["agent_2", "agent_3", "__end__"]]:
    # you can pass relevant parts of the state to the LLM (e.g., state["messages"])
    # to determine which agent to call next. a common pattern is to call the model
    # with a structured output (e.g. force it to return an output with a "next_agent" field)
    print("agent_1")
    response = {"next_agent": "agent_2", "content": "..."}
    # route to one of the agents or exit based on the LLM's decision
    # if the LLM returns "__end__", the graph will finish execution
    return Command(
        goto=response["next_agent"],
        update={"messages": [response["content"]]},
    )


def agent_2(state: MessagesState) -> Command[Literal["agent_1", "agent_3", "__end__"]]:
    print("agent_2")
    response = {"next_agent": "agent_3", "content": "..."}
    return Command(
        goto=response["next_agent"],
        update={"messages": [response["content"]]},
    )


def agent_3(state: MessagesState) -> Command[Literal["agent_1", "agent_2", "__end__"]]:
    print("agent_3")
    response = {"next_agent": END, "content": "..."}
    return Command(
        goto=response["next_agent"],
        update={"messages": [response["content"]]},
    )


builder = StateGraph(MessagesState)
builder.add_node(agent_1)
builder.add_node(agent_2)
builder.add_node(agent_3)

builder.add_edge(START, "agent_1")
network = builder.compile()
