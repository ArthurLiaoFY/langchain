# %%
from typing import List

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState


class BaseState(AgentState):
    schema: List[int]


model = ChatOllama(
    model="qwen2.5:14b",
    temperature=0,
)
graph = StateGraph(BaseState)
# graph = StateGraph(AgentState)


agent = create_react_agent(
    model=model,
    tools=[],
    state_schema=BaseState,
)

graph.add_node("agent", agent)
graph.add_edge(START, "agent")
graph.add_edge("agent", END)
chain = graph.compile()

a = chain.invoke({"messages": [("human", "Hi")], "schema": [1, 2, 3]})


# %%
a
# %%
