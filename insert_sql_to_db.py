# %%
import json

import psycopg2
from IPython.display import Image, display
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from psycopg2.extensions import connection
from typing_extensions import Annotated, Dict, List, TypedDict

from agent_framework.core.nodes.pg_nodes import connect_db_node
from agent_framework.core.states.pg_states import PostgresDatabaseState, TableState
from agent_framework.core.tools.qdrant_utils import (
    connect_collection_vector_store,
    connect_qdrant_client,
    insert_vector_store,
)
from langchain.cache import InMemoryCache
from langchain.globals import set_debug, set_llm_cache
from langchain.text_splitter import RecursiveCharacterTextSplitter

# %%
set_llm_cache(InMemoryCache())
set_debug(False)
with open("secrets.json") as f:
    secrets = json.loads(f.read())

with open("config.json") as f:
    config = json.loads(f.read())


# %%
def route(state: PostgresDatabaseState):
    if state["is_connected"]:
        return END
    else:
        if state["recursion_time"] < config.get("database", {}).get("recursion_limit"):
            return "connect_db_node"
        else:
            return END


# %%
graph = StateGraph(PostgresDatabaseState)
graph.add_node("connect_db_node", connect_db_node)
# graph.add_node("get_table_list", get_table_list)
# graph.add_node("run_table_infos", RunnableParallel(context=RunnablePassthrough()))


graph.add_edge(START, "connect_db_node")
graph.add_conditional_edges("connect_db_node", route)
# Define edges

app = graph.compile()
# %%
for s in app.stream(
    {
        "postgres_connection_infos": secrets.get("postgres"),
        "recursion_time": 0,
    }
):
    print(s)
# %%
display(Image(app.get_graph().draw_mermaid_png()))
# %%
