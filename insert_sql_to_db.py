# %%
import json

import psycopg2
from IPython.display import Image, display
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from psycopg2.extensions import connection
from typing_extensions import Annotated, Dict, List, TypedDict

from agent_framework.core.nodes.pg_nodes import (
    connect_database_node,
    extract_table_info_node,
    get_database_common_info_node,
    upsert_to_vector_database_node,
)
from agent_framework.core.routes.pg_routes import reconnect_db
from agent_framework.core.states.pg_states import DatabaseState, TableState
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
# load model to state

# %%


# %%
graph = StateGraph(DatabaseState)
graph.add_node("connect_db", connect_database_node)
graph.add_node("get_database_common_info", get_database_common_info_node)
graph.add_node("extract_table_info", extract_table_info_node)
graph.add_node("upsert_to_vector_database", upsert_to_vector_database_node)
# graph.add_node("get_table_list", get_table_list)
# graph.add_node("run_table_info", RunnableParallel(context=RunnablePassthrough()))


graph.add_edge(START, "connect_db")
graph.add_conditional_edges("connect_db", reconnect_db)
graph.add_edge("get_database_common_info", "extract_table_info")
graph.add_edge("extract_table_info", "upsert_to_vector_database")
graph.add_edge("upsert_to_vector_database", END)


app = graph.compile()
# %%
for s in app.stream(
    {
        "postgres_connection_info": secrets.get("postgres"),
        "recursion_time": 0,
        "recursion_limit": config.get("database", {}).get("recursion_limit"),
        "question": "What information does Album table contains?",
    }
):
    print(s)
# %%
display(Image(app.get_graph().draw_mermaid_png()))
# %%
