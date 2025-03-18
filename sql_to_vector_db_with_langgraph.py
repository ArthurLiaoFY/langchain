# %%
# https://python.langchain.com/docs/tutorials/sql_qa/
import json
import operator
from collections import defaultdict
from typing import Annotated, Dict, List, Literal, TypedDict

from langchain_community.utilities import SQLDatabase
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from langchain.cache import InMemoryCache
from langchain.globals import set_debug, set_llm_cache

set_llm_cache(InMemoryCache())
set_debug(True)

with open("api_keys.json") as f:
    api_keys = json.loads(f.read())
model = "deepseek-r1:14b"
collection = "Chinook_DB_table_summary"
# setup DB
db = SQLDatabase.from_uri(database_uri="sqlite:///Chinook.db")
db._sample_rows_in_table_info = 3


# %% stage 1, inspect each table
class OverallSQLState(TypedDict):
    db: SQLDatabase
    owner: str
    dialect: str
    tables: Dict[str, str | None]


class SingleSQLDistillState(TypedDict):
    table_name: str
    columns: List[str]
    data_description: str


json_parser = JsonOutputParser(pydantic_object=SingleSQLDistillState)
sql_schema_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an SQL expert specializing in analyzing SQLite schema structures, "
            "helping users understand what information this table contains."
            "{format_instructions}",
        ),
        (
            "user",
            "Here is the SQLite database schema information. "
            "Please analyze its table structures and provide a detailed explanation. {schema}.",
        ),
        (
            "assistant",
            "Sure, let's analyze this schema "
            "and determine what kind of information exists in this table.",
        ),
    ]
)
model = ChatOllama(model=model, temperature=0)
chain = sql_schema_prompt | model | json_parser


# %%
def get_sql_dialect(state: OverallSQLState):
    return {"dialect": state["db"].dialect}


def get_sql_tables(state: OverallSQLState):
    return {
        "tables": {
            table: {
                "columns": ", ".join(
                    [
                        col_info[1]
                        for col_info in state["db"]
                        .run(f"PRAGMA table_info({table});", fetch="cursor")
                        .fetchall()
                    ]
                )
            }
            for table in state["db"].get_usable_table_names()
        }
    }


def single_sql_distill(state: SingleSQLDistillState):
    return {
        "data_description": db.run(
            f"SELECT {state['columns']} FROM {state['table_name']} LIMIT 10;"
        )
        .replace("\n", "")
        .replace("\t", "")
        .replace("\t", "")
    }


def sql_merge(state: SingleSQLDistillState):
    return {"tables": {state["table_name"]: state["data_description"]}}


def continue_to_sqls(state: OverallSQLState):
    return [
        Send(
            node="sql_distill",
            arg={
                "table_name": table,
                "columns": ", ".join(
                    [
                        col_info[1]
                        for col_info in state["db"]
                        .run(f"PRAGMA table_info({table});", fetch="cursor")
                        .fetchall()
                    ]
                ),
            },
        )
        for table in db.get_usable_table_names()
    ]


graph = StateGraph(OverallSQLState)
graph.add_node("get_sql_dialect", get_sql_dialect)
graph.add_node("get_sql_tables", get_sql_tables)
graph.add_node("single_sql_distill", single_sql_distill)

# %%
graph.add_edge(START, "get_sql_dialect")
graph.add_edge(START, "get_sql_tables")
app = graph.compile()
# %%
for s in app.stream(
    {
        "owner": "Arthur",
        "db": SQLDatabase.from_uri(database_uri="sqlite:///Chinook.db"),
    }
):
    print(s)
# %%
s
# %%
