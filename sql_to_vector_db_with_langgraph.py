# %%
# https://python.langchain.com/docs/tutorials/sql_qa/
import json

import psycopg2
from IPython.display import Image, display
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
from typing_extensions import Annotated, Dict, List, Literal, TypedDict

from langchain.cache import InMemoryCache
from langchain.globals import set_debug, set_llm_cache
from langchain.tools import BaseTool, StructuredTool, tool

set_llm_cache(InMemoryCache())
set_debug(False)

with open("secrets.json") as f:
    config = json.loads(f.read())
model = "deepseek-r1:14b"
collection = "Chinook_DB_table_summary"
database_uri = "sqlite:///Chinook.db"
# %%
# setup DB
db = SQLDatabase.from_uri(database_uri=database_uri)
db._sample_rows_in_table_info = 3
# %%


@tool
def get_pg_table_list(database: SQLDatabase) -> list[str]:
    """Get a list of tables in the database."""
    with database.cursor() as curs:
        curs.execute(
            """
            SELECT table_name 
            FROM information_schema.tables
            WHERE table_schema = 'public'
            """
        )
        return [row[0] for row in curs.fetchall()]


@tool
def get_pg_related_tables(
    database: psycopg2.extensions.connection, table_name: str
) -> list[str]:
    """Find all the table related with table input."""
    with database.cursor() as curs:
        curs.execute(
            f"""
            SELECT 
                ccu.table_name AS referenced_table
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
                ON tc.constraint_name = kcu.constraint_name
            JOIN information_schema.constraint_column_usage AS ccu
                ON ccu.constraint_name = tc.constraint_name
            WHERE 1=1
                AND tc.constraint_type = 'FOREIGN KEY' 
                AND tc.table_name = '{table_name}';
            """
        )
        return [row[0] for row in curs.fetchall() if row[0] != table_name]


# %% stage 1, inspect each table
class OverallSQLState(TypedDict):
    db: SQLDatabase
    owner: str
    dialect: str
    tables: Dict[str, str | None]
    schemas: Dict[str, str | None]


class SingleSQLDistillState(TypedDict):
    table_name: str
    data_sample: str
    data_description: str


sql_schema_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            "Here is the SQLite database schema information. "
            "Please analyze table structures and provide a detailed explanation. {schema}.",
        ),
        (
            "assistant",
            "Sure, let's analyze this schema "
            "and determine what kind of information exists in this table.",
        ),
    ]
)
model = ChatOllama(model=model, temperature=0)


# %%
@tool
def sql_info_distill(data_sample: str) -> str:
    """Distill SQL table information."""
    return model.invoke(
        input=sql_schema_prompt.invoke(input=data_sample),
    ).content


def get_basic_info(state: OverallSQLState):
    """Get basic information about the database."""
    return {
        "dialect": state["db"].dialect,
    }


def get_table_info(state: OverallSQLState):
    """Get information about each table in the database."""
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
                ),
                "schema": db.get_table_info([table])
                .replace("\n", "")
                .replace("\t", ""),
                "sample_data": (
                    db.run(
                        command=f"SELECT * FROM {table} LIMIT 10;",
                        include_columns=True,
                    )
                    .replace("\n", "")
                    .replace("\t", "")
                ),
            }
            for table in state["db"].get_usable_table_names()
        }
    }


# %%
graph = StateGraph(OverallSQLState)
graph.add_node("get_basic_info", get_basic_info)
graph.add_node("get_table_info", get_table_info)

graph.add_edge(START, "get_basic_info")
graph.add_edge(START, "get_table_info")
app = graph.compile()
# %%

# display(Image(app.get_graph().draw_mermaid_png()))
# %%
for s in app.stream(
    {
        "owner": "Arthur",
        "db": SQLDatabase.from_uri(database_uri=database_uri),
    }
):
    print(s)
# %%
