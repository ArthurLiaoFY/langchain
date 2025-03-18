# %%
# https://python.langchain.com/docs/tutorials/sql_qa/
import json
from typing import Annotated, Dict, List, Literal, TypedDict

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

from langchain.cache import InMemoryCache
from langchain.globals import set_debug, set_llm_cache

set_llm_cache(InMemoryCache())
set_debug(False)

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
    data_sample: str
    data_description: str


sql_schema_prompt = ChatPromptTemplate.from_messages(
    [
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


# %%
def sql_info_distill(data_sample: str) -> str:
    return model.invoke(
        input=sql_schema_prompt.invoke(input=data_sample),
    ).content


def get_basic_infos(state: OverallSQLState):
    return {"dialect": state["db"].dialect}


def get_table_infos(state: OverallSQLState):
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
                "desc": sql_info_distill(
                    data_sample=(
                        db.run(
                            command=f"SELECT * FROM {table} LIMIT 10;",
                            include_columns=True,
                        )
                        .replace("\n", "")
                        .replace("\t", "")
                        .replace("\t", "")
                    )
                ),
            }
            for table in state["db"].get_usable_table_names()
        }
    }


# %%
graph = StateGraph(OverallSQLState)
graph.add_node("get_basic_infos", get_basic_infos)
graph.add_node("get_table_infos", get_table_infos)

graph.add_edge(START, "get_basic_infos")
graph.add_edge(START, "get_table_infos")
app = graph.compile()
# %%

display(Image(app.get_graph().draw_mermaid_png()))
# %%
for s in app.stream(
    {
        "owner": "Arthur",
        "db": SQLDatabase.from_uri(database_uri="sqlite:///Chinook.db"),
    }
):
    print(s)
# %%
