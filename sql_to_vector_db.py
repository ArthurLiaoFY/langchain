# %%
# https://python.langchain.com/docs/tutorials/sql_qa/
import json
from typing import List

import numpy as np
import pandas as pd
from langchain_community.utilities import SQLDatabase
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from langchain.cache import InMemoryCache
from langchain.globals import set_debug, set_llm_cache

set_llm_cache(InMemoryCache())
set_debug(True)

with open("api_keys.json") as f:
    api_keys = json.loads(f.read())


# %% stage 1, inspect each table
class SQLDescription(BaseModel):
    table_name: str = Field(
        description="name of this table?",
    )
    columns: List[str] = Field(
        description="list all the column name in this table.",
    )
    data_description: str = Field(
        description="what kind of information does this table provide? "
        "please answer this problem precisely.",
    )


json_parser = JsonOutputParser(pydantic_object=SQLDescription)
sqlite_schema_prompt = ChatPromptTemplate.from_messages(
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
model = ChatOllama(model="deepseek-r1:14b", temperature=0)
chain = sqlite_schema_prompt | model | json_parser
# %%
db = SQLDatabase.from_uri(database_uri="sqlite:///Chinook.db")
db._sample_rows_in_table_info = 3
# %%
result = {}
# %%
for idx, single_table_info in enumerate(
    db.get_table_info().replace("\n", "").replace("\t", "").split("CREATE")
):
    if single_table_info:
        result[idx] = chain.invoke(
            {
                "format_instructions": json_parser.get_format_instructions(),
                "schema": "CREATE" + single_table_info,
            }
        )
# %% save to vector DB
collection = "Chinook_DB_table_summary"

embeddings = OllamaEmbeddings(model="deepseek-r1:14b")
qdrant_client = QdrantClient(
    url=api_keys.get("qdrant_url"),
    api_key=api_keys.get("qdrant_api_key"),
)
if not qdrant_client.collection_exists(collection):
    qdrant_client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(
            size=5120,
            distance=Distance.COSINE,
        ),
    )

vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=collection,
    embedding=embeddings,
)
# %%

# %%

vector_store.add_documents(
    documents=[
        Document(
            page_content="Here is a SQL table name: "
            + table_desc.get("table_name")
            + ", with columns: "
            + ", ".join(table_desc.get("columns"))
            + ". "
            + table_desc.get("data_description"),
            metadata={"source": "Chinook_DB"},
        )
        for table_desc in result.values()
    ]
)
# %%
result[3]
# %%
