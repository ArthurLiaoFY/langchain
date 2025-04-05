# %%
# https://python.langchain.com/docs/tutorials/sql_qa/
import json
import operator
from time import sleep
from typing import Annotated, List, Literal, TypedDict

import pandas as pd
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langsmith import Client
from pydantic import BaseModel, Field
from typing_extensions import Annotated

from langchain.cache import InMemoryCache
from langchain.globals import set_debug, set_llm_cache

set_llm_cache(InMemoryCache())
set_debug(True)
with open("/Users/wr80340/WorkSpace/langchain/secrets.json") as f:
    secrets = json.loads(f.read())

# %%
# prompt

client = Client(api_key=secrets.get("langsmith").get("api_key"))
query_prompt = client.pull_prompt(
    prompt_identifier="langchain-ai/sql-query-system-prompt:5d6c20e9",
    include_model=False,
)
sql_map_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Write a concise summary of the following SQL information: \\n\\n"
            "{single_table_info}",
        )
    ]
)
# %%
reduce_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """
            The following is a set of summaries of SQL tables:
            {join_single_table_summary}
            Take these and distill it into a final, consolidated summary
            of the main themes.
            """,
        )
    ]
)

# %%
# state


class OverallState(TypedDict):
    # Notice here we use the operator.add
    # This is because we want combine all the summaries we generate
    # from individual nodes back into one list - this is essentially
    # the "reduce" part
    contents: List[str]
    summaries: Annotated[list, operator.add]
    final_summary: str


# This will be the state of the node that we will "map" all
# documents to in order to generate summaries


class SingleSQLSummary(BaseModel):
    single_table_info: str = Field(description="Summarized information of this table.")


class SQLQueryOutput(BaseModel):
    """Generated SQL query."""

    query: Annotated[str, ..., "Syntactically valid SQL query."]


class State(BaseModel):
    question: str
    query: str
    result: str
    answer: str


# %%

db = SQLDatabase.from_uri(database_uri="sqlite:///Chinook.db")
db._sample_rows_in_table_info = 0

# %%
llm = ChatOllama(model="qwen2.5:14b", temperature=0).with_structured_output(
    schema=SQLQueryOutput,
)
llm4SQL = ChatOllama(model="qwen2.5:14b", temperature=0)


# %%
def generate_single_SQL_summary(single_table_info: str):
    response = llm4SQL.invoke(sql_map_prompt_template.invoke(single_table_info))
    return response.content


# add await


result = {}
delimiter = "CREATE"
for idx, single_table_info in enumerate(db.get_table_info().split(delimiter)):
    print(idx)
    if single_table_info not in ("", "\n"):
        try:
            result[idx] = generate_single_SQL_summary(
                single_table_info=delimiter + single_table_info
            )
        except ValueError:
            print(delimiter + single_table_info)
        sleep(0.5)

# %%
final_sql_query_result = llm.invoke(
    query_prompt.invoke(
        input={
            "dialect": db.dialect,
            "top_k": 10,
            "table_name": db.get_usable_table_names(),
            "table_info": ", ".join([res for res in result.values()]),
            "input": "show 5 row of data in table Employee",
        }
    )
)
# %%

df = pd.DataFrame(
    db.run(
        final_sql_query_result.query, include_columns=True, fetch="cursor"
    ).fetchall()
)
# %%
df
# %%
