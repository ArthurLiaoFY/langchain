# %%
# https://python.langchain.com/docs/tutorials/sql_qa/
import operator
from time import sleep
from typing import Annotated, List, Literal, TypedDict

import pandas as pd
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from typing_extensions import Annotated

from langchain.cache import InMemoryCache
from langchain.globals import set_debug, set_llm_cache

set_llm_cache(InMemoryCache())
set_debug(True)


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
            "你是一個SQL專家, 擅長分析SQLite的Schema結構, 幫助使用者理解資料庫的設計與關聯."
            "{format_instructions}",
        ),
        (
            "user",
            "以下是SQLite資料庫的Schema資訊, 請分析其表結構、主鍵、外鍵、索引及資料型態，並提供深入解釋：\n\n{schema}",
        ),
        (
            "assistant",
            "好的,讓我們分析這個Schema中的,表格名稱與結構, 主鍵, 外鍵關係, 索引, 表中呈現資訊",
        ),
    ]
)
model = ChatOllama(model="deepseek-r1:14b", temperature=0)
chain = sqlite_schema_prompt | model | json_parser
# %%
db = SQLDatabase.from_uri(database_uri="sqlite:///Chinook.db")
db._sample_rows_in_table_info = 50
# %%
result = {}
# %%
for idx, single_table_info in enumerate(db.get_table_info().split("*/")):
    result[idx] = chain.invoke(
        {
            "format_instructions": json_parser.get_format_instructions(),
            "schema": single_table_info.replace("/*", ""),
        }
    )
# %%
result
