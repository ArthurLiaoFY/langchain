# %%
# Chat models

from typing import List

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


class MultiplierOutput(BaseModel):
    a: int = Field(description="the first number")
    b: int = Field(description="the second number ")


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "there will be two number in the following message, please return it.\n"
            "{format_instructions}",
        ),
        ("human", "{query}"),
    ]
)
model = ChatOllama(model="deepseek-r1:14b", temperature=0)
json_parser = JsonOutputParser(pydantic_object=MultiplierOutput)


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


chain = prompt | model | json_parser | multiply
# %%
chain.invoke(
    {
        "format_instructions": json_parser.get_format_instructions(),
        "query": "i have two child, one is 12 years old, one is 15 years old.",
    }
)
