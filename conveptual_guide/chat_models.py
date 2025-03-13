# %%
# Chat models

from typing import List

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


class TravelPlan(BaseModel):
    destination: str = Field(description="旅遊目的地, 如日本北海道")
    activities: List[str] = Field(description="推薦的活動")
    budget: float = Field(description="預算範圍,單位新台幣")
    accommodation: List[str] = Field(description="住宿選項")


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "使用繁體中文回答以下問題,\n" "{format_instructions}"),
        ("human", "{query}"),
    ]
)
model = ChatOllama(model="deepseek-r1:14b", temperature=0)
json_parser = JsonOutputParser(pydantic_object=TravelPlan)
chain = prompt | model | json_parser

# %%
ai_msg = chain.invoke(
    {
        "format_instructions": json_parser.get_format_instructions(),
        "query": "我喜歡潛水以及在日落時散步, 所以想要安排一個海邊假期",
    }
)

