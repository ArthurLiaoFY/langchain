from pydantic import BaseModel, Field


class PostgresInformationDistill(BaseModel):
    Question: str = Field(
        description="the question statement.",
    )
    SQLQuery: str = Field(
        description="SQL Query to run.",
    )
    SQLResult: str = Field(
        description="Result of the SQLQuery.",
    )
    Answer: str = Field("Final answer here")
