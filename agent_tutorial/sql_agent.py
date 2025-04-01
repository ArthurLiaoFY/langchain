# %%
import json
import os
from typing import Annotated

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from langchain.agents import AgentType, create_sql_agent, tool
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase

# %%

with open("/Users/wr80340/WorkSpace/langchain/secrets.json") as f:
    secrets = json.loads(f.read())
with open("/Users/wr80340/WorkSpace/langchain/config.json") as f:
    config = json.loads(f.read())
os.environ["LANGSMITH_API_KEY"] = secrets.get("langsmith").get("api_key")
os.environ["TAVILY_API_KEY"] = secrets.get("tavily").get("api_key")
# %%
llm = ChatOllama(
    model="qwen2.5:14b",
    temperature=0,
)
db = SQLDatabase.from_uri(
    database_uri="postgresql://{user}:{password}@{host}:{port}/{dbname}".format(
        **secrets.get("postgres")
    )
)


prefix_template = ChatPromptTemplate(
    messages=[
        (
            "system",
            """
            You are an agent designed to interact with a SQL database.
            Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
            Unless the user specifies a specific number of examples they wish to obtain, do not limit the amount of query results.

            You can order the results by a relevant column to return the most interesting examples in the database
            You have access to tools for interacting with the database.
            Never query for all the columns from a specific table, only ask for the relevant columns given the question.
            Wrap each column name in double quotes to denote them as delimited identifiers.
            Only use the below tools. Only use the information returned by the below tools to construct your final answer.
            You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

            DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

            To start you should ALWAYS look at the tables in the database to see what you can query.
            Do NOT skip this step.
            Then you should query the schema of the most relevant tables.
            """,
        ),
    ]
)
format_instructions_template = """
Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question.
"""


toolkit = SQLDatabaseToolkit(db=db, llm=llm)
sql_agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    prefix=prefix_template,
    format_instructions=format_instructions_template,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    # extra_tools=[return_sql_query],
)

# %%


result = sql_agent.invoke(
    {
        "input": "what is the mean and std of sepal length for 'setosa'",
    }
)

# %%
result

# %%
