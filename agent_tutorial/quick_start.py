# %%
import json
import os

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import HumanMessage
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from langchain import hub

os.chdir("..")

with open("secrets.json") as f:
    secrets = json.loads(f.read())
with open("config.json") as f:
    config = json.loads(f.read())
os.environ["LANGSMITH_API_KEY"] = secrets.get("langsmith").get("api_key")
os.environ["TAVILY_API_KEY"] = secrets.get("tavily").get("api_key")
# %%
model = ChatOllama(
    model="qwen2.5:14b",
    temperature=0,
)
db = SQLDatabase.from_uri(
    database_uri="postgresql://{user}:{password}@{host}:{port}/{dbname}".format(
        **secrets.get("postgres")
    )
)
# %%
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()
# %%
prompt_template = ChatPromptTemplate(
    messages=[
        (
            "system",
            """
            You are an agent designed to interact with a SQL database.
            Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
            Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.

            You can order the results by a relevant column to return the most interesting examples in the database
            Never query for all the columns from a specific table, only ask for the relevant columns given the question.
            You have access to tools for interacting with the database.
            Wrap each column name in double quotes (") to denote them as delimited identifiers.
            Only use the below tools. Only use the information returned by the below tools to construct your final answer.
            You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

            DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

            To start you should ALWAYS look at the tables in the database to see what you can query.
            Do NOT skip this step.
            Then you should query the schema of the most relevant tables.
            """,
        )
    ]
)
system_message = prompt_template.format(dialect=db.dialect, top_k=5)
print(system_message)

# %%

# Create the agent
memory = MemorySaver()
model = ChatOllama(
    model="qwen2.5:14b",
    temperature=0,
)
# search = TavilySearchResults(max_results=2)
# tools = [search]
agent_executor = create_react_agent(
    model=model,
    tools=tools,
    checkpointer=memory,
    prompt=system_message,
)
# %%
# Use the agent
config = {"configurable": {"thread_id": "abc123"}}
for step in agent_executor.stream(
    {"messages": [HumanMessage(content="Which country's customers spent the most?")]},
    config,
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
# %%
# Use the agent
config = {"configurable": {"thread_id": "abc123"}}
for step in agent_executor.stream(
    {"messages": [HumanMessage(content="hi im bob! and i live in sf")]},
    config,
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
# %%
for step in agent_executor.stream(
    {"messages": [HumanMessage(content="whats the weather where I live?")]},
    config,
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
# %%
for step in agent_executor.stream(
    {
        "messages": [
            HumanMessage(
                content="Yes, please search for the current weather conditions using an available source"
            )
        ]
    },
    config,
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
# %%
