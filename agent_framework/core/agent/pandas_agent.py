from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langgraph.prebuilt import create_react_agent

from agent_framework.core.formatter.sql_formatter import SQLQueryOutput
from agent_framework.core.model import sql_coder_model
from agent_framework.core.prompts.sql_prompts import sql_agent_prompt
from agent_framework.core.tools.handoff_tools import make_handoff_tool
from langchain.agents import AgentType, create_sql_agent, create_tool_calling_agent
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent import AgentExecutor, RunnableMultiActionAgent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase


class PandasAgentExecutor:
    def __init__(
        self,
        other_agents_names: list[str],
        agent_name: str = "Pandas Agent Executor",
    ):
        self.agent_name = agent_name
        self.other_agents_names = other_agents_names
      