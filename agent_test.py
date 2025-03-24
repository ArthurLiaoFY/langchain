# %%
import json

with open("secrets.json") as f:
    secrets = json.loads(f.read())

with open("config.json") as f:
    config = json.loads(f.read())

# %%
from agent_framework.core.agent.connect_pg import connect_postgres_agent

connect_postgres_agent().invoke(
    {
        "postgres_connection_info": secrets.get("postgres"),
        "recursion_time": 0,
        "recursion_limit": 4,
    }
)["database"]

# %%
from agent_framework.core.agent.connect_pg import connect_postgres_agent
from agent_framework.core.agent.extract_table_summary import extract_table_summary_agent

extract_table_summary_agent().invoke(
    {
        "database": connect_postgres_agent().invoke(
            {
                "postgres_connection_info": secrets.get("postgres"),
                "recursion_time": 0,
                "recursion_limit": 4,
            }
        )["database"]
    }
)

# %%
