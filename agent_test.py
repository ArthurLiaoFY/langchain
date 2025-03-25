# %%
import json

with open("secrets.json") as f:
    secrets = json.loads(f.read())

with open("config.json") as f:
    config = json.loads(f.read())

# %%
from agent_framework.core.agent.pg_agent import (
    connect_postgres_agent,
    extract_table_summary_agent,
)

connect_postgres_agent().invoke(
    {
        "postgres_connection_info": secrets.get("postgres"),
        "recursion_limit": 4,
    }
)["database"]

# %%

extract_table_summary_agent().invoke(
    {
        "database": connect_postgres_agent().invoke(
            {
                "postgres_connection_info": secrets.get("postgres"),
                "recursion_limit": 4,
            }
        )["database"]
    }
)

# %%
