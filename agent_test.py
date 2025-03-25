# %%
import json

from agent_framework.core.agent.pg_agent import (
    connect_postgres_agent,
    extract_table_summary_agent,
)
from agent_framework.core.agent.qdrant_agent import (
    connect_collection_agent,
    connect_qdrant_agent,
)

with open("secrets.json") as f:
    secrets = json.loads(f.read())

with open("config.json") as f:
    config = json.loads(f.read())

# %%


postgres = connect_postgres_agent().invoke(
    {
        "postgres_connection_info": secrets.get("postgres"),
        "recursion_limit": 4,
    }
)

# %%

extract_table_summary_agent().invoke(
    {
        "database": postgres["database"],
        "debug": True,
    }
)

# %%

qdrant = connect_qdrant_agent().invoke(
    {
        "qdrant_connection_info": secrets.get("qdrant"),
        "recursion_limit": 4,
    }
)
# %%
connect_collection_agent().invoke(
    {
        "qdrant_client": qdrant["qdrant_client"],
        "collection": config.get("vector_store").get("collection"),
    }
)

# %%
