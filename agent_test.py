# %%
from agent_framework.core.agent.connect_pg import connect_postgres_agent

connect_postgres_agent().invoke(
    {
        "postgres_connection_info": {
            "host": "localhost",
            "port": 5432,
            "dbname": "chinook",
            "user": "wr80340",
            "password": "1qaz2wsx3edc4rfv",
        },
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
                "postgres_connection_info": {
                    "host": "localhost",
                    "port": 5432,
                    "dbname": "chinook",
                    "user": "wr80340",
                    "password": "1qaz2wsx3edc4rfv",
                },
                "recursion_time": 0,
                "recursion_limit": 4,
            }
        )["database"]
    }
)

# %%
