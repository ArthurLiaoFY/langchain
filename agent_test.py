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

table_summary = extract_table_summary_agent().invoke(
    {
        "database": postgres["database"],
        "debug": True,
    }
)
# %%

# %%

qdrant = connect_qdrant_agent().invoke(
    {
        "qdrant_connection_info": secrets.get("qdrant"),
        "recursion_limit": 4,
    }
)
# %%
vector_store = connect_collection_agent().invoke(
    {
        "qdrant_client": qdrant["qdrant_client"],
        "collection": config.get("vector_store").get("collection"),
    }
)

# %%
from agent_framework.core.tools.qdrant_utils import insert_vector_store

insert_vector_store.invoke(
    {
        "vector_store": vector_store["vector_store"],
        "docs": [
            table_summary["tables"][table]["table_info_summary"]
            for table in table_summary["tables"].keys()
        ],
    }
)

# %%
from qdrant_client import QdrantClient, models

vector_store["qdrant_client"].scroll(
    collection_name="Chinook_database_table_summary",
    scroll_filter=models.Filter(
        must=[
            models.FieldCondition(
                key="metadata",
                match=models.MatchValue(
                    value=str(
                        {
                            "table": "Track",
                            "columns": "UnitPrice, Bytes, TrackId, AlbumId, MediaTypeId, GenreId, Milliseconds, Name, Composer",
                            "primary_key": "TrackId",
                            "related_tables_desc": "Track with columns: UnitPrice, Bytes, TrackId, AlbumId, MediaTypeId, GenreId, Milliseconds, Name, Composer and Track with columns: UnitPrice, Bytes, TrackId, AlbumId, MediaTypeId, GenreId, Milliseconds, Name, Composer and Track with columns: UnitPrice, Bytes, TrackId, AlbumId, MediaTypeId, GenreId, Milliseconds, Name, Composer. ",
                            "relationship_desc": "foreign key AlbumId references AlbumId in table Album and foreign key GenreId references GenreId in table Genre and foreign key MediaTypeId references MediaTypeId in table MediaType. ",
                        }
                    )
                ),
            ),
        ]
    ),
    limit=1,
    with_payload=True,
    with_vectors=False,
)

# %%
