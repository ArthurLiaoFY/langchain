from agent_framework.core.agent.pg_agent import (
    connect_postgres_agent,
    get_postgres_table_info_agent,
)
from agent_framework.core.agent.qdrant_agent import (
    collection_checking_agent,
    connect_qdrant_agent,
)
from agent_framework.core.model import llm_model
from agent_framework.core.prompts.pg_prompts import pg_table_information_extractor
from agent_framework.core.states.pg_to_qdrant_states import Postgres2QdrantState
from agent_framework.core.tools.doc_utils import str_to_doc


def get_table_info_node(state: Postgres2QdrantState):
    postgres = connect_postgres_agent().invoke(
        {
            "postgres_connection_info": state["postgres_connection_info"],
            "recursion_limit": state["recursion_limit"],
        }
    )
    table_summary = get_postgres_table_info_agent().invoke(
        {
            "database": postgres["database"],
        }
    )
    return {
        "database": postgres["database"],
        "tables": table_summary["tables"],
    }


def get_vector_store_info_node(state: Postgres2QdrantState):
    qdrant = connect_qdrant_agent().invoke(
        {
            "qdrant_connection_info": state["qdrant_connection_info"],
            "recursion_limit": state["recursion_limit"],
        }
    )

    vector_store = collection_checking_agent().invoke(
        {
            "qdrant_client": qdrant["qdrant_client"],
            "collection": state["collection"],
        }
    )
    return {
        "qdrant_client": qdrant["qdrant_client"],
        "vector_store": vector_store["vector_store"],
    }


def extract_table_summary_node(state: Postgres2QdrantState):
    return {
        "tables": {
            table_name: {
                **table_details,
                "table_info_summary": str_to_doc.invoke(
                    {
                        "content": (
                            "Hello World"
                            if state["debug"]
                            else llm_model.invoke(
                                input=pg_table_information_extractor.invoke(
                                    {
                                        # ---------------------------
                                        "table": table_details["table"],
                                        "columns": ", ".join(table_details["columns"]),
                                        "primary_key": ", ".join(
                                            table_details["primary_key"]
                                        ),
                                        "related_tables_desc": table_details[
                                            "related_tables_desc"
                                        ],
                                        "relationship_desc": table_details[
                                            "relationship_desc"
                                        ],
                                        # ---------------------------
                                        "question": "What information does {table_name} table contains?".format(
                                            table_name=table_details["table"]
                                        ),
                                    }
                                )
                            ).content
                        ),
                        "metadata": {
                            k: ", ".join(detail) if type(detail) == list else detail
                            for k, detail in table_details.items()
                        },
                    }
                ),
            }
            for table_name, table_details in state["tables"].items()
        }
    }


def upsert_to_vector_database_node(state: Postgres2QdrantState):
    pass
