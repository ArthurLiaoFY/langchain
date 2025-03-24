from langchain_core.output_parsers import JsonOutputParser
from typing_extensions import Dict, List

from agent_framework.core.base_models.pg_base_models import PostgresInformationDistill
from agent_framework.core.prompts.pg_prompts import pg_table_information_extractor
from agent_framework.core.states.pg_states import PostgresDatabaseState
from agent_framework.core.tools.pg_utils import (
    close_connection,
    connection,
    database_connection,
    get_related_tables_desc,
    get_relationship_desc,
    get_sample_data,
    get_table_columns,
    get_table_list,
    get_table_primary_key,
    query,
)


def connect_database_node(
    state: PostgresDatabaseState,
) -> Dict[connection | None, bool]:
    conn = database_connection.invoke(
        input={"postgres_connection_info": state["postgres_connection_info"]}
    )

    return {
        "dialect": "Postgres",
        "database": conn,
        "recursion_time": 0 if conn is not None else state["recursion_time"] + 1,
        "is_connected": True if conn is not None else False,
    }


def get_database_common_info_node(state: PostgresDatabaseState):
    return {
        "tables": {
            table_name: {
                "table": table_name,
                "columns": get_table_columns.invoke(
                    input={"database": state["database"], "table_name": table_name}
                ),
                "primary_key": get_table_primary_key.invoke(
                    input={"database": state["database"], "table_name": table_name}
                ),
                # "sample_data": get_sample_data.invoke(
                #     input={"database": state["database"], "table_name": table_name}
                # ),
                "related_tables_desc": get_related_tables_desc.invoke(
                    input={"database": state["database"], "table_name": table_name}
                ),
                "relationship_desc": get_relationship_desc.invoke(
                    input={"database": state["database"], "table_name": table_name}
                ),
            }
            for table_name, in zip(
                get_table_list.invoke(input={"database": state["database"]})
            )
        },
    }


def summarize_table_information(state: PostgresDatabaseState):
    return state["llm_model"].invoke(
        {
            "input": pg_table_information_extractor.invoke(
                {
                    "dialect": state["dialect"],
                    "format_instructions": JsonOutputParser(
                        pydantic_object=PostgresInformationDistill
                    ),
                    # ---------------------------
                    "table": state["table"],
                    "columns": state["columns"],
                    "primary_key": state["primary_key"],
                    "relationship_desc": state["relationship_desc"],
                    "related_tables_desc": state["related_tables_desc"],
                    # ---------------------------
                    "question": state["question"],
                }
            )
        }
    )


def question_answering_node(state: PostgresDatabaseState):
    state["llm_model"].invoke(
        {
            "input": pg_table_information_extractor.invoke(
                {
                    "dialect": state["dialect"],
                    "format_instructions": JsonOutputParser(
                        pydantic_object=PostgresInformationDistill
                    ),
                    # ---------------------------
                    "table": state["table"],
                    "columns": state["columns"],
                    "primary_key": state["primary_key"],
                    "relationship_desc": state["relationship_desc"],
                    "related_tables_desc": state["related_tables_desc"],
                    # ---------------------------
                    "question": state["question"],
                }
            )
        }
    )
