from typing_extensions import Dict, List

from agent_framework.core.states.pg_states import PostgresDatabaseState
from agent_framework.core.tools.pg_utils import (
    close_connection,
    connect_db,
    connection,
    get_related_tables_desc,
    get_relationship_desc,
    get_sample_data,
    get_table_columns,
    get_table_list,
    get_table_primary_key,
    query,
)


def connect_db_node(state: PostgresDatabaseState) -> Dict[connection | None, bool]:
    conn = connect_db.invoke(
        input={"postgres_connection_infos": state["postgres_connection_infos"]}
    )

    return {
        "dialect": "Postgres",
        "database": conn,
        "recursion_time": 0 if conn is not None else state["recursion_time"] + 1,
        "is_connected": True if conn is not None else False,
    }


def get_database_common_infos_node(state: PostgresDatabaseState):

    return {
        "tables": {
            table_name: {
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
