from typing_extensions import Dict, List

from agent_framework.core.states.pg_states import PostgresDatabaseState
from agent_framework.core.tools.pg_utils import (
    close_connection,
    connect_db,
    connection,
    get_related_tables,
    get_relationship_desc,
    get_sample_data,
    get_table_columns,
    get_table_list,
    query,
)


def connect_db_node(state: PostgresDatabaseState) -> Dict[connection | None, bool]:
    conn = connect_db.invoke(
        input={"postgres_connection_infos": state["postgres_connection_infos"]}
    )

    return {
        "database": conn,
        "recursion_time": 0 if conn is not None else state["recursion_time"] + 1,
        "is_connected": True if conn is not None else False,
    }
