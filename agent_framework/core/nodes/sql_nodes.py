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


def get_db_common_infos(host: str, port: int, dbname: str, user: str, password: str):
    db = connect_db(host=host, port=port, dbname=dbname, user=user, password=password)
    table_list = get_table_list(database=db)
    return {"database": db, "tables": table_list}
