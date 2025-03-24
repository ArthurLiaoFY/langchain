from psycopg2.extensions import connection
from typing_extensions import Annotated, Dict, List, TypedDict, Union


class PostgresDatabaseState(TypedDict):
    # ---------------------------
    postgres_connection_infos: Dict[str, Union[int, str]]
    recursion_limit: int
    # ---------------------------
    database: connection
    recursion_time: int
    is_connected: bool
    # ---------------------------
    tables: Dict[str, Dict[str, str]]
    # table_infos: Dict[str, Dict[str, str]]
    # ---------------------------


class TableState(TypedDict):
    table_name: str
    table_columns: List[str]
    related_tables: List[str]
    relationship_desc: str
    table_desc: str
