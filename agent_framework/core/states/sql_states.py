from psycopg2.extensions import connection
from typing_extensions import Annotated, Dict, List, TypedDict


class PostgresDatabaseState(TypedDict):
    # ---------------------------
    host: str
    port: int
    dbname: str
    user: str
    password: str
    # ---------------------------
    database: connection
    # ---------------------------
    tables: List[str]
    table_infos: Dict[str, Dict[str, str]]
    # ---------------------------


class TableState(TypedDict):
    table_name: str
    table_columns: List[str]
    related_tables: List[str]
    relationship_desc: str
    table_desc: str
