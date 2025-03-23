import psycopg2
from psycopg2.extensions import connection
from typing_extensions import Dict, List, Union

from langchain.tools import tool


@tool
def connect_db(
    postgres_connection_infos: Dict[str, Union[int, str]],
) -> Union[connection, None]:
    """Connect to Postgres database"""
    try:
        return psycopg2.connect(
            host=postgres_connection_infos.get("host"),
            port=postgres_connection_infos.get("port"),
            dbname=postgres_connection_infos.get("dbname"),
            user=postgres_connection_infos.get("user"),
            password=postgres_connection_infos.get("password"),
        )

    except psycopg2.OperationalError:
        return None


@tool
def close_connection(database: connection) -> None:
    """Close connection to Postgres database"""
    database.close()


@tool
def get_table_list(database: connection) -> List[str]:
    """Get a list of tables in the database."""
    with database.cursor() as curs:
        curs.execute(
            """
            SELECT table_name 
            FROM information_schema.tables
            WHERE table_schema = 'public'
            """
        )
        return [row[0] for row in curs.fetchall()]


@tool
def get_table_columns(database: connection, table_name: str) -> List[str]:
    """Get whole columns from table."""
    with database.cursor() as curs:
        curs.execute(
            f"""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = '{table_name}';
            """
        )
        return [row[0] for row in curs.fetchall()]


@tool
def get_related_tables(database: connection, table_name: str) -> List[str]:
    """Find all the table related with table input."""
    with database.cursor() as curs:
        curs.execute(
            f"""
            SELECT 
                tc.table_name AS table_name
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
                ON tc.constraint_name = kcu.constraint_name
            WHERE 1=1
                AND tc.constraint_type = 'FOREIGN KEY' 
                AND kcu.table_name = '{table_name}';
            """
        )
        return [row[0] for row in curs.fetchall()]


@tool
def get_relationship_desc(database: connection, table_name: str) -> str:
    """Find all the table relationship desc."""
    with database.cursor() as curs:
        curs.execute(
            f"""
            SELECT 
                kcu.column_name AS fk_column,
                ccu.table_name AS referenced_table,
                ccu.column_name AS referenced_column
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
                ON tc.constraint_name = kcu.constraint_name
            JOIN information_schema.constraint_column_usage AS ccu
                ON ccu.constraint_name = tc.constraint_name
            WHERE 1=1
                AND tc.constraint_type = 'FOREIGN KEY' 
                AND tc.table_name = '{table_name}';
            """
        )
        foreign_key_infos = [
            f"foreign key {row[0]} references {row[2]} in table {row[1]}"
            for row in curs.fetchall()
        ]
        return (
            ", " + " and ".join(foreign_key_infos) + "."
            if len(foreign_key_infos) > 0
            else ""
        )


@tool
def get_sample_data(
    database: connection, table_name: str, sample_size: int = 10
) -> List[tuple]:
    """Query sample from table with specified sample size."""
    with database.cursor() as curs:
        curs.execute(
            f"""
            SELECT *
            FROM public."{table_name}"
            LIMIT {sample_size};
            """
        )
        return curs.fetchall()


@tool
def query(database: connection, query: str) -> List[tuple]:
    """Query database with specified query."""
    with database.cursor() as curs:
        curs.execute(query)
        return curs.fetchall()


@tool
def generate_table_description(database: connection, query: str) -> str:
    """Generate table description from SQL infos by LLM model"""
    pass
