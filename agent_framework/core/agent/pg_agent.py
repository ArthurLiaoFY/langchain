from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent_framework.core.nodes.pg_nodes import (
    connect_database_node,
    extract_table_summary_node,
    get_database_common_info_node,
    reconnect_database_node,
)
from agent_framework.core.routes.pg_routes import database_connection_route
from agent_framework.core.states.pg_states import DatabaseState, PostgresConnectionInfo


def connect_postgres_agent() -> CompiledStateGraph:
    graph = StateGraph(PostgresConnectionInfo)
    graph.add_node(node="connect_db", action=connect_database_node)
    graph.add_node(node="reconnect_db", action=reconnect_database_node)

    graph.add_edge(start_key=START, end_key="connect_db")
    graph.add_conditional_edges(
        source="connect_db",
        path=database_connection_route,
    )
    graph.add_conditional_edges(
        source="reconnect_db",
        path=database_connection_route,
    )

    return graph.compile()


def extract_table_summary_agent() -> CompiledStateGraph:
    graph = StateGraph(DatabaseState)
    graph.add_node("get_database_common_info", get_database_common_info_node)
    graph.add_node("get_database_common_info", get_database_common_info_node)
    graph.add_node("extract_table_summary", extract_table_summary_node)

    graph.add_edge(START, "get_database_common_info")
    graph.add_edge("get_database_common_info", "extract_table_summary")
    graph.add_edge("extract_table_summary", END)

    return graph.compile()
