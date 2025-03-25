from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent_framework.core.nodes.pg_nodes import (
    connect_database_node,
    delete_connection_info_node,
    extract_fake_summary_node,
    extract_table_summary_node,
    get_database_common_info_node,
    reconnect_database_node,
)
from agent_framework.core.routes.llm_routes import llm_inference_route
from agent_framework.core.routes.pg_routes import database_connection_route
from agent_framework.core.states.pg_states import DatabaseState, PostgresConnectionInfo


def connect_postgres_agent() -> CompiledStateGraph:
    graph = StateGraph(PostgresConnectionInfo)
    graph.add_node(node="connect_db", action=connect_database_node)
    graph.add_node(node="reconnect_db", action=reconnect_database_node)
    graph.add_node(node="delete_sensitive_info", action=delete_connection_info_node)

    graph.add_edge(start_key=START, end_key="connect_db")
    graph.add_conditional_edges(
        source="connect_db",
        path=database_connection_route,
    )
    graph.add_conditional_edges(
        source="reconnect_db",
        path=database_connection_route,
    )
    graph.add_edge(start_key="delete_sensitive_info", end_key=END)

    return graph.compile()


def extract_table_summary_agent() -> CompiledStateGraph:
    graph = StateGraph(DatabaseState)
    graph.add_node(
        node="get_database_common_info", action=get_database_common_info_node
    )
    graph.add_node(node="extract_table_summary", action=extract_table_summary_node)
    graph.add_node(node="extract_fake_summary", action=extract_fake_summary_node)

    graph.add_edge(start_key=START, end_key="get_database_common_info")
    graph.add_conditional_edges(
        source="get_database_common_info", path=llm_inference_route
    )
    graph.add_edge(start_key="extract_table_summary", end_key=END)
    graph.add_edge(start_key="extract_fake_summary", end_key=END)

    return graph.compile()
