from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent_framework.core.nodes.pg_to_qdrant_nodes import (
    check_point_exist_node,
    extract_table_summary_node,
    generate_respective_sql_code_node,
    get_related_documents_node,
    get_table_info_node,
    get_vector_store_info_node,
    upsert_to_vector_database_node,
)
from agent_framework.core.states.pg_to_qdrant_states import PostgresQdrantState


def table_summary_upsert_agent() -> CompiledStateGraph:
    graph = StateGraph(PostgresQdrantState)
    graph.add_node(node="get_tables_info", action=get_table_info_node)
    graph.add_node(node="get_vector_store_info", action=get_vector_store_info_node)
    graph.add_node(node="gather", action=RunnablePassthrough())
    graph.add_node(node="filter_exist_points", action=check_point_exist_node)
    graph.add_node(node="get_related_documents", action=get_related_documents_node)
    graph.add_node(node="extract_table_summary", action=extract_table_summary_node)
    graph.add_node(node="point_upsert", action=upsert_to_vector_database_node)

    graph.add_node(
        node="generate_respective_sql_code", action=generate_respective_sql_code_node
    )

    graph.add_edge(start_key=START, end_key="get_tables_info")
    graph.add_edge(start_key=START, end_key="get_vector_store_info")
    graph.add_edge(
        start_key=["get_tables_info", "get_vector_store_info"],
        end_key="gather",
    )
    graph.add_edge(
        start_key="gather",
        end_key="filter_exist_points",
    )
    graph.add_edge(start_key="filter_exist_points", end_key="extract_table_summary")
    graph.add_edge(start_key="extract_table_summary", end_key="point_upsert")
    graph.add_edge(start_key="point_upsert", end_key=END)

    graph.add_edge(
        start_key="gather",
        end_key="get_related_documents",
    )
    graph.add_edge(
        start_key="get_related_documents", end_key="generate_respective_sql_code"
    )
    graph.add_edge(start_key="generate_respective_sql_code", end_key=END)

    return graph.compile()
