from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent_framework.core.states.pg_to_qdrant_states import Postgres2QdrantState


def foo():
    return None


def table_summary_upsert_agent() -> CompiledStateGraph:
    graph = StateGraph(Postgres2QdrantState)
    graph.add_node(node="get_tables_info", action=foo)
    graph.add_node(node="get_vector_store_info", action=foo)
    graph.add_node(node="check_point_exist", action=foo)
    graph.add_node(node="point_upsert", action=foo)

    graph.add_edge(start_key=START, end_key="get_tables_info")
    graph.add_edge(start_key=START, end_key="get_vector_store_info")
    graph.add_edge(
        start_key=["get_tables_info", "get_vector_store_info"],
        end_key="check_point_exist",
    )

    graph.add_edge(start_key="check_point_exist", end_key="point_upsert")
    graph.add_edge(start_key="point_upsert", end_key=END)

    return graph.compile()
