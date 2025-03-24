from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent_framework.core.nodes.pg_nodes import (
    extract_table_summary_node,
    get_database_common_info_node,
)
from agent_framework.core.states.pg_states import DatabaseState


def extract_table_summary_agent() -> CompiledStateGraph:
    graph = StateGraph(DatabaseState)
    graph.add_node("get_database_common_info", get_database_common_info_node)
    graph.add_node("get_database_common_info", get_database_common_info_node)
    graph.add_node("extract_table_summary", extract_table_summary_node)

    graph.add_edge(START, "get_database_common_info")
    graph.add_edge("get_database_common_info", "extract_table_summary")
    graph.add_edge("extract_table_summary", END)

    return graph.compile()
