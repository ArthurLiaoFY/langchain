from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent_framework.core.nodes.qdrant_nodes import (
    connect_collection_vector_store_node,
    connect_qdrant_client_node,
    reconnect_qdrant_client_node,
)
from agent_framework.core.routes.llm_routes import llm_inference_route
from agent_framework.core.routes.qdrant_routes import client_connection_route
from agent_framework.core.states.qdrant_states import QdrantConnectionInfo


def connect_vector_database() -> CompiledStateGraph:
    graph = StateGraph(QdrantConnectionInfo)
    graph.add_node(node="connect_qdrant_client", action=connect_qdrant_client_node)
    graph.add_node(node="reconnect_qdrant_client", action=reconnect_qdrant_client_node)

    graph.add_edge(start_key=START, end_key="connect_qdrant_client")
    graph.add_conditional_edges(
        source="connect_qdrant_client", path=client_connection_route
    )
    graph.add_conditional_edges(
        source="reconnect_qdrant_client", path=client_connection_route
    )
    return graph.compile()
