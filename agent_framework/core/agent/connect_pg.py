from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent_framework.core.nodes.pg_nodes import connect_database_node
from agent_framework.core.routes.pg_routes import reconnect_db
from agent_framework.core.states.pg_states import PostgresConnectionInfo


def connect_postgres_agent() -> CompiledStateGraph:
    graph = StateGraph(PostgresConnectionInfo)
    graph.add_node("connect_db", connect_database_node)

    graph.add_edge(START, "connect_db")
    graph.add_conditional_edges("connect_db", reconnect_db, [END])

    return graph.compile()
