from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agent_framework.core.states.qdrant_states import QdrantConnectionInfo


def client_connection_route(state: QdrantConnectionInfo):
    """Reconnect to client"""
    if state["is_connected"]:
        return END
    else:
        if state["recursion_time"] < state["recursion_limit"]:
            return "reconnect_qdrant_client"
        else:
            # go to init chat bot
            return END
