from langgraph.graph import END, START, StateGraph

from agent_framework.core.states.pg_states import PostgresDatabaseState


def reconnect_db(state: PostgresDatabaseState):
    """Reconnect to database"""
    if state["is_connected"]:
        return END
    else:
        if state["recursion_time"] < state["recursion_limit"]:
            return "connect_db_node"
        else:
            return END
