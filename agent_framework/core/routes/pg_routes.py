from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agent_framework.core.states.pg_states import DatabaseState


def reconnect_db(state: DatabaseState):
    """Reconnect to database"""
    if state["is_connected"]:
        return "get_database_common_info"
    else:
        if state["recursion_time"] < state["recursion_limit"]:
            return "connect_db_node"
        else:
            # go to init chat bot
            return END


def inspect_table(state: DatabaseState):
    # We will return a list of `Send` objects
    # Each `Send` object consists of the name of a node in the graph
    # as well as the state to send to that node
    return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]
