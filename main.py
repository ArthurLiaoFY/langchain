# %%
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph


class State(MessagesState):
    db: SQLDatabase
    table_infos: str


def connector(state: State):
    return {"messages": HumanMessage(content="connector")}


def sql_table_info_splitter(state: State):
    result = {}
    delimiter = "CREATE"
    for idx, single_table_info in enumerate(
        state["db"].get_table_info().split(delimiter)
    ):
        print(idx)
        if single_table_info not in ("", "\n"):
            result[idx] = delimiter + single_table_info
    return result


def sql_entry(sql_query: str, owner: str):
    # select database_uri by owner
    match owner:
        case "WJ1":
            database_uri = "sqlite:///Chinook.db"

        case "WJ2":
            database_uri = "sqlite:///Chinook.db"

        case "WJ3":
            database_uri = "sqlite:///Chinook.db"

    # connect to DB and setup
    db = SQLDatabase.from_uri(database_uri=database_uri)
    db._sample_rows_in_table_info = 0

    # init graph
    rma_graph = StateGraph(State)

    # init nodes
    rma_graph.add_node(
        "sql_table_info_splitter",
        sql_table_info_splitter,
    )
    rma_graph.add_node(
        "single_sql_table_info_summarizer",
        single_sql_table_info_summarizer,
    )
    rma_graph.add_node(
        "connector",
        connector,
    )

    # init edges
