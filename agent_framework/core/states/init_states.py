from typing_extensions import Annotated, Dict, List, TypedDict, Union

from agent_framework.core.states.pg_states import DatabaseState


class InitState(TypedDict):
    # a chat bot
    database_related: DatabaseState
