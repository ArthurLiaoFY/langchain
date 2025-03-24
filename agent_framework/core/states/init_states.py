from typing_extensions import Annotated, Dict, List, TypedDict, Union

from agent_framework.core.states.pg_states import PostgresDatabaseState


class InitState(TypedDict):
    # a chat bot
    database_related: PostgresDatabaseState
