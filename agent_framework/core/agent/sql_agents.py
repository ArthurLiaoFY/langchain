from langgraph.prebuilt import create_react_agent

from agent_framework.core.model import sql_coder_model

sql_agent = create_react_agent(
    model=sql_coder_model,
    tools=[],
    prompt=(
        "You are a web search assistant, "
        "Always search the internet before answering questions."
    ),
)
