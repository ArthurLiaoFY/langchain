# %%
import json
import os
from typing import Annotated, Literal

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

os.chdir("..")

with open("secrets.json") as f:
    secrets = json.loads(f.read())
with open("config.json") as f:
    config = json.loads(f.read())
os.environ["LANGSMITH_API_KEY"] = secrets.get("langsmith").get("api_key")
os.environ["TAVILY_API_KEY"] = secrets.get("tavily").get("api_key")

llm_model = ChatOllama(
    model="qwen2.5:14b",
    temperature=0,
)


# %%


@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """
    Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user.
    """
    repl = PythonREPL()
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = (
        f"Successfully executed:\n\`\`\`python\n{code}\n\`\`\`\nStdout: {result}"
    )
    return (
        result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )


# python_repl_tool.invoke(
#     {
#         "code": "import matplotlib.pyplot as plt; plt.plot([1,2,3]); plt.show()",
#     }
# )


# %%
def make_system_prompt(suffix: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant, collaborating with other assistants."
                " Use the provided tools to progress towards answering the question."
                " If you are unable to fully answer, that's OK, another assistant with different tools "
                " will help where you left off. Execute what you can to make progress."
                " If you or any of the other assistants have the final answer or deliverable,"
                " prefix your response with FINAL ANSWER so the team knows to stop."
                f"\n{suffix}",
            )
        ]
    )


# %%
chart_agent = create_react_agent(
    model=llm_model,
    tools=[python_repl_tool],
    prompt=make_system_prompt(
        suffix="You can only generate chat. "
        "You are working with a model fitting colleague and model performance checking colleague and load data colleague."
    ),
)
model_fitting_agent = create_react_agent(
    model=llm_model,
    tools=[python_repl_tool],
    prompt=make_system_prompt(
        suffix="You can only fit model from provided data. "
        "You are working with a chart colleague and model performance checking colleague and load data colleague."
    ),
)

model_performance_check_agent = create_react_agent(
    model=llm_model,
    tools=[python_repl_tool],
    prompt=make_system_prompt(
        suffix="You can only check the performance of the model after fit model. "
        "You are working with model fitting colleague and chart colleague and load data colleague."
    ),
)
load_data_agent = create_react_agent(
    model=llm_model,
    tools=[python_repl_tool],
    prompt=make_system_prompt(
        suffix="You can only load the data from the provide file path. "
        "You are working with model fitting colleague and chart colleague and model performance checking colleague."
    ),
)


def get_next_node(last_message: BaseMessage, goto: str):
    if "FINAL ANSWER" in last_message.content:
        # Any agent decided the work is done
        return END
    return goto


# %%
def chart_node(state: MessagesState) -> Command[Literal["researcher", "__end__"]]:
    result = chart_agent.invoke(state)
    goto = get_next_node(last_message=result["messages"][-1], goto=result["next_agent"])

    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content, name="chart_generator"
    )
    return Command(
        update={
            # share internal message history of chart agent with other agents
            "messages": result["messages"],
        },
        goto=goto,
    )


def chart_node(state: MessagesState) -> Command[Literal["researcher", "__end__"]]:
    result = chart_agent.invoke(state)
    goto = get_next_node(last_message=result["messages"][-1], goto=result["next_agent"])

    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content, name="chart_generator"
    )
    return Command(
        update={
            # share internal message history of chart agent with other agents
            "messages": result["messages"],
        },
        goto=goto,
    )


# %%

workflow = StateGraph(MessagesState)
workflow.add_node("load data", load_data_agent)
workflow.add_node("fit model", model_fitting_agent)
workflow.add_node("check model performance", model_performance_check_agent)
workflow.add_node("plot fitted plot", model_fitting_agent)

workflow.add_edge(START, "load data")
graph = workflow.compile()

# %%
filepath = "/Users/wr80340/WorkSpace/langchain/agent_tutorial/iris/bezdekIris.data"
x_columns = ", ".join(["sepallength", "sepalwidth", "petallength", "petalwidth"])
y_columns = "target"
events = graph.stream(
    input={
        "messages": [
            (
                "user",
                f"First, load the data from path: {filepath}, set column name as: {x_columns}, {y_columns}"
                f"fit a classification model from X data: {x_columns}, and y: {y_columns}. "
                "And make the model performance checking.",
                "Finally you make the model performance chart, finish.",
            )
        ],
    },
    config={"recursion_limit": 4},
)
for s in events:
    print(s)
    print("----")
# %%%
