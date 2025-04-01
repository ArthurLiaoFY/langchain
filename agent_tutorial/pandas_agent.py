# %%
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_ollama import ChatOllama

df = pd.read_csv("att_rawdata_202503310947.csv")
# %%
llm = ChatOllama(
    model="qwen2.5:14b",
    temperature=0,
)
agent_executor = create_pandas_dataframe_agent(
    llm=llm,
    df=df,
    agent_type="tool-calling",
    verbose=True,
    allow_dangerous_code=True,
)

# %%
agent_executor.invoke(
    {
        "what is the mean and std "
        "of att time of WJ3 factory CDP plant device E023140501 "
        "in date 2025/03/26"
    }
)
# %%
# %%
agent_executor.invoke(
    {
        "how many zeros are there in att time "
        "of WJ3 factory CDP plant, give a a summarize group by device, also return the ratio."
    }
)
# %%
df['value'] = df['value'].astype(float); summary_df = df[df['factory']=='WJ3'][df['mfgplantcode']=='CDP']['value'].eq(0).groupby(df['deviceid']).agg(['sum', 'count']); summary_df['ratio'] = (summary_df['sum']/summary_df['count']) * 100; summary_df
# %%
