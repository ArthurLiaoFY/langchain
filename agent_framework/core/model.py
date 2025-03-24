import json

from langchain_ollama import ChatOllama

with open("config.json") as f:
    config = json.loads(f.read())

llm_model = ChatOllama(
    model=config.get("llm_model", {}).get("model_name", "deepseek-r1:14b"),
    temperature=0,
)
