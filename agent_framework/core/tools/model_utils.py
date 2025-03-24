from langchain_ollama import ChatOllama


def load_llm_model(model_name: str):
    return ChatOllama(model=model_name, temperature=0)
