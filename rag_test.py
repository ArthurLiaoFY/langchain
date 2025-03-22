# %%
import json

import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Qdrant
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from langchain import hub
from langchain.cache import InMemoryCache
from langchain.globals import set_debug, set_llm_cache
from langchain.text_splitter import RecursiveCharacterTextSplitter

set_llm_cache(InMemoryCache())
set_debug(True)
with open("api_keys.json") as f:
    api_keys = json.loads(f.read())
# %%
# Load Documents
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

# %%

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Embed
# %%
# setup vector store
qdrant_client = QdrantClient(
    url=api_keys.get("qdrant_url"),
    api_key=api_keys.get("qdrant_api_key"),
)
collection = "test_split_example"

if not qdrant_client.collection_exists(collection):
    qdrant_client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(
            size=5120,
            distance=Distance.COSINE,
        ),
    )
vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=collection,
    embedding=OllamaEmbeddings(model="deepseek-r1:14b"),
)
# vector_store.add_documents(
#     documents=splits,
# )

retriever = vector_store.as_retriever(search_kwargs={"k": 3})
# %%
#### RETRIEVAL and GENERATION ####

# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """
            You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question. 
            If you don't know the answer, just say that you don't know. 
            Use three sentences maximum and keep the answer concise.

            Question: {question} 
            Context: {context} 
            Answer:
            """,
        )
    ]
)

# %%
# LLM
llm = ChatOllama(model="deepseek-r1:14b", temperature=0)


# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Chain
rag_chain = (
    RunnableParallel(
        context=retriever | format_docs,
        question=RunnablePassthrough(),
    )
    | prompt
    | llm
    | StrOutputParser()
)
# %%
# Question
res = rag_chain.invoke(input="What is Task Decomposition?")
# %%