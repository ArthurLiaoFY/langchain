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
with open("secrets.json") as f:
    secrets = json.loads(f.read())

with open("config.json") as f:
    config = json.loads(f.read())
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
qdrant_client = QdrantClient(**secrets.get("qdrant"))
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
    # | prompt
    # | llm
    # | StrOutputParser()
)
# %%
# Question
res = rag_chain.invoke(input="What is Task Decomposition?")
# %%
res["context"]
# %%
"(4) Response generation: LLM receives the execution results and provides summarized results to users.\nTo put HuggingGPT into real world usage, a couple challenges need to solve: (1) Efficiency improvement is needed as both LLM inference rounds and interactions with other models slow down the process; (2) It relies on a long context window to communicate over complicated task content; (3) Stability improvement of LLM outputs and external model services.\n\nThey did an experiment on fine-tuning LLM to call a calculator, using arithmetic as a test case. Their experiments showed that it was harder to solve verbal math problems than explicitly stated math problems because LLMs (7B Jurassic1-large model) failed to extract the right arguments for the basic arithmetic reliably. The results highlight when the external symbolic tools can work reliably, knowing when to and how to use the tools are crucial, determined by the LLM capability.\nBoth TALM (Tool Augmented Language Models; Parisi et al. 2022) and Toolformer (Schick et al. 2023) fine-tune a LM to learn to use external tool APIs. The dataset is expanded based on whether a newly added API call annotation can improve the quality of model outputs. See more details in the “External APIs” section of Prompt Engineering.\n\nFig. 10. A picture of a sea otter using rock to crack open a seashell, while floating in the water. While some other animals can use tools, the complexity is not comparable with humans. (Image source: Animals using tools)\nMRKL (Karpas et al. 2022), short for “Modular Reasoning, Knowledge and Language”, is a neuro-symbolic architecture for autonomous agents. A MRKL system is proposed to contain a collection of “expert” modules and the general-purpose LLM works as a router to route inquiries to the best suitable expert module. These modules can be neural (e.g. deep learning models) or symbolic (e.g. math calculator, currency converter, weather API)."
