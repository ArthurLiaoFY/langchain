from typing import List

from langchain_core.documents.base import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from langchain.tools import tool


@tool
def connect_qdrant_client(url: str, api_key: str) -> QdrantClient:
    """Connect to Qdrant client"""
    return QdrantClient(
        url=url,
        api_key=api_key,
    )


@tool
def connect_collection_vector_store(
    client: QdrantClient, collection_name: str, llm_vector_size: int, llm_model: str
) -> QdrantVectorStore:
    """Connect to collection vector store in Qdrant database"""
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=llm_vector_size,
                distance=Distance.COSINE,
            ),
        )
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=OllamaEmbeddings(model=llm_model),
    )


def retrieve_vector_store(vector_store: QdrantVectorStore, k_related_docs: int):
    """Retrieve data from vector store"""
    vector_store.as_retriever(search_kwargs={"k": k_related_docs})
    pass


def insert_vector_store(vector_store: QdrantVectorStore, docs: List[Document]):
    """Insert data to vector store"""
    vector_store.add_documents(
        documents=docs,
    )
    pass
