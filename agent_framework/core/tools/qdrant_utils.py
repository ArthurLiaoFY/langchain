from typing import Dict, List, Union

from langchain_core.documents.base import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams

from langchain.tools import tool


@tool
def connect_qdrant_client(
    qdrant_connection_info: Dict[str, str],
) -> Union[QdrantClient, None]:
    """Connect to Qdrant client"""
    client = QdrantClient(
        url=qdrant_connection_info.get("url"),
        api_key=qdrant_connection_info.get("api_key"),
    )
    try:
        client.get_locks()
        return client
    except Exception as e:
        return None


@tool
def create_collection_vector_store(
    qdrant_client: QdrantClient,
    collection: str,
    llm_vector_size: int,
) -> None:
    """Connect to collection vector store in Qdrant database"""
    qdrant_client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(
            size=llm_vector_size,
            distance=Distance.COSINE,
        ),
    )
    return None


@tool
def connect_collection_vector_store(
    qdrant_client: QdrantClient,
    collection: str,
    llm_embd: OllamaEmbeddings,
) -> Union[QdrantVectorStore, None]:
    """Connect to collection vector store in Qdrant database"""
    try:
        return QdrantVectorStore(
            client=qdrant_client,
            collection_name=collection,
            embedding=llm_embd,
        )
    except:
        return None


def retrieve_vector_store(vector_store: QdrantVectorStore, k_related_docs: int):
    """Retrieve data from vector store"""
    vector_store.as_retriever(search_kwargs={"k": k_related_docs})
    pass


def insert_vector_store(
    vector_store: QdrantVectorStore,
    docs: List[Document],
):
    """Insert data to vector store"""
    vector_store.add_documents(documents=docs)
    pass


def check_payload_match(client: QdrantClient, collection_name: str, table_name: str):
    print(
        client.count(
            collection_name=collection_name,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="table_name",
                        match=models.MatchValue(value=table_name),
                    ),
                ]
            ),
            exact=True,
        )
    )
