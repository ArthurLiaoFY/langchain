from typing import List

from langchain_core.documents.base import Document

from langchain.tools import tool


@tool
def str_to_doc(content: str, source: str) -> Document:
    """Convert content to Document"""
    return Document(page_content=content, metadata={"source": source})


@tool
def join_docs(docs: List[Document], separator: str):
    """Join Docs by separator"""
    return f"{separator}".join(doc.page_content for doc in docs)
