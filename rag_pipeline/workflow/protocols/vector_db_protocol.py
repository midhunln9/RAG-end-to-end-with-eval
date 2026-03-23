"""
Vector database protocol - defines contract for vector storage and retrieval.
"""

from typing import Protocol

from langchain_core.documents import Document


class VectorDBProtocol(Protocol):
    """Protocol for vector database implementations."""

    def query(self, query: str) -> list[Document]:
        """
        Query the vector database for documents matching the query.
        
        Args:
            query: The query string to search for.
            
        Returns:
            List of relevant documents.
        """
        ...
