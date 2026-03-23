"""
Sparse embedding strategy - abstract base for sparse embeddings.

Defines the interface for generating sparse vector embeddings.
"""

from abc import ABC, abstractmethod


class SparseEmbeddingStrategy(ABC):
    """
    Abstract base class for sparse embedding implementations.
    
    Sparse embeddings are high-dimensional vectors with mostly zero values,
    typically used for lexical matching in hybrid search systems.
    """

    @abstractmethod
    def embed_documents(self, documents: list[str]) -> list[dict]:
        """
        Generate sparse embeddings for multiple documents.
        
        Args:
            documents: List of document texts to embed.
            
        Returns:
            List of sparse embedding dicts with 'indices' and 'values' keys.
        """
        ...

    @abstractmethod
    def embed_query(self, query: str) -> dict:
        """
        Generate a sparse embedding for a query string.
        
        Args:
            query: Query text to embed.
            
        Returns:
            Sparse embedding dict with 'indices' and 'values' keys.
        """
        ...