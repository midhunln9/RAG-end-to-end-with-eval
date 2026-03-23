"""
Dense embedding strategy - abstract base for dense embeddings.

Defines the interface for generating dense vector embeddings.
"""

from abc import ABC, abstractmethod

from langchain_core.documents import Document


class DenseEmbeddingStrategy(ABC):
    """
    Abstract base class for dense embedding implementations.
    
    Dense embeddings are typically semantic vectors of moderate dimensionality
    (e.g., 384-768 dimensions) suitable for similarity-based retrieval.
    """

    @abstractmethod
    def get_sentence_embedding_dimension(self) -> int:
        """
        Get the dimensionality of embeddings from this strategy.
        
        Returns:
            Number of dimensions in the embedding vectors.
        """
        ...

    @abstractmethod
    def get_embeddings(self, documents: list[Document]) -> list[list[float]]:
        """
        Generate embeddings for multiple documents.
        
        Args:
            documents: List of documents to embed.
            
        Returns:
            List of embedding vectors.
        """
        ...

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """
        Generate an embedding for a query string.
        
        Args:
            query: Query text to embed.
            
        Returns:
            Embedding vector for the query.
        """
        ...
