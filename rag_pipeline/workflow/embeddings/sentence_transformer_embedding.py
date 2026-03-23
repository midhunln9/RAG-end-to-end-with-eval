"""
Sentence Transformer dense embedding implementation.

Uses HuggingFace's Sentence Transformers for generating dense embeddings.
"""

from langchain_huggingface import HuggingFaceEmbeddings

from rag_pipeline.workflow.configs.pinecone_config import PineconeConfig
from rag_pipeline.workflow.strategies.dense_embedding_strategy import (
    DenseEmbeddingStrategy,
)
from langchain_core.documents import Document


class SentenceTransformerEmbedding(DenseEmbeddingStrategy):
    """
    Dense embedding implementation using Sentence Transformers.
    
    Leverages pre-trained transformer models for semantic embedding of text.
    """

    def __init__(self, pinecone_config: PineconeConfig):
        """
        Initialize embedding model.
        
        Args:
            pinecone_config: Configuration containing model name.
        """
        self.model = HuggingFaceEmbeddings(
            model_name=pinecone_config.dense_embedding_model_name
        )

    def get_sentence_embedding_dimension(self) -> int:
        """
        Get the dimensionality of embeddings.
        
        Returns:
            Number of dimensions in the embedding vectors.
        """
        embedding_dim = len(
            self.model.embed_query("This is a test query to get embedding dimension")
        )
        return embedding_dim

    def get_embeddings(self, documents: list[Document]) -> list[list[float]]:
        """
        Generate embeddings for documents.
        
        Args:
            documents: List of documents to embed.
            
        Returns:
            List of embedding vectors.
        """
        return self.model.embed_documents([doc.page_content for doc in documents])

    def embed_query(self, query: str) -> list[float]:
        """
        Generate embedding for a query.
        
        Args:
            query: Query text to embed.
            
        Returns:
            Embedding vector.
        """
        return self.model.embed_query(query)