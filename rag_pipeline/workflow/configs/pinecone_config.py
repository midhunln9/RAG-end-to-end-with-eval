"""Pinecone vector database configuration."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PineconeConfig:
    """Pinecone index and embedding model configuration."""

    index_name: str = "final-rag-index-openai-small"
    metric: str = "dotproduct"
    batch_size: int = 200
    dense_embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    sparse_embedding_model_name: str = "naver/splade-cocondenser-ensembledistil"
    cloud: str = "aws"
    region: str = "us-east-1"
    
    @classmethod
    def from_settings(cls, settings):
        """
        Create PineconeConfig from application settings.
        
        This ensures configuration alignment between Settings and PineconeConfig,
        preventing dimension mismatches and other config divergence issues.
        
        Args:
            settings: Application Settings instance
            
        Returns:
            PineconeConfig instance with values from settings
        """
        return cls(
            index_name=settings.pinecone_index_name,
            metric=settings.pinecone_metric,
            batch_size=settings.pinecone_batch_size,
            dense_embedding_model_name=settings.pinecone_dense_embedding_model,
            sparse_embedding_model_name=settings.pinecone_sparse_embedding_model,
            cloud=settings.pinecone_cloud,
            region=settings.pinecone_region,
        )

