"""
SPLADE sparse embedding implementation.

Uses the SPLADE (Sparse Lexical and Dense Embedding) model for hybrid search.
"""

import torch
from sentence_transformers import SparseEncoder

from rag_pipeline.workflow.configs.pinecone_config import PineconeConfig
from rag_pipeline.workflow.strategies.sparse_embedding_strategy import (
    SparseEmbeddingStrategy,
)


class SentenceTransformerSparseEmbedding(SparseEmbeddingStrategy):
    """
    Sparse embedding implementation using SPLADE model.
    
    Generates sparse vectors suitable for hybrid search combining
    lexical and semantic matching.
    """

    def __init__(self, pinecone_config: PineconeConfig):
        """
        Initialize sparse embedding model.
        
        Args:
            pinecone_config: Configuration containing model name.
        """
        self.model = SparseEncoder(pinecone_config.sparse_embedding_model_name)

    def _sparse_tensor_to_pinecone_dict(self, sparse_tensor: torch.Tensor) -> dict:
        """
        Convert PyTorch sparse tensor to Pinecone sparse dict format.
        
        Args:
            sparse_tensor: PyTorch tensor from SPLADE model.
            
        Returns:
            Dict with 'indices' and 'values' keys for Pinecone.
        """
        dense = sparse_tensor.to_dense().cpu()
        non_zero = torch.nonzero(dense).squeeze(1)
        return {
            "indices": non_zero.tolist(),
            "values": dense[non_zero].tolist(),
        }

    def embed_documents(self, documents: list[str]) -> list[dict]:
        """
        Generate sparse embeddings for documents.
        
        Args:
            documents: List of document texts.
            
        Returns:
            List of sparse embedding dicts.
        """
        embeddings = self.model.encode(documents)
        return [
            self._sparse_tensor_to_pinecone_dict(embedding) for embedding in embeddings
        ]

    def embed_query(self, query: str) -> dict:
        """
        Generate sparse embedding for a query.
        
        Args:
            query: Query text.
            
        Returns:
            Sparse embedding dict.
        """
        embedding = self.model.encode([query])[0]
        return self._sparse_tensor_to_pinecone_dict(embedding)