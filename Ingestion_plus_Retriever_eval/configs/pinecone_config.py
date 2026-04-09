from dataclasses import dataclass
import re

@dataclass
class PineconeConfig:
    index_name : str = "final-rag-index-openai-small"
    metric : str = "dotproduct"
    batch_size : int = 50
    dense_embedding_model_name : str = "sentence-transformers/all-MiniLM-L6-v2"
    sparse_embedding_model_name : str = "naver/splade-cocondenser-ensembledistil"
    cloud : str = "aws"
    region : str = "us-east-1"
