from dataclasses import dataclass
import re

@dataclass
class PineconeConfig:
    index_name : str = "final-rag-index"
    metric : str = "dotproduct"
    batch_size : int = 200
    dense_embedding_model_name : str = "sentence-transformers/all-MiniLM-L6-v2"
    sparse_embedding_model_name : str = "naver/splade-cocondenser-ensembledistil"
    cloud : str = "aws"
    region : str = "us-east-1"
    
    def __post_init__(self):
        """Validate and sanitize index name to meet Pinecone requirements."""
        if not self._is_valid_index_name(self.index_name):
            raise ValueError(
                f"Invalid index name '{self.index_name}'. "
                "Index name must consist of lowercase alphanumeric characters or hyphens only, "
                "and must start with an alphanumeric character."
            )
    
    @staticmethod
    def _is_valid_index_name(name: str) -> bool:
        """Check if index name follows Pinecone naming rules."""
        if not name:
            return False
        # Must start with alphanumeric, contain only lowercase alphanumeric and hyphens
        pattern = r'^[a-z0-9][a-z0-9-]*$'
        return bool(re.match(pattern, name))
