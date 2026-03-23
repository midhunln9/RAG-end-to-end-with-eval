"""Vector database client configuration."""

from dataclasses import dataclass


@dataclass
class VectorDBConfig:
    """Vector database connection configuration."""

    api_key: str = ""
    environment: str = "production"
