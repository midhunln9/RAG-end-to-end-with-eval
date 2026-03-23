"""LLM configuration."""

from dataclasses import dataclass


@dataclass
class LLMConfig:
    """Language model configuration."""

    model_name: str = "llama3.2"