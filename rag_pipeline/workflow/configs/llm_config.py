"""LLM configuration."""

from dataclasses import dataclass


@dataclass
class LLMConfig:
    """Language model configuration."""

    model_name: str = "llama3.2" #for ollama
    openai_model_name: str = "gpt-4o-mini" #for openai