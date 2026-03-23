"""
LLM protocol - defines contract for language model implementations.
"""

from typing import Protocol

from langchain_core.messages import BaseMessage


class LLMProtocol(Protocol):
    """Protocol for LLM implementations."""

    def invoke(self, messages: list[BaseMessage]) -> BaseMessage:
        """
        Invoke the LLM with a list of messages.
        
        Args:
            messages: List of messages to send to the LLM.
            
        Returns:
            The LLM's response as a BaseMessage.
        """
        ...