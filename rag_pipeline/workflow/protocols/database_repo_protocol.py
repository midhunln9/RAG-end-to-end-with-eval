"""
Database repository protocol - defines contract for conversation persistence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from sqlalchemy.orm import Session
    from rag_pipeline.workflow.database.models.conversations import Conversation


class DatabaseRepositoryProtocol(Protocol):
    """Protocol for database repository implementations."""

    def add_conversation(
        self, session: Session, session_id: str, messages: str
    ) -> None:
        """Add a conversation record to the database."""
        ...

    def get_conversations_by_session_id(
        self, session: Session, session_id: str
    ) -> list[Conversation]:
        """Retrieve all conversations for a given session ID."""
        ...