"""
Database session management for SQLAlchemy.

Provides connection pool management and session scoping for ORM operations.
"""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from rag_pipeline.workflow.database.base import Base


class Database:
    """
    Database connection and session manager.
    
    Handles engine initialization, session creation, and transaction management.
    """

    def __init__(self, database_url: str) -> None:
        """
        Initialize database connection pool.
        
        Args:
            database_url: SQLAlchemy database URL.
        """
        self.engine = create_engine(
            database_url, 
            echo=False, 
            future=True
        )
        self.session_maker = sessionmaker(
            bind=self.engine,
            autoflush=False,
            autocommit=False,
        )
        # Create all tables
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """
        Provide a transactional scope for database operations.
        
        Automatically commits on success and rolls back on exception.
        
        Yields:
            SQLAlchemy session object.
        """
        session = self.session_maker()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()