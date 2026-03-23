"""Database configuration (deprecated - use workflow.config.Settings instead)."""

from dataclasses import dataclass


@dataclass
class DBConfig:
    """Legacy database config - kept for backward compatibility."""
    database_url: str = "sqlite:///sample.db"