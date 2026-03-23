"""
FastAPI application entry point for RAG Pipeline.

Configures the FastAPI application with routes, middleware, and lifespan management.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from rag_pipeline.api.container import get_container
from rag_pipeline.api.routes.ask_endpoint import ask_endpoint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.
    
    Handles startup and shutdown events for dependency initialization.
    """
    # Startup
    container = get_container()
    await container.startup()
    yield
    # Shutdown
    await container.shutdown()


# Create FastAPI application
app = FastAPI(
    title="RAG Pipeline API",
    description="Retrieval-Augmented Generation pipeline API",
    version="1.0.0",
    lifespan=lifespan,
)

# Include routers
app.include_router(ask_endpoint)


@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {
        "message": "RAG Pipeline API is running",
        "status": "healthy",
    }


@app.get("/health", tags=["Health"])
async def health():
    """Detailed health check endpoint."""
    return {
        "status": "healthy",
        "service": "rag_pipeline",
    }

