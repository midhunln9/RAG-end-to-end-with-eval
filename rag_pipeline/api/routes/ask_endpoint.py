"""
Ask endpoint for RAG pipeline.

Provides HTTP interface to query the RAG pipeline and get responses.
"""

import logging

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from rag_pipeline.api.container import get_container
from rag_pipeline.workflow.service import RAGService

logger = logging.getLogger(__name__)
ask_endpoint = APIRouter()


class AskRequest(BaseModel):
    """Request model for the ask endpoint."""

    query: str
    session_id: str


class AskResponse(BaseModel):
    """Response model from the ask endpoint."""

    response: str
    rewritten_query: str
    documents_retrieved: int
    session_id: str


def get_rag_service() -> RAGService:
    """Dependency injection function for RAGService."""
    container = get_container()
    return container.get_rag_service()


@ask_endpoint.post("/ask", response_model=AskResponse)
async def ask(
    request: AskRequest,
    service: RAGService = Depends(get_rag_service),
) -> AskResponse:
    """
    Query the RAG pipeline and get a response.
    
    Args:
        request: The ask request containing query and session_id.
        service: RAGService injected via dependency.
        
    Returns:
        AskResponse containing the generated response and metadata.
        
    Raises:
        HTTPException: If processing fails.
    """
    try:
        logger.info(
            f"Processing query for session {request.session_id}: {request.query[:50]}..."
        )
        result = service.run(query=request.query, session_id=request.session_id)

        return AskResponse(
            response=result["response"],
            rewritten_query=result["rewritten_query"],
            documents_retrieved=result["documents_retrieved"],
            session_id=result["session_id"],
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing query")

