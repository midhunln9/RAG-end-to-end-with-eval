"""
Ask endpoint for RAG pipeline.

Provides HTTP interface to query the RAG pipeline and get responses.
"""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Router will be imported in main.py and registered with the app
router = APIRouter()


class AskRequest(BaseModel):
    """Request model for the ask endpoint."""

    query: str
    session_id: str


class AskResponse(BaseModel):
    """Response model from the ask endpoint."""

    response: str
    session_id: str


# This will be set by main.py during initialization
_workflow = None


def set_workflow(workflow):
    """
    Set the workflow instance for this router.
    Called by main.py during initialization.
    """
    global _workflow
    _workflow = workflow


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """
    Query the RAG pipeline and get a response.
    
    Args:
        request: The ask request containing query and session_id.
        
    Returns:
        AskResponse containing the generated response.
        
    Raises:
        HTTPException: If processing fails.
    """
    if _workflow is None:
        raise HTTPException(status_code=500, detail="Workflow not initialized")
    
    try:
        logger.info(
            f"Processing query for session {request.session_id}: {request.query[:50]}..."
        )
        
        # Execute the workflow
        result = _workflow.execute(
            query=request.query,
            session_id=request.session_id
        )

        return AskResponse(
            response=result["response"],
            session_id=request.session_id,
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing query")

