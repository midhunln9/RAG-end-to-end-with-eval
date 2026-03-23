"""
FastAPI application for RAG Pipeline.

Simple and direct initialization without complex dependency injection.
"""

import logging

from fastapi import FastAPI

from rag_pipeline.workflow.config import Settings
from rag_pipeline.workflow.service import RAGService
from rag_pipeline.workflow.graph import RAGWorkflow
from rag_pipeline.workflow.node_orchestrator import Nodes
from rag_pipeline.workflow.database.sessions import Database
from rag_pipeline.workflow.repositories.pinecone_repository import PineconeRepository
from rag_pipeline.workflow.embeddings.sentence_transformer_embedding import (
    SentenceTransformerEmbedding,
)
from rag_pipeline.workflow.embeddings.sparse_embedding import (
    SentenceTransformerSparseEmbedding,
)
from rag_pipeline.workflow.llms.ollama_llama import OllamaLLM
from rag_pipeline.workflow.database.db_repositories.conversation_repository import (
    ConversationRepository,
)
from rag_pipeline.workflow.configs.pinecone_config import PineconeConfig
from rag_pipeline.workflow.configs.llm_config import LLMConfig
from rag_pipeline.api.routes import ask_endpoint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================================
# Initialize dependencies (simple and direct)
# ============================================================================

settings = Settings()
logger.info(f"Loaded settings for environment: {settings.environment}")

# Database
database = Database(settings.database_url)
logger.info("Database initialized")

# Embeddings
pinecone_config = PineconeConfig(
    index_name=settings.pinecone_index_name,
    metric=settings.pinecone_metric,
    batch_size=settings.pinecone_batch_size,
    dense_embedding_model_name=settings.pinecone_dense_embedding_model,
    sparse_embedding_model_name=settings.pinecone_sparse_embedding_model,
    cloud=settings.pinecone_cloud,
    region=settings.pinecone_region,
)

dense_embedding = SentenceTransformerEmbedding(pinecone_config)
sparse_embedding = SentenceTransformerSparseEmbedding(pinecone_config)
logger.info("Embedding strategies initialized")

# Vector Database
vector_db = PineconeRepository(
    api_key=settings.pinecone_api_key,
    pinecone_config=pinecone_config,
    dense_embedding_strategy=dense_embedding,
    sparse_embedding_strategy=sparse_embedding,
    environment=settings.pinecone_environment,
)
logger.info("Vector database initialized")

# LLM
llm_config = LLMConfig(model_name=settings.llm_model_name)
llm = OllamaLLM(llm_config)
logger.info("LLM initialized")

# Repository
conversation_repo = ConversationRepository()
logger.info("Conversation repository initialized")

# Service
service = RAGService(
    database=database,
    vector_db=vector_db,
    conversation_repository=conversation_repo,
    llm=llm,
)
logger.info("RAG service initialized")

# Workflow
nodes = Nodes(service=service)
workflow = RAGWorkflow(nodes=nodes)
logger.info("Workflow initialized and compiled")

# ============================================================================
# Create FastAPI app and register routes
# ============================================================================

app = FastAPI(
    title="RAG Pipeline API",
    description="Retrieval-Augmented Generation pipeline API",
    version="1.0.0",
)

# Set workflow in the endpoint module
ask_endpoint.set_workflow(workflow)

# Register router
app.include_router(ask_endpoint.router)


@app.get("/", tags=["Health"])
def root():
    """Health check endpoint."""
    return {
        "message": "RAG Pipeline API is running",
        "status": "healthy",
    }


@app.get("/health", tags=["Health"])
def health():
    """Detailed health check endpoint."""
    return {
        "status": "healthy",
        "service": "rag_pipeline",
        "environment": settings.environment,
    }

