"""
Dependency container for FastAPI application.

Manages initialization and lifetime of application dependencies.
"""

import logging
from typing import AsyncGenerator

from rag_pipeline.workflow.config import Settings
from rag_pipeline.workflow.database.db_repositories.conversation_repository import (
    ConversationRepository,
)
from rag_pipeline.workflow.database.sessions import Database
from rag_pipeline.workflow.embeddings.sentence_transformer_embedding import (
    SentenceTransformerEmbedding,
)
from rag_pipeline.workflow.embeddings.sparse_embedding import (
    SentenceTransformerSparseEmbedding,
)
from rag_pipeline.workflow.llms.ollama_llama import OllamaLLM
from rag_pipeline.workflow.node_orchestrator import Nodes
from rag_pipeline.workflow.repositories.pinecone_repository import PineconeRepository
from rag_pipeline.workflow.service import RAGService
from rag_pipeline.workflow.graph import RAGWorkflow

logger = logging.getLogger(__name__)


class AppContainer:
    """
    Application dependency container.
    
    Manages initialization and lifecycle of all application dependencies.
    """

    def __init__(self, settings: Settings):
        """
        Initialize container with settings.
        
        Args:
            settings: Application settings.
        """
        self.settings = settings
        self._database: Database | None = None
        self._vector_db: PineconeRepository | None = None
        self._llm: OllamaLLM | None = None
        self._conversation_repo: ConversationRepository | None = None
        self._rag_service: RAGService | None = None
        self._workflow: RAGWorkflow | None = None

    async def startup(self) -> None:
        """Initialize all dependencies at application startup."""
        logger.info("Starting up RAG Pipeline application")

        # Initialize database
        self._database = Database(self.settings.database_url)
        logger.info("Database initialized")

        # Initialize embeddings
        dense_embedding = SentenceTransformerEmbedding(
            pinecone_config=self._get_pinecone_config()
        )
        sparse_embedding = SentenceTransformerSparseEmbedding(
            pinecone_config=self._get_pinecone_config()
        )
        logger.info("Embedding strategies initialized")

        # Initialize vector database
        self._vector_db = PineconeRepository(
            api_key=self.settings.pinecone_api_key,
            environment=self.settings.pinecone_environment,
            pinecone_config=self._get_pinecone_config(),
            dense_embedding_strategy=dense_embedding,
            sparse_embedding_strategy=sparse_embedding,
        )
        logger.info("Vector database initialized")

        # Initialize LLM
        from rag_pipeline.workflow.configs.llm_config import LLMConfig

        llm_config = LLMConfig(model_name=self.settings.llm_model_name)
        self._llm = OllamaLLM(llm_config)
        logger.info("LLM initialized")

        # Initialize repositories
        self._conversation_repo = ConversationRepository()
        logger.info("Conversation repository initialized")

        # Initialize RAG service
        self._rag_service = RAGService(
            database=self._database,
            vector_db=self._vector_db,
            conversation_repository=self._conversation_repo,
            llm=self._llm,
        )
        logger.info("RAG service initialized")

        # Initialize workflow
        nodes = Nodes(service=self._rag_service)
        self._workflow = RAGWorkflow(nodes=nodes)
        logger.info("Workflow initialized")

    async def shutdown(self) -> None:
        """Clean up resources at application shutdown."""
        logger.info("Shutting down RAG Pipeline application")
        # Add cleanup logic here if needed

    def get_workflow(self) -> RAGWorkflow:
        """
        Get the compiled RAG workflow.
        
        Returns:
            RAGWorkflow instance.
            
        Raises:
            RuntimeError: If container not initialized.
        """
        if self._workflow is None:
            raise RuntimeError("Container not initialized. Call startup() first.")
        return self._workflow

    def get_rag_service(self) -> RAGService:
        """
        Get the RAG service.
        
        Returns:
            RAGService instance.
            
        Raises:
            RuntimeError: If container not initialized.
        """
        if self._rag_service is None:
            raise RuntimeError("Container not initialized. Call startup() first.")
        return self._rag_service

    def _get_pinecone_config(self):
        """Create PineconeConfig from settings."""
        from rag_pipeline.workflow.configs.pinecone_config import PineconeConfig

        return PineconeConfig(
            index_name=self.settings.pinecone_index_name,
            metric=self.settings.pinecone_metric,
            batch_size=self.settings.pinecone_batch_size,
            dense_embedding_model_name=self.settings.pinecone_dense_embedding_model,
            sparse_embedding_model_name=self.settings.pinecone_sparse_embedding_model,
            cloud=self.settings.pinecone_cloud,
            region=self.settings.pinecone_region,
        )


# Global container instance
_container: AppContainer | None = None


def get_container() -> AppContainer:
    """Get or create the global container instance."""
    global _container
    if _container is None:
        _container = AppContainer(Settings())
    return _container


async def lifespan_context() -> AsyncGenerator:
    """
    FastAPI lifespan context manager for dependency initialization.
    
    Usage:
        @app.get("/")
        async def root():
            ...
            
        app = FastAPI(lifespan=lifespan_context)
    """
    container = get_container()
    await container.startup()
    yield
    await container.shutdown()
