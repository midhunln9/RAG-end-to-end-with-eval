"""
Main entry point for the document ingestion pipeline.

This module initializes and runs the document processing pipeline,
including document loading, chunking, and text splitting.
"""

import os
import logging

from src.pipeline import Pipeline
from src.chunker_service import ChunkerService
from src.upsert_service import UpsertService
from Repositories.file_repository import FileRepository
from Repositories.pinecone_repository import PineconeRepository
from src.recursive_character_text_splitting import RecursiveCharacterTextSplitting
from src.sentence_transformer_embedding import SentenceTransformerEmbedding
from src.sparse_embedding import SentenceTransformerSparseEmbedding
from configs.recursive_text_splitter_config import RecursiveCharacterTextSplittingConfig
from configs.pinecone_config import PineconeConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter
from logger import setup_logging
from dotenv import load_dotenv, find_dotenv


def main():
    """
    Main function to initialize and run the document ingestion pipeline.

    This function sets up logging, creates all necessary components
    (file repository, text splitter, chunker service), and executes
    the pipeline to process documents.

    Returns:
        list: A list of document chunks extracted from the processed documents.
    """
    load_dotenv("/Users/midhunln/Documents/rag20march_with_eval/Ingestion_plus_Retriever_eval/ingestion.env")
    
    # Suppress verbose logging from third-party libraries before they're imported
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers.modeling_utils").setLevel(logging.WARNING)
    
    logger = setup_logging()
    logger.info("Starting document ingestion pipeline")

    logger.info("Initializing file repository")
    file_repository = FileRepository()

    logger.info("Configuring text splitting parameters")
    config = RecursiveCharacterTextSplittingConfig(
        chunk_size=1000,
        chunk_overlap=200
    )
    logger.debug(f"Chunk size: {config.chunk_size}, Chunk overlap: {config.chunk_overlap}")

    logger.info("Initializing recursive character text splitter")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap
    )
    splitting_strategy = RecursiveCharacterTextSplitting(config, splitter)

    logger.info("Initializing chunker service")
    chunker_service = ChunkerService(
        document_repository=file_repository,
        splitter_strategy=splitting_strategy
    )

    logger.info("Initializing Pinecone configuration and embedding strategies")
    pinecone_config = PineconeConfig()
    dense_embedding_strategy = SentenceTransformerEmbedding(pinecone_config)
    sparse_embedding_strategy = SentenceTransformerSparseEmbedding(pinecone_config)

    logger.info("Initializing Pinecone repository")
    pinecone_repository = PineconeRepository(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws"),
        dense_embedding_strategy=dense_embedding_strategy,
        sparse_embedding_strategy=sparse_embedding_strategy,
        pinecone_config=pinecone_config
    )

    logger.info("Initializing upsert service")
    upsert_service = UpsertService(vector_db_repository=pinecone_repository)

    logger.info("Initializing pipeline")
    pipeline = Pipeline(
        chunker_service=chunker_service,
        document_repository=file_repository,
        upsert_service=upsert_service
    )

    logger.info("Running pipeline")
    chunks = pipeline.run()

    logger.info(f"Successfully processed {len(chunks)} chunks from documents")
    print(f"Successfully processed {len(chunks)} chunks from documents")
    return chunks


if __name__ == "__main__":
    main()
