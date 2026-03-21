from typing import List
from langchain_core.documents import Document
from Protocols.vector_db_protocol import VectorDBProtocol
import logging

class UpsertService:
    def __init__(self, vector_db_repository: VectorDBProtocol):
        self.vector_db_repository = vector_db_repository
        self.logger = logging.getLogger(__name__)

    def upsert_chunks(self, chunks: List[Document]):
        self.vector_db_repository.upsert_chunks(chunks)
        self.logger.info(f"Successfully upserted {len(chunks)} chunks into the vector database")