from typing import List
from langchain_core.documents import Document
from rag_pipeline.workflow.strategies.dense_embedding_strategy import DenseEmbeddingStrategy
from openai import OpenAI
import os


class OpenAIEmbedding(DenseEmbeddingStrategy):
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = "text-embedding-3-small"

    def get_sentence_embedding_dimension(self) -> int:
        response = self.client.embeddings.create(
            model=self.model_name,
            input="This is a test query to get embedding dimension"
        )
        embedding_dim = len(response.data[0].embedding)
        return embedding_dim

    def get_embeddings(self, documents: List[Document]) -> List[List[float]]:
        texts = [doc.page_content for doc in documents]

        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )

        return [item.embedding for item in response.data]

    def embed_query(self, query: str) -> List[float]:
        response = self.client.embeddings.create(
            model=self.model_name,
            input=query
        )

        return response.data[0].embedding