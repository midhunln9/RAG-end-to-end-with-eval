"""
RAG (Retrieval-Augmented Generation) utility service.

Provides utility methods for individual workflow nodes to call.
Does not orchestrate - nodes handle the workflow coordination.
"""

import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from rag_pipeline.workflow.database.db_repositories.conversation_repository import (
    ConversationRepository,
)
from rag_pipeline.workflow.database.sessions import Database
from rag_pipeline.workflow.prompts.augment_query_rag import RAG_SYSTEM_PROMPT, RAG_USER_PROMPT
from rag_pipeline.workflow.prompts.query_rewriter import QUERY_REWRITER_PROMPT
from rag_pipeline.workflow.prompts.summary_so_far import SUMMARY_SO_FAR
from rag_pipeline.workflow.protocols.llm_protocol import LLMProtocol
from rag_pipeline.workflow.protocols.vector_db_protocol import VectorDBProtocol

logger = logging.getLogger(__name__)


class RAGService:
    """
    Utility service providing helper methods for RAG workflow nodes.
    
    Each method performs a specific task that individual nodes call.
    Does NOT orchestrate the workflow - nodes coordinate the flow.
    """

    def __init__(
        self,
        database: Database,
        vector_db: VectorDBProtocol,
        conversation_repository: ConversationRepository,
        llm: LLMProtocol,
    ):
        """
        Initialize service with required dependencies.
        
        Args:
            database: Database connection manager.
            vector_db: Vector store for document retrieval.
            conversation_repository: Repository for conversation history.
            llm: Language model for rewriting and generation.
        """
        self.database = database
        self.vector_db = vector_db
        self.conversation_repository = conversation_repository
        self.llm = llm

    def rewrite_query(self, query: str) -> str:
        """
        Rewrite user query for better retrieval.
        
        Args:
            query: Original user query.
            
        Returns:
            Rewritten query string.
        """
        try:
            system_message = SystemMessage(content=QUERY_REWRITER_PROMPT)
            human_message = HumanMessage(content=query)
            response = self.llm.invoke([system_message, human_message])
            return response.content
        except Exception as e:
            error_msg = f"Query rewriting failed: {type(e).__name__}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def retrieve_documents(self, query: str) -> list[Document]:
        """
        Retrieve relevant documents from vector store.
        
        Args:
            query: The query to search for.
            
        Returns:
            List of retrieved documents.
        """
        try:
            documents = self.vector_db.query(query)
            logger.info(f"Retrieved {len(documents)} documents for query")
            return documents
        except Exception as e:
            error_msg = f"Document retrieval failed: {type(e).__name__}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return []

    def generate_context_summary(self, session_id: str) -> str:
        """
        Generate a summary of recent conversation history for context.
        
        Args:
            session_id: User session identifier.
            
        Returns:
            Summary of past conversations or default message if none available.
        """
        try:
            with self.database.session_scope() as session:
                past_conversations = (
                    self.conversation_repository.get_conversations_by_session_id(
                        session, session_id
                    )
                )

                if len(past_conversations) > 5:
                    recent = past_conversations[-5:]
                    summary_prompt = SystemMessage(content=SUMMARY_SO_FAR)
                    history_message = HumanMessage(content=str(recent))
                    summary = self.llm.invoke([summary_prompt, history_message])
                    return summary.content
                else:
                    return "No past conversation summary available."
        except Exception as e:
            error_msg = f"Context summary generation failed: {type(e).__name__}: {str(e)}"
            logger.warning(error_msg, exc_info=True)
            return "No past conversation summary available."

    def generate_response(
        self,
        query: str,
        documents: list[Document],
        context_summary: str,
    ) -> str:
        """
        Generate final RAG response using retrieved documents and context.
        
        Args:
            query: The rewritten query.
            documents: Retrieved documents.
            context_summary: Summary of past conversations.
            
        Returns:
            Generated response from LLM.
        """
        try:
            docs_text = (
                "\n".join(
                    [
                        f"Document: {doc.page_content}\nMetadata: {doc.metadata}"
                        for doc in documents
                    ]
                )
                if documents
                else "No documents retrieved"
            )

            formatted_user_prompt = RAG_USER_PROMPT.format(
                query=query,
                summary=context_summary,
                documents=docs_text
            )
            
            logger.debug(f"Formatted user prompt length: {len(formatted_user_prompt)} characters")
            logger.debug(f"Formatted user prompt preview: {formatted_user_prompt[:500]}...")

            system_message = SystemMessage(content=RAG_SYSTEM_PROMPT)
            user_message = HumanMessage(content=formatted_user_prompt)
            logger.debug("System and user messages created successfully")

            response = self.llm.invoke([system_message, user_message])
            logger.debug(f"LLM response type: {type(response)}")
            logger.debug(f"LLM response object: {response}")
            logger.debug(f"LLM response has content attr: {hasattr(response, 'content')}")
            
            if hasattr(response, 'content'):
                content = response.content
                logger.info(f"Extracted response content (length: {len(content)}): {content[:200] if content else 'EMPTY'}...")
                return content
            else:
                logger.error(f"Response object does not have 'content' attribute. Response: {response}")
                return str(response)
                
        except Exception as e:
            error_msg = f"Response generation failed: {type(e).__name__}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def save_conversation(
        self, session_id: str, messages: list[BaseMessage], response: str
    ) -> None:
        """
        Save conversation to database.
        
        Args:
            session_id: User session identifier.
            messages: List of messages in the conversation.
            response: The final response from the LLM.
        """
        try:
            with self.database.session_scope() as session:
                serialized = [str(msg) for msg in messages]
                serialized.append(f"Assistant: {response}")
                full_conversation = "\n".join(serialized)
                self.conversation_repository.add_conversation(
                    session, session_id, full_conversation
                )
            logger.info(f"Saved conversation for session {session_id}")
        except Exception as e:
            error_msg = f"Failed to save conversation for session {session_id}: {type(e).__name__}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
