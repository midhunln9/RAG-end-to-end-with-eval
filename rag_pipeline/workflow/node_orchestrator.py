"""
LangGraph node implementations for RAG workflow.

These nodes are used by the LangGraph state machine to implement
individual steps of the RAG pipeline.
"""

from rag_pipeline.workflow.state import AgentState
from rag_pipeline.workflow.service import RAGService
from langchain_core.messages import HumanMessage, AIMessage


class Nodes:
    """
    Graph node implementations for the RAG workflow.
    
    Each method corresponds to a node in the LangGraph state machine.
    """

    def __init__(self, service: RAGService):
        """
        Initialize nodes with RAG service.
        
        Args:
            service: RAGService instance that handles business logic.
        """
        self.service = service

    def query_rewriter(self, state: AgentState) -> dict:
        """
        Rewrite user query for better retrieval.
        
        Args:
            state: Current workflow state.
            
        Returns:
            Updated state with rewritten_query and conversation_history.
        """
        
        query = state["query"]
        rewritten_query = self.service.rewrite_query(query)
        
        # Build conversation history
        conversation_history = [
            HumanMessage(content=query),
            AIMessage(content=rewritten_query),
        ]
        
        return {
            "rewritten_query": rewritten_query,
            "conversation_history": conversation_history,
        }

    def fetch_documents(self, state: AgentState) -> dict:
        """
        Retrieve relevant documents from vector store.
        
        Args:
            state: Current workflow state.
            
        Returns:
            Updated state with retrieved_documents.
        """
        query = state["rewritten_query"]
        documents = self.service.retrieve_documents(query)
        return {"retrieved_documents": documents}

    def generate_summary_last_5_messages(self, state: AgentState) -> dict:
        """
        Generate summary of recent conversation history.
        
        Args:
            state: Current workflow state.
            
        Returns:
            Updated state with summary_before_last_five_messages.
        """
        session_id = state["session_id"]
        summary = self.service.generate_context_summary(session_id)
        return {"summary_before_last_five_messages": summary}

    def llm_call(self, state: AgentState) -> dict:
        """
        Generate final response using LLM with context.
        
        Args:
            state: Current workflow state.
            
        Returns:
            Updated state with response.
        """
        response = self.service.generate_response(
            query=state["rewritten_query"],
            documents=state["retrieved_documents"],
            context_summary=state["summary_before_last_five_messages"],
        )
        return {"response": response}

    def add_conversation_to_db(self, state: AgentState) -> dict:
        """
        Save conversation to database for history tracking.
        
        Args:
            state: Current workflow state.
            
        Returns:
            Empty dict (no state updates).
        """
        self.service.save_conversation(
            session_id=state["session_id"],
            messages=state["conversation_history"],
            response=state["response"],
        )
        return {}