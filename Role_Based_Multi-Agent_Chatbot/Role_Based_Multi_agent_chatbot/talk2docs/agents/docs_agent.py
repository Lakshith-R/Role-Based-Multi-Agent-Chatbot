"""
Document Q&A agent with persistent memory using Qdrant.
"""
import yaml
from pathlib import Path
from typing import List, Tuple, Optional

from agentic_student_assistant.core.base.base_agent import BaseAgent
from agentic_student_assistant.core.utils.llm_factory import LLMFactory
from agentic_student_assistant.core.utils.logging_manager import LoggingManager
from agentic_student_assistant.core.utils.config_loader import get_config
from agentic_student_assistant.talk2docs.tools.document_search_tool import DocumentSearchTool
from agentic_student_assistant.talk2docs.tools.document_management_tool import DocumentManagementTool
try:
    import agentlightning as agl
    AGL_AVAILABLE = True
except ImportError:
    AGL_AVAILABLE = False


class DocsRecommendAgent(BaseAgent):
    """Agent for answering questions about uploaded documents."""
    
    def __init__(self):
        """Initialize document Q&A agent."""
        config = get_config()
        super().__init__(
            agent_config=config,
            agent_name="DocsRecommendAgent",
            llm=LLMFactory.create_llm(config.models)
        )
        
        # Initialize tools
        self.search_tool = DocumentSearchTool()
        self.management_tool = DocumentManagementTool()
        
        # Load prompts
        prompts_path = Path(__file__).parent.parent / "configs" / "prompts.yaml"
        with open(prompts_path, "r", encoding="utf-8") as f:
            self.prompts = yaml.safe_load(f)
    
    def process(
        self,
        query: str,
        document_id: Optional[str] = None,
        chat_history: Optional[List[Tuple[str, str]]] = None,
        **kwargs
    ) -> str:
        """
        Answer questions about uploaded documents.
        
        Args:
            query: User's question
            document_id: Optional specific document to query
            chat_history: Previous conversation for context
            **kwargs: Additional arguments
        
        Returns:
            Answer to the user's question
        """
        # Check if there are any documents
        documents = self.management_tool.list_documents()
        if not documents:
            return (
                "❌ **No documents uploaded yet.**\n\n"
                "Please upload a document first using the `/api/upload-document` endpoint."
            )
        
        # Search for relevant chunks
        top_k = kwargs.get("top_k", 10)  # Increased from 5 to 10 for better coverage
        chunks = self.search_tool.search_documents(
            query=query,
            document_id=document_id,
            top_k=top_k
        )
        
        if not chunks:
            return (
                "❌ **No relevant information found.**\n\n"
                "The uploaded documents don't seem to contain information about your query."
            )
        
        # Format chunks for prompt
        chunks_text = self._format_chunks(chunks)
        
        # Format chat history
        history_text = self._format_history(chat_history or [])
        
        # Build prompt
        prompt = self.prompts["document_qa_prompt"].format(
            query=query,
            document_chunks=chunks_text,
            chat_history=history_text
        )
        
        # Agent Lightning Instrumentation
        if AGL_AVAILABLE:
            with agl.trace(name="docs_agent_process", agent_name="docs_agent") as trace:
                agl.emit_prompt(prompt, tags=["docs", "qa"])
                
                response = self.llm.predict(prompt)
                
                agl.emit_llm_response(response)
        else:
            response = self.llm.predict(prompt)
            
        # Add source information
        sources = self._format_sources(chunks)
        final_response = f"{response}\n\n{sources}"
        
        return final_response
    
    def _format_chunks(self, chunks: List[dict]) -> str:
        """Format document chunks for the prompt."""
        formatted = []
        for i, chunk in enumerate(chunks, 1):
            formatted.append(
                f"**Chunk {i}** (from {chunk['filename']}, relevance: {chunk['score']:.2f}):\n"
                f"{chunk['content']}\n"
            )
        return "\n".join(formatted)
    
    def _format_history(self, chat_history: List[Tuple[str, str]]) -> str:
        """Format chat history for context."""
        if not chat_history:
            return "No previous conversation."
        
        formatted = []
        for role, message in chat_history[-3:]:  # Last 3 exchanges
            formatted.append(f"{role}: {message}")
        return "\n".join(formatted)
    
    def _format_sources(self, chunks: List[dict]) -> str:
        """Format source citations."""
        unique_docs = {}
        for chunk in chunks:
            doc_id = chunk['document_id']
            if doc_id not in unique_docs:
                unique_docs[doc_id] = chunk['filename']
        
        if not unique_docs:
            return ""
        
        sources_list = [f"- {filename}" for filename in unique_docs.values()]
        return "📚 **Sources:**\n" + "\n".join(sources_list)
    
    def summarize_document(self, document_id: str) -> str:
        """
        Generate a summary of a document.
        
        Args:
            document_id: Document to summarize
        
        Returns:
            Document summary
        """
        # Get all chunks for the document
        chunks = self.search_tool.search_documents(
            query="summary overview main points",  # Generic query to get diverse chunks
            document_id=document_id,
            top_k=10
        )
        
        if not chunks:
            return "❌ Document not found."
        
        # Combine chunks
        content = "\n\n".join([chunk['content'] for chunk in chunks])
        
        # Build prompt
        prompt = self.prompts["document_summary_prompt"].format(
            document_content=content
        )
        
        # Get summary
        summary = self.llm.predict(prompt)
        
        return summary


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    agent = DocsRecommendAgent()
    
    # Test query
    test_query = "What is this document about?"
    print(f"Query: {test_query}\n")
    
    result = agent.process(test_query)
    print(result)
