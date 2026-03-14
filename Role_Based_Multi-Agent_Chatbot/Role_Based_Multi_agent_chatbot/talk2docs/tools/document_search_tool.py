"""
Tool for searching documents using semantic search.
"""
from typing import List, Dict, Optional
from qdrant_client.models import Filter, FieldCondition, MatchValue
from agentic_student_assistant.talk2docs.tools.base_vector_tool import BaseVectorTool

class DocumentSearchTool(BaseVectorTool):
    """
    Tool for searching within the document vector store.
    """
    
    def search_documents(
        self,
        query: str,
        document_id: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, any]]:
        """
        Search for relevant document chunks.
        """
        if not self.client:
            return []
            
        # Generate query embedding
        query_embedding = self.embedder.encode(query).tolist()
        
        # Build filter if document_id specified
        query_filter = None
        if document_id:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            )
        
        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                query_filter=query_filter,
                limit=top_k
            )
            results = response.points
            
            chunks = []
            for result in results:
                payload = result.payload or {}
                # Langchain usually nests metadata or uses specific keys
                metadata = payload.get("metadata", payload)
                
                chunks.append({
                    "content": payload.get("page_content", payload.get("content", "")),
                    "score": result.score,
                    "document_id": metadata.get("document_id", "Unknown"),
                    "filename": metadata.get("source", metadata.get("filename", "Unknown")),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "metadata": metadata
                })
            return chunks
            
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return []
