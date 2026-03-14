"""
Tool for managing documents (CRUD) in the vector store.
"""
import uuid
from typing import List, Dict
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
from agentic_student_assistant.talk2docs.tools.base_vector_tool import BaseVectorTool

class DocumentManagementTool(BaseVectorTool):
    """
    Tool for Uploading, Deleting, and Listing documents.
    """
    
    def upload_document(
        self,
        document_id: str,
        chunks: List[Dict[str, any]],
        metadata: Dict[str, any]
    ) -> int:
        """
        Upload document chunks to Qdrant.
        """
        if not self.client: 
            return 0
            
        points = []
        
        for chunk in chunks:
            chunk_id = str(uuid.uuid4())
            content = chunk["content"]
            chunk_metadata = chunk.get("metadata", {})
            
            # Generate embedding
            embedding = self.embedder.encode(content).tolist()
            
            # Combine payload
            payload = {
                "document_id": document_id,
                "content": content,
                "chunk_index": chunk_metadata.get("chunk_index", 0),
                **metadata,
                **chunk_metadata
            }
            
            points.append(PointStruct(
                id=chunk_id,
                vector=embedding,
                payload=payload
            ))
        
        # Upload
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            print(f"✅ Uploaded {len(points)} chunks for document: {document_id}")
            return len(points)
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            return 0

    def delete_document(self, document_id: str) -> int:
        """Delete all chunks for a document."""
        if not self.client:
            return 0
            
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                )
            )
            print(f"✅ Deleted document: {document_id}")
            return 1
        except Exception as e:
            print(f"❌ Deletion failed: {e}")
            return 0

    def list_documents(self) -> List[Dict[str, any]]:
        """List all uploaded documents."""
        if not self.client:
            return []
            
        documents = {}
        try:
            offset = None
            while True:
                results, offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                
                if not results:
                    break
                
                for point in results:
                    doc_id = point.payload.get("document_id")
                    if not doc_id:
                        continue
                        
                    if doc_id not in documents:
                        documents[doc_id] = {
                            "document_id": doc_id,
                            "filename": point.payload.get("filename", "Unknown"),
                            "title": point.payload.get("title", "Untitled"),
                            "upload_time": point.payload.get("upload_time", "Unknown"),
                            "chunk_count": 0
                        }
                    documents[doc_id]["chunk_count"] += 1
                
                if offset is None:
                    break
        except Exception as e:
            print(f"⚠️ Error listing documents: {e}")
            return []
        
        return list(documents.values())
