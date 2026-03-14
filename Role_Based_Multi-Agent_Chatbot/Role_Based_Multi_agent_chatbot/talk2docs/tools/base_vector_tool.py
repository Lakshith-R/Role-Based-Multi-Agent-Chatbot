"""
Base vector tool providing shared Qdrant connection and embedding model logic.
"""
import os
import time
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer

class BaseVectorTool:
    """
    Base tool for interaction with Qdrant Vector Database.
    Handles shared client connectivity and model loading with caching.
    """
    
    _model = None  # Class-level cache for embedding model
    _client = None # Class-level cache for Qdrant client
    
    def __init__(self, collection_name: str = "user_documents"):
        """Initialize the base vector tool."""
        self.collection_name = collection_name
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
        
        # Initialize Client (Singleton-ish)
        if BaseVectorTool._client is None:
            BaseVectorTool._client = self._connect_client()
        self.client = BaseVectorTool._client
        
        # Initialize Model (Singleton-ish)
        if BaseVectorTool._model is None:
            print("⏳ Loading embedding model (base tool)...")
            BaseVectorTool._model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedder = BaseVectorTool._model
        
        # Ensure collection exists
        self._ensure_collection()

    def _connect_client(self) -> QdrantClient:
        """Connect to Qdrant securely."""
        try:
            return QdrantClient(
                url=os.getenv("QDRANT_URL"),
                api_key=os.getenv("QDRANT_API_KEY")
            )
        except Exception as e:
            print(f"⚠️ Warning: Qdrant initial connection failed: {e}")
            return None

    def _ensure_collection(self):
        """Ensure the collection and required indices exist."""
        if not self.client:
            return

        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                print(f"✅ Created Qdrant collection: {self.collection_name}")
            
            # Always try to create the index to ensure it exists for legacy collections
            # Qdrant handles idempotency or we catch the error if it exists
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="document_id",
                    field_schema="keyword"
                )
                print(f"✅ Verified/Created index for: {self.collection_name}")
            except Exception as e:
                # Ignore if it implies index already exists or other non-critical issue
                print(f"ℹ️ Index creation check: {e}")
        except Exception as e:
            print(f"⚠️ Warning: Could not ensure collection exists: {e}")
