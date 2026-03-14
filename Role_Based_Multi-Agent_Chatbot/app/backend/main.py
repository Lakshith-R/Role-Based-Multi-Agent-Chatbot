"""
FastAPI backend for Agentic Student Assistant.
Production-ready API with health checks, chat endpoints, and cache management.
"""
import time
from typing import Optional, List, Tuple, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi import File, UploadFile, Form
import sys
from pathlib import Path

# Add project root to sys.path
root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from dotenv import load_dotenv
load_dotenv()

# --- Pydantic Models ---

class ChatRequest(BaseModel):
    """Request model for the chat endpoint."""
    query: str = Field(..., description="The user's question or query")
    use_cache: bool = Field(default=True, description="Whether to use caching")
    chat_history: Optional[List[Tuple[str, str]]] = Field(
        default=[], description="List of (role, message) tuples"
    )

class ChatResponse(BaseModel):
    """Response model for the chat endpoint."""
    answer: str
    agent: str
    confidence: Optional[float] = None
    latency: float
    reasoning: Optional[str] = None

class HealthResponse(BaseModel):
    """Response model for the health check."""
    status: str = "healthy"
    version: str
    timestamp: float

class CacheStatsResponse(BaseModel):
    """Response model for cache statistics."""
    hits: int = 0
    misses: int = 0
    hit_rate: float = 0.0
    size: int = 0

class PaperRequest(BaseModel):
    """Request model for paper recommendations."""
    query: str = Field(..., description="Research topic or paper query")
    use_cache: bool = Field(default=True, description="Whether to use caching")

class JobRequest(BaseModel):
    """Request model for job search."""
    query: str = Field(..., description="Job search query (role, location, etc.)")
    use_cache: bool = Field(default=True, description="Whether to use caching")

class BookRequest(BaseModel):
    """Request model for book recommendations."""
    query: str = Field(..., description="Book topic or genre query")
    use_cache: bool = Field(default=True, description="Whether to use caching")

class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    document_id: str
    filename: str
    chunk_count: int
    message: str

class DocumentQueryRequest(BaseModel):
    """Request model for document Q&A."""
    query: str = Field(..., description="Question about uploaded documents")
    document_id: Optional[str] = Field(default=None, description="Specific document ID to query")
    use_cache: bool = Field(default=True, description="Whether to use caching")

class DocumentListResponse(BaseModel):
    """Response model for listing documents."""
    documents: List[dict]

class FeedbackRequest(BaseModel):
    """Request model for user feedback."""
    query: str = Field(..., description="User's query")
    response: str = Field(..., description="Agent's response")
    agent: str = Field(..., description="Agent that handled the query")
    rating: int = Field(..., description="1 for positive, -1 for negative")
    session_id: str = Field(..., description="User session ID")
    latency: Optional[float] = Field(default=None, description="Response latency")
    confidence: Optional[float] = Field(default=None, description="Router confidence")
    comment: Optional[str] = Field(default=None, description="Optional user comment")

class FeedbackResponse(BaseModel):
    """Response model for feedback submission."""
    message: str
    feedback_id: str

class FeedbackStatsResponse(BaseModel):
    """Response model for feedback statistics."""
    stats: Dict[str, Dict]


# --- Lifespan ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown events."""
    # Startup: Import heavy modules
    from agentic_student_assistant.core.orchestration.main_graph import app as agent_app
    from agentic_student_assistant.core.utils.cache import get_cache
    app.state.agent_app = agent_app
    app.state.cache = get_cache()
    yield
    # Shutdown: cleanup if needed


# --- FastAPI App ---

app = FastAPI(
    title="Agentic Student Assistant API",
    description="Intelligent AI agents for academic career support",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Endpoints ---

@app.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint for monitoring and load balancers."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=time.time()
    )


@app.post("/api/v1/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Main chat endpoint.
    Routes the query to the appropriate agent and returns the response.
    """
    cache = app.state.cache
    agent_app = app.state.agent_app

    # Check cache first
    if request.use_cache:
        cached_result = cache.get(request.query)
        if cached_result:
            return ChatResponse(
                answer=cached_result,
                agent="cached",
                confidence=1.0,
                latency=0.01,
                reasoning="Retrieved from semantic cache"
            )

    start_time = time.time()
    try:
        result = agent_app.invoke({
            "query": request.query,
            "chat_history": request.chat_history or []
        })

        agent_used = result.get("agent", "unknown")
        answer = result.get("result", "I couldn't find a specific answer.")
        confidence = result.get("confidence")
        reasoning = result.get("reasoning", "")
        latency = time.time() - start_time

        # Store in cache
        if request.use_cache and agent_used != "error":
            cache.set(request.query, answer, agent=agent_used)

        return ChatResponse(
            answer=answer,
            agent=agent_used,
            confidence=confidence,
            latency=latency,
            reasoning=reasoning
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/v1/cache/stats", response_model=CacheStatsResponse, tags=["Cache"])
async def get_cache_stats():
    """Get cache statistics."""
    stats = app.state.cache.get_stats()
    return CacheStatsResponse(**stats)


@app.delete("/api/v1/cache", tags=["Cache"])
async def clear_cache():
    """Clear the entire cache."""
    app.state.cache.clear()
    return {"message": "Cache cleared successfully"}


# --- Agent-Specific Endpoints ---

@app.post("/api/recommend-papers", response_model=ChatResponse, tags=["Agents"])
async def recommend_papers(request: PaperRequest):
    """
    Direct endpoint for paper recommendations.
    Routes directly to the Talk2Papers agent.
    """
    cache = app.state.cache
    agent_app = app.state.agent_app

    # Check cache first
    if request.use_cache:
        cached_result = cache.get(request.query)
        if cached_result:
            return ChatResponse(
                answer=cached_result,
                agent="cached",
                confidence=1.0,
                latency=0.01,
                reasoning="Retrieved from semantic cache"
            )

    start_time = time.time()
    try:
        result = agent_app.invoke({
            "query": request.query,
            "chat_history": []
        })

        agent_used = result.get("agent", "unknown")
        answer = result.get("result", "I couldn't find relevant papers.")
        confidence = result.get("confidence")
        reasoning = result.get("reasoning", "")
        latency = time.time() - start_time

        # Store in cache
        if request.use_cache and agent_used != "error":
            cache.set(request.query, answer, agent=agent_used)

        return ChatResponse(
            answer=answer,
            agent=agent_used,
            confidence=confidence,
            latency=latency,
            reasoning=reasoning
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/find-jobs", response_model=ChatResponse, tags=["Agents"])
async def find_jobs(request: JobRequest):
    """
    Direct endpoint for job search.
    Routes directly to the Talk2Jobs agent.
    """
    cache = app.state.cache
    agent_app = app.state.agent_app

    # Check cache first
    if request.use_cache:
        cached_result = cache.get(request.query)
        if cached_result:
            return ChatResponse(
                answer=cached_result,
                agent="cached",
                confidence=1.0,
                latency=0.01,
                reasoning="Retrieved from semantic cache"
            )

    start_time = time.time()
    try:
        result = agent_app.invoke({
            "query": request.query,
            "chat_history": []
        })

        agent_used = result.get("agent", "unknown")
        answer = result.get("result", "I couldn't find relevant jobs.")
        confidence = result.get("confidence")
        reasoning = result.get("reasoning", "")
        latency = time.time() - start_time

        # Store in cache
        if request.use_cache and agent_used != "error":
            cache.set(request.query, answer, agent=agent_used)

        return ChatResponse(
            answer=answer,
            agent=agent_used,
            confidence=confidence,
            latency=latency,
            reasoning=reasoning
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/recommend-books", response_model=ChatResponse, tags=["Agents"])
async def recommend_books(request: BookRequest):
    """
    Direct endpoint for book recommendations.
    Routes directly to the Talk2Books agent.
    """
    cache = app.state.cache
    agent_app = app.state.agent_app

    # Check cache first
    if request.use_cache:
        cached_result = cache.get(request.query)
        if cached_result:
            return ChatResponse(
                answer=cached_result,
                agent="cached",
                confidence=1.0,
                latency=0.01,
                reasoning="Retrieved from semantic cache"
            )

    start_time = time.time()
    try:
        result = agent_app.invoke({
            "query": request.query,
            "chat_history": []
        })

        agent_used = result.get("agent", "unknown")
        answer = result.get("result", "I couldn't find relevant books.")
        confidence = result.get("confidence")
        reasoning = result.get("reasoning", "")
        latency = time.time() - start_time

        # Store in cache
        if request.use_cache and agent_used != "error":
            cache.set(request.query, answer, agent=agent_used)

        return ChatResponse(
            answer=answer,
            agent=agent_used,
            confidence=confidence,
            latency=latency,
            reasoning=reasoning
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# --- Document Endpoints ---

@app.post("/api/upload-document", response_model=DocumentUploadResponse, tags=["Documents"])
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None)
):
    """
    Upload a document (PDF or TXT) for Q&A.
    """
    from agentic_student_assistant.talk2docs.tools.document_processor import DocumentProcessor
    from agentic_student_assistant.talk2docs.tools.qdrant_store import QdrantDocumentStore
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Process document
        processor = DocumentProcessor()
        processed = processor.process_upload(
            file_content=file_content,
            filename=file.filename,
            title=title
        )
        
        # Upload to Qdrant
        vector_store = QdrantDocumentStore()
        chunk_count = vector_store.upload_document(
            document_id=processed["document_id"],
            chunks=processed["chunks"],
            metadata=processed["metadata"]
        )
        
        return DocumentUploadResponse(
            document_id=processed["document_id"],
            filename=file.filename,
            chunk_count=chunk_count,
            message=f"Successfully uploaded {file.filename} with {chunk_count} chunks"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/query-document", response_model=ChatResponse, tags=["Documents"])
async def query_document(request: DocumentQueryRequest):
    """
    Ask questions about uploaded documents.
    """
    from agentic_student_assistant.talk2docs.agents.docs_agent import DocsRecommendAgent
    
    cache = app.state.cache
    
    # Check cache first
    if request.use_cache:
        cached_result = cache.get(request.query)
        if cached_result:
            return ChatResponse(
                answer=cached_result,
                agent="cached",
                confidence=1.0,
                latency=0.01,
                reasoning="Retrieved from semantic cache"
            )
    
    start_time = time.time()
    try:
        agent = DocsRecommendAgent()
        answer = agent.process(
            query=request.query,
            document_id=request.document_id
        )
        
        latency = time.time() - start_time
        
        # Store in cache
        if request.use_cache:
            cache.set(request.query, answer, agent="documents")
        
        return ChatResponse(
            answer=answer,
            agent="documents",
            confidence=1.0,
            latency=latency,
            reasoning="Document Q&A"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/list-documents", response_model=DocumentListResponse, tags=["Documents"])
async def list_documents():
    """
    List all uploaded documents.
    """
    from agentic_student_assistant.talk2docs.tools.qdrant_store import QdrantDocumentStore
    
    try:
        vector_store = QdrantDocumentStore()
        documents = vector_store.list_documents()
        return DocumentListResponse(documents=documents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.delete("/api/delete-document/{document_id}", tags=["Documents"])
async def delete_document(document_id: str):
    """
    Delete an uploaded document.
    """
    from agentic_student_assistant.talk2docs.tools.qdrant_store import QdrantDocumentStore
    from agentic_student_assistant.talk2docs.tools.document_processor import DocumentProcessor
    
    try:
        vector_store = QdrantDocumentStore()
        vector_store.delete_document(document_id)
        
        # Also delete the file from disk
        processor = DocumentProcessor()
        # Note: We'd need to store file_path in metadata to delete from disk
        # For now, just delete from Qdrant
        
        return {"message": f"Successfully deleted document {document_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# --- Feedback Endpoints ---

@app.post("/api/feedback", response_model=FeedbackResponse, tags=["Feedback"])
async def submit_feedback(request: FeedbackRequest):
    """
    Submit user feedback (thumbs up/down) for an agent response.
    Also emits reward to Agent Lightning.
    """
    from agentic_student_assistant.core.utils.feedback_store import get_feedback_store
    
    try:
        import agentlightning as agl
        AGL_AVAILABLE = True
    except ImportError:
        AGL_AVAILABLE = False
    
    try:
        # Validate rating
        if request.rating not in [1, -1]:
            raise HTTPException(status_code=400, detail="Rating must be 1 or -1")
        
        # 1. Store in Redis (for analytics)
        feedback_store = get_feedback_store()
        feedback_id = feedback_store.add_feedback(
            query=request.query,
            response=request.response,
            agent=request.agent,
            rating=request.rating,
            session_id=request.session_id,
            latency=request.latency,
            confidence=request.confidence,
            comment=request.comment
        )
        
        # 2. Emit Reward to Agent Lightning (for RL training)
        if AGL_AVAILABLE:
            try:
                agl.emit_reward(
                    value=float(request.rating),  # 1.0 or -1.0
                    agent_name=request.agent,
                    meta={
                        "query": request.query,
                        "session_id": request.session_id,
                        "feedback_id": feedback_id
                    }
                )
                print(f"⚡ [AGL] Emitted reward {request.rating} for agent {request.agent}")
            except Exception as e:
                print(f"⚠️ [AGL] Failed to emit reward: {e}")
        
        return FeedbackResponse(
            message="Feedback recorded successfully",
            feedback_id=feedback_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/feedback/stats", response_model=FeedbackStatsResponse, tags=["Feedback"])
async def get_feedback_stats():
    """
    Get feedback statistics for all agents.
    """
    from agentic_student_assistant.core.utils.feedback_store import get_feedback_store
    
    try:
        feedback_store = get_feedback_store()
        stats = feedback_store.get_stats()
        return FeedbackStatsResponse(stats=stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# --- Run with Uvicorn ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
