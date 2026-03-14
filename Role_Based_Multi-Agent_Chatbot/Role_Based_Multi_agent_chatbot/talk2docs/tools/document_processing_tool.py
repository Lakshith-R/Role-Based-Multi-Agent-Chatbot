"""
Tool for processing uploaded files (extraction and chunking).
"""
import uuid
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path

from agentic_student_assistant.core.utils.parse_pdf import extract_text_from_pdf
from agentic_student_assistant.core.utils.chunker import chunk_text

class DocumentProcessingTool:
    """Processes uploaded documents and prepares them for vector storage."""
    
    SUPPORTED_EXTENSIONS = {".pdf", ".txt"}
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    
    def __init__(self, upload_dir: str = "uploads"):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(exist_ok=True, parents=True)
    
    def process_upload(
        self,
        file_content: bytes,
        filename: str,
        title: Optional[str] = None
    ) -> Dict[str, any]:
        """Process an uploaded file."""
        
        # Validate file
        file_ext = Path(filename).suffix.lower()
        if file_ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        if len(file_content) > self.MAX_FILE_SIZE:
            raise ValueError(f"File too large. Max: 10MB")
        
        # Generate ID
        document_id = str(uuid.uuid4())
        
        # Save file
        file_path = self.upload_dir / f"{document_id}{file_ext}"
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        # Extract text
        try:
            if file_ext == ".pdf":
                text = extract_text_from_pdf(str(file_path))
            elif file_ext == ".txt":
                text = file_content.decode("utf-8", errors="ignore")
            else:
                text = ""
        except Exception as e:
            print(f"❌ Extraction error: {e}")
            text = "" # Graceful failure
            
        if not text:
             # Basic fallback or warning
             print(f"⚠️ Warning: No text extracted from {filename}")
        
        # Chunk text
        chunks = chunk_text(
            text=text,
            source=filename,
            chunk_size=500,
            chunk_overlap=50
        )
        
        # Metadata
        metadata = {
            "document_id": document_id,
            "filename": filename,
            "title": title or filename,
            "upload_time": datetime.utcnow().isoformat(),
            "file_path": str(file_path),
            "file_size": len(file_content),
            "chunk_count": len(chunks)
        }
        
        return {
            "document_id": document_id,
            "chunks": chunks,
            "metadata": metadata
        }
    
    def delete_file(self, file_path: str):
        """Delete local file."""
        path = Path(file_path)
        if path.exists():
            path.unlink()
            print(f"✅ Deleted local file: {file_path}")
