"""
Streamlit application for Agentic Student Assistant.
Redesigned with a professional two-column layout.
"""
import os
import json
import time
import uuid
from typing import List, Tuple, Dict
import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
import sys
from pathlib import Path

# Add project root to sys.path
root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

# Utilities
from agentic_student_assistant.core.utils.parse_pdf import parse_single_pdf
from agentic_student_assistant.core.utils.chunker import chunk_text
from agentic_student_assistant.core.utils.logging_manager import LoggingManager
from agentic_student_assistant.core.utils.cache import get_cache
from agentic_student_assistant.core.orchestration.main_graph import app

# UI Utils
from app.frontend.utils import apply_custom_css

load_dotenv()

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Agentic Student Assistant", 
    layout="wide", 
    page_icon="🎓"
)
apply_custom_css()

# Caching for performance
@st.cache_resource
def get_doc_manager():
    try:
        from agentic_student_assistant.talk2docs.tools.document_management_tool import DocumentManagementTool
        return DocumentManagementTool()
    except Exception as e:
        print(f"❌ Failed to initialize Document Manager: {e}")
        return None

@st.cache_data(ttl=60) # Cache for 1 minute
def cached_list_documents():
    manager = get_doc_manager()
    if manager:
        return manager.list_documents()
    return []

from langsmith import Client
from streamlit_feedback import streamlit_feedback

# Optional: initialize LangSmith Client (relies on LANGSMITH_API_KEY in .env)
ls_client = Client()

def send_feedback(feedback_dict, run_id):
    """Callback to send feedback directly to LangSmith."""
    if not feedback_dict or not run_id:
        return
    
    score = feedback_dict.get("score")
    text = feedback_dict.get("text", "")
    
    # Map thumbs up/down to numeric scores
    if score == "👍":
        numeric_score = 1.0
    elif score == "👎":
        numeric_score = 0.0
    else:
        # For faces emoji mapping if changed later
        numeric_score = None
        
    try:
        ls_client.create_feedback(
            run_id=run_id,
            key="user_score",
            score=numeric_score,
            comment=text
        )
        st.toast("Feedback sent to LangSmith!", icon="✅")
    except Exception as e:
        print(f"Error sending feedback to LangSmith: {e}")
        st.toast("Could not send feedback. Check LangSmith API key.", icon="⚠️")

if "logger" not in st.session_state:
    st.session_state.logger = LoggingManager(
        enable_file=True,
        enable_gsheets=True,
        enable_console=False
    )

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_documents" not in st.session_state:
    st.session_state.uploaded_documents = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "last_response_data" not in st.session_state:
    st.session_state.last_response_data = None

# ---------------- Top Heading ----------------
st.markdown(
    """
    <div style='text-align: center; padding: 0rem 0 1rem 0;'>
        <h1 style='margin: 0; font-weight: 800; color: #ffffff; font-size: 2.5rem; letter-spacing: -0.02em;'>
            🎓 Agentic Student Assistant
        </h1>
        <p style='margin: 5px 0 0 0; color: #808495; font-size: 1.1rem;'>
            Your Intelligent Partner for Academic Success
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- Two-Column Layout ----------------
col1, col2 = st.columns([3, 7])

# ========== LEFT COLUMN: Control Panel ==========
with col1:
    # Performance Settings Container
    with st.container(border=True):
        st.markdown("#### ⚙️ Settings")
        use_cache = st.toggle(
            "⚡ Engine Caching", 
            value=True,
            help="Cache responses"
        )
        
        # Compact Cache Stats
        if use_cache:
            cache = get_cache()
            cache_stats = cache.get_stats()
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Hits", cache_stats.get('hits', 0), label_visibility="visible")
            with c2:
                hit_rate = cache_stats.get('hit_rate', 0)
                st.metric("Rate", f"{hit_rate:.0%}")
            with c3:
                st.metric("Miss", cache_stats.get('misses', 0))
            
            if st.button("Clear", use_container_width=True, key="clear_cache"):
                cache.clear()
                st.success("✓ Cleared!")
                time.sleep(0.3)
                st.rerun()
    
    # Compact Session Stats
    with st.container(border=True):
        st.markdown("#### 📊 Session")
        user_msgs = len([m for m in st.session_state.chat_history if m[0] == "user"])
        st.metric("Questions", user_msgs, label_visibility="visible")
        
        # DEBUG: Show last_run_id in sidebar
        run_id = st.session_state.get("last_run_id")
        if run_id:
            st.caption(f"Last Trace: `{run_id[:8]}...`")
    
    # # Feedback Stats
    # with st.container(border=True):
    #     st.markdown("#### 📈 Feedback")
    #     try:
    #         import requests
    #         response = requests.get("http://localhost:8000/api/feedback/stats", timeout=2)
    #         if response.status_code == 200:
    #             stats = response.json()["stats"]
    #             overall = stats.get("overall", {})
    #             if overall.get("total", 0) > 0:
    #                 st.metric(
    #                     "Satisfaction",
    #                     f"{overall.get('satisfaction_rate', 0):.0f}%",
    #                     delta=f"{overall.get('positive', 0)} 👍"
    #                 )
    #                 st.caption(f"{overall.get('total', 0)} total ratings")
    #             else:
    #                 st.info("No feedback yet")
    #     except Exception:
    #         st.caption("Stats unavailable")
    
    # Document Upload Section
    with st.container(border=True):
        st.markdown("#### 📄 Documents")
        # Document Manager
        doc_manager = get_doc_manager()
        
        # File Uploader
        uploaded_file = st.file_uploader(
            "Drag and drop file here", 
            type=["pdf", "txt"],
            help="Limit 200MB per file • PDF, TXT"
        )
        
        if uploaded_file and doc_manager:
            if st.button("📥 Upload", type="primary", use_container_width=True):
                with st.spinner("Processing document..."):
                    try:
                        # Initialize Processor
                        from agentic_student_assistant.talk2docs.tools.document_processing_tool import DocumentProcessingTool
                        processor = DocumentProcessingTool()
                        
                        # Process
                        bytes_data = uploaded_file.getvalue()
                        doc_data = processor.process_upload(
                            file_content=bytes_data,
                            filename=uploaded_file.name
                        )
                        
                        # Upload to Qdrant using Manager
                        count = doc_manager.upload_document(
                            document_id=doc_data["document_id"],
                            chunks=doc_data["chunks"],
                            metadata=doc_data["metadata"]
                        )
                        
                        st.success(f"✓ Uploaded {uploaded_file.name}!")
                        st.toast(f"📄 {count} chunks indexed", icon="✅")
                        time.sleep(1)
                        # Clear cache to show new doc immediately
                        cached_list_documents.clear()
                        st.rerun()
                    except Exception as e:
                         st.error(f"Upload failed: {str(e)}")
        
        # List uploaded documents
        try:
            docs = cached_list_documents()
            
            if docs:
                st.markdown(f"**{len(docs)} document(s)**")
                for doc in docs[:5]:  # Show max 5
                    col_doc, col_del = st.columns([4, 1])
                    with col_doc:
                        # Format timestamp (e.g. 2023-10-25T14:30 -> 14:30)
                        ts = doc.get('upload_time', '')
                        ts_display = ts.split('T')[1][:5] if 'T' in ts else ''
                        
                        # Show filename + unique info
                        label = f"📄 {doc['filename'][:15]}..." if len(doc['filename']) > 15 else f"📄 {doc['filename']}"
                        st.caption(f"{label} ({ts_display})")
                        
                    with col_del:
                        if st.button("🗑️", key=f"del_{doc['document_id']}", help=f"Delete {doc['filename']}"):
                            try:
                                if doc_manager:
                                    doc_manager.delete_document(doc['document_id'])
                                    st.success("Deleted!")
                                    time.sleep(0.5)
                                    cached_list_documents.clear() # Clear cache
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
            else:
                st.info("No documents uploaded yet")
        except Exception as e:
            st.warning("⚠️ **Network Issue:** Could not connect to the document database or AI model.")
            st.caption(f"Error details: {str(e)}")
            st.info("💡 **Tip:** This usually happens if your internet is blocked or you can't reach Hugging Face/Qdrant Cloud.")
    
    # Chat Input - Always at bottom
    user_query = st.chat_input("Type your question here...")


# ========== RIGHT COLUMN: Chat Area ==========
with col2:
    with st.container(border=True, height=800):
        # Display chat history
        for i, (role, message) in enumerate(st.session_state.chat_history):
            with st.chat_message(role, avatar="🤖" if role == "assistant" else "👤"):
                st.markdown(message)
                
                # Feedback widget for the *latest* assistant message only
                if role == "assistant" and i == len(st.session_state.chat_history) - 1:
                    # We need the run_id from the last response to attach feedback to the right trace
                    run_id = st.session_state.get("last_run_id")
                    
                    if run_id:
                        feedback = streamlit_feedback(
                            feedback_type="thumbs",
                            optional_text_label="[Optional] Why did you give this rating? What went wrong?",
                            on_submit=send_feedback,
                            kwargs={"run_id": run_id},
                            key=f"fb_{run_id}"
                        )
        
        # Welcome message for new users
        if not st.session_state.chat_history:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown("""
                Hello! I'm your **SRH Smart Assistant**. I can help you with:
                
                1. **Job Market & Career Opportunities** - Find job trends, career paths, and improve employability
                2. **Research Papers & Academic Articles** - Discover, understand, and create research papers
                3. **Book Recommendations** - Get suggestions on books related to your field of study
                4. **Document Q&A** - Upload your documents (PDF/TXT) and ask questions about them
                
                Feel free to ask me anything!
                """)
        
        # Process user input
        if user_query:
            # Add user message
            st.session_state.chat_history.append(("user", user_query))
            with st.chat_message("user", avatar="👤"):
                st.markdown(user_query)
            
            # Check cache
            cached_result = None
            if use_cache:
                cache = get_cache()
                cached_result = cache.get(user_query)
            
            if cached_result:
                answer = cached_result
                agent_used = "cached"
                confidence = 1.0
                reasoning = "Retrieved from semantic cache"
                latency = 0.01
                st.toast("⚡ Retrieved from Cache", icon="📦")
            else:
                start_time = time.time()
                try:
                    with st.spinner("Analyzing your request..."):
                        # We use .invoke(), but to get the run_id we can wrap it or inspect the output
                        # In newer LangChain, invoke() doesn't directly return run_id unless we use a callback
                        # For simplicity, we can use the tracer if enabled, or just generate a manual UUID 
                        # if we aren't using deep tracing. Let's use get_callbacks to capture run_id
                        
                        from langchain_core.tracers.context import collect_runs
                        with collect_runs() as cb:
                            result = app.invoke({
                                "query": user_query,
                                "chat_history": st.session_state.chat_history
                            })
                            # Capture the root run ID
                            if cb.traced_runs:
                                st.session_state.last_run_id = str(cb.traced_runs[0].id)
                            else:
                                st.session_state.last_run_id = str(uuid.uuid4())
                    
                    agent_used = result.get("agent", "unknown")
                    confidence = result.get("confidence")
                    reasoning = result.get("reasoning", "")
                    answer = result.get("result", "I couldn't find a specific answer.")
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    answer = "I'm sorry, I encountered an error."
                    agent_used = "error"
                    confidence = 0
                    reasoning = str(e)
                
                latency = time.time() - start_time
                
                if use_cache and agent_used != "error":
                    cache.set(user_query, answer, agent=agent_used)
            
            # Display response
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(answer)
                
                # Store response data for feedback
                st.session_state.last_response_data = {
                    "query": user_query,
                    "response": answer,
                    "agent": agent_used,
                    "latency": latency,
                    "confidence": confidence
                }
            
            # Add to history
            st.session_state.chat_history.append(("assistant", answer))
            
            # Log interaction
            st.session_state.logger.log_interaction(
                query=user_query,
                agent=agent_used,
                result=answer,
                latency=latency,
                is_fallback=(agent_used == "fallback"),
                confidence=confidence,
                reasoning=reasoning
            )
            
            # Use rerun to update chat history immediately and show feedback buttons for new message
            st.rerun()




