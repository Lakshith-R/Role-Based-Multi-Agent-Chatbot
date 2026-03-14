# Agentic Student Assistant

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A modular AI assistant designed to help students and researchers with daily academic tasks. It uses multiple specialized agents to handle specific queries like finding research papers, searching for jobs, recommending books, or answering questions from your own documents.

## 🤖 What It Does

The system routes your query to the most relevant specialist agent:

*   **Talk2Jobs**: Searches for job postings globally (LinkedIn, Glassdoor via Google Jobs). You can specify locations (e.g., "Data Science jobs in Berlin").
*   **Talk2Papers**: Searches academic databases (ArXiv, Semantic Scholar, CORE) for research papers and provides summaries.
*   **Talk2Books**: Recommends books based on your interests using Open Library and Google Books.
*   **Talk2Docs**: Allows you to upload your own PDF or TXT files and ask questions about them.

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) (recommended for fast dependency management) or `pip`

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Agentic_Student_Assistant
cd Agentic_Student_Assistant
```

### 2. Install Dependencies
Using `uv` (faster):
```bash
uv sync
```
Or using standard `pip`:
```bash
pip install -r requirements.txt
```

### 3. Configure API Keys
Create a `.env` file in the root directory. You will need at least an OpenAI key (or Groq for a free alternative).

```bash
# Required
OPENAI_API_KEY=your_openai_key
# OR
GROQ_API_KEY=your_groq_key

# Optional (for specific features)
SERPAPI_API_KEY=your_serpapi_key       # For Job Search stability
SEMANTIC_SCHOLAR_API_KEY=your_ss_key   # For higher rate limits on paper search
REDIS_HOST=localhost                   # For caching (optional, defaults to local memory if failed)
```

### 4. Run the Application
Start the Streamlit dashboard:
```bash
streamlit run app/frontend/streamlit_app.py
```
*The app will open in your browser at `http://localhost:8501`*

---

## 🏗️ Architecture

The project is built using:
- **LangGraph**: Manages the workflow and routing between agents.
- **FastAPI**: Provides the backend API endpoints.
- **Streamlit**: Renders the frontend user interface.
- **Qdrant**: Vector database for storing and searching your uploaded documents.
- **Redis & Local Cache**: Speeds up repeated queries to save API costs.

## 🛠️ Advanced Usage (API)

If you prefer to use the backend API directly (e.g., for building a different frontend):

1. **Start the Backend Server**:
   ```bash
   uvicorn app.backend.main:app --reload
   ```

2. **Access Documentation**:
   Go to `http://localhost:8000/docs` to see the interactive Swagger UI.

3. **Example API Call (Document Upload)**:
   ```bash
   curl -X POST "http://localhost:8000/api/upload-document" \
     -F "file=@lecture_notes.pdf"
   ```

## 📊 Project Status
- **Core Agents**: Active & Tested
- **UI**: Functional (Streamlit)
- **Memory**: Persistent for Documents (Qdrant)
- **Status**: Stable Beta

---
*Built for educational purposes.*
