# Role-Based-Multi-Agent-Chatbot

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Welcome to the **Agentic Student Assistant**! This is a modular, AI-powered system designed specifically to help students, researchers, and lifelong learners with their daily academic and professional tasks. 

By using "agentic AI" (AI agents that can make decisions and use tools), this assistant intelligently routes your questions to specialized agents depending on what you need—whether that's searching for academic papers, looking for a job, getting book recommendations, or asking questions about your personal documents.

---

## ✨ Features

The system acts like a master orchestrator. You ask a question, and it routes it to the most relevant specialist agent:

1. 🎓 **Talk2Papers (Research Agent)**
   - Searches major academic databases (like ArXiv, Semantic Scholar, CORE).
   - Finds research papers relevant to your specific query and provides structured summaries.
2. 💼 **Talk2Jobs (Career Agent)**
   - Uses Google Jobs (via SerpAPI) to search for active job postings on platforms like LinkedIn and Glassdoor.
   - You can specify locations and roles (e.g., "Find me Junior Python Developer roles in Berlin").
3. 📚 **Talk2Books (Library Agent)**
   - Connects to Open Library and Google Books to recommend literature based on your interests or specific topics.
4. 📄 **Talk2Docs (Personal Document Agent)**
   - Allows you to upload your own files (PDFs or TXTs).
   - "Reads" them using a Vector Database (Qdrant) and allows you to chat with your documents.

---

## 🚀 Getting Started

These instructions will help you set up and run the system on your local machine. You don't need to be an AI expert to get this running!

### Prerequisites
Before you start, make sure you have:
- **Python 3.10 or higher** installed on your computer.
- **Git** (to clone this repository).
- *(Recommended)* **[uv](https://github.com/astral-sh/uv)** - An extremely fast Python package and project manager. Standard `pip` works too!

### 1. Clone the Repository
Open your terminal (Command Prompt or PowerShell on Windows, Terminal on macOS/Linux) and run:
```bash
git clone https://github.com/hsrak/Agentic_Student_Assistant.git
cd Agentic_Student_Assistant
```

### 2. Install Dependencies
You need to install the required Python libraries.

**Using `uv` (Recommended and much faster):**
```bash
uv sync
```

**Using standard `pip`:**
```bash
pip install -r requirements.txt
```

### 3. Set Up API Keys (.env File)
The AI needs "API keys" to talk to services like OpenAI or Search engines. 
1. Create a file named exactly `.env` in the root folder (where this README is).
2. Open it in a text editor and add your keys:

```ini
# --- REQUIRED ---
# You need at least ONE Language Model key (OpenAI or Groq)
OPENAI_API_KEY=your_openai_api_key_here
# OR (for a free/fast alternative)
GROQ_API_KEY=your_groq_api_key_here

# --- OPTIONAL (For specific functions) ---
SERPAPI_API_KEY=your_serpapi_key       # Highly recommended for robust Job Searches
SEMANTIC_SCHOLAR_API_KEY=your_ss_key   # For better/faster Research Paper searches
REDIS_HOST=localhost                   # To make the app run faster using cache
```

---

## 🏃‍♂️ How to Run the App

The project consists of two parts: a **Backend Server** (which does the heavy lifting) and a **Frontend User Interface** (the visual dashboard you interact with). 

### Option 1: The Easy Way (Streamlit UI)
To launch the beautiful user interface, just run:
```bash
streamlit run app/frontend/streamlit_app.py
```
*Your browser will automatically open a new tab at `http://localhost:8501`. You can chat, upload documents, and interact with the AI directly from there!*

### Option 2: Advanced Usage / API Only (FastAPI)
If you want to use the assistant as an API (for example, to build your own custom web, mobile app, or integrate it elsewhere):
```bash
uvicorn app.backend.main:app --reload
```
*You can view the interactive developer documentation by visiting `http://localhost:8000/docs` in your browser.*

---

## 📁 Project Structure (What does every folder do?)

Here is a simple breakdown of the files in this project so you can easily understand where everything lives:

```text
📂 Agentic_Student_Assistant/
├── 📂 agentic_student_assistant/   # 🧠 Core Brain of the AI
│   ├── 📂 core/                    # Contains orchestrator (LangGraph) that decides which agent to use
│   ├── 📂 talk2books/              # The logic for the Book Recommendation agent
│   ├── 📂 talk2docs/               # The logic for parsing PDFs and the Vector Database
│   ├── 📂 talk2jobs/               # The logic for finding Jobs online
│   └── 📂 talk2papers/             # The logic for searching Academic Papers
│
├── 📂 app/                         # 🖥️ The Web Application
│   ├── 📂 backend/                 # FastAPI server (`main.py`) connecting the "Brain" to the web
│   └── 📂 frontend/                # Streamlit UI (`streamlit_app.py`) which you see in the browser
│
├── 📂 docs/                        # 📊 Documentation & Testing
│   └── 📂 evaluation/              # Scripts to test accuracy, speed, and robustness of the AI
│
├── 📂 data/                        # 💾 Local storage for downloaded/cached data
├── 📂 uploads/                     # 📄 Where your uploaded PDFs are temporarily saved
│
├── 📜 .env                         # (You create this) Your secret API keys
├── 📜 pyproject.toml / uv.lock     # List of all Python packages required to make this work
├── 📜 Dockerfile                   # Instructions for running the app inside isolated Docker containers
└── 📜 README.md                    # You are reading this right now!
```

---

## 🐳 Docker Deployment (Optional)

If you prefer using Docker to avoid installing Python dependencies directly:
```bash
docker-compose up --build
```
This will start both the backend API and the frontend dashboard automatically in isolated containers.

---

## 🤝 Contributing
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License
Distributed under the MIT License. See `pyproject.toml` for metadata. Built for educational and research purposes.
