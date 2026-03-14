# ==========================================
# Optimized System Install (No Virtual Env)
# ==========================================
FROM python:3.10-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy files
COPY . .

# 1. Export dependencies to requirements.txt
# 2. Install dependencies globally (--system)
# 3. Install the project package globally
RUN uv export --format requirements-txt > requirements.txt && \
    uv pip install --system --no-cache --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt && \
    uv pip install --system --no-cache . && \
    rm requirements.txt

# Set environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8000

# Run FastAPI directly (no path adjustment needed)
CMD sh -c "uvicorn app.backend.main:app --host 0.0.0.0 --port ${PORT}"
