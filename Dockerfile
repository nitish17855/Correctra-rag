# Production Dockerfile for Enterprise Corrective RAG (CRAG) on Hugging Face Spaces
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    HOME=/home/user

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up standard non-root user (UID 1000 required by Hugging Face Spaces)
RUN useradd -m -u 1000 user
WORKDIR $HOME/app

# Install Python dependencies as root
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK sentence tokenization models
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Copy application files and set permissions for non-root user
COPY --chown=user:user . $HOME/app

# Switch to non-root user
USER user
ENV PATH=$HOME/.local/bin:$PATH

# Expose standard Hugging Face Spaces port
EXPOSE 7860

# Container health probe
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Launch ASGI FastAPI application on port 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
