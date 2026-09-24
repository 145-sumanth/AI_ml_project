FROM python:3.11-slim

# Use /app/support_assistant as the working directory so CMD can be 'uvicorn main:app'
WORKDIR /app/support_assistant

# Copy the module into the image
COPY support_assistant /app/support_assistant
COPY support_assistant/requirements.txt /app/support_assistant/requirements.txt

# Install system deps and Python requirements. Keep image slim by cleaning caches.
RUN apt-get update && apt-get install -y --no-install-recommends gcc build-essential libpq-dev \
    && python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /app/support_assistant/requirements.txt \
    && rm -rf /var/lib/apt/lists/*

# Pre-download the embedding model and build the Chroma index at build time by running ingest.py.
# This ensures the runtime starts without needing to download large models.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
RUN python ingest.py

EXPOSE 7860
# Use the module-local main:app as requested
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
