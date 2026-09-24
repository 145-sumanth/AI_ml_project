FROM python:3.11-slim

WORKDIR /app

# Copy only what is needed for the support assistant
COPY support_assistant /app/support_assistant
COPY requirements.txt /app/requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends gcc build-essential libpq-dev \
    && python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /app/requirements.txt \
    && rm -rf /var/lib/apt/lists/*

# Pre-download the embedding model at build time to avoid download on first run
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 7860
CMD ["uvicorn", "support_assistant.main:app", "--host", "0.0.0.0", "--port", "7860"]
