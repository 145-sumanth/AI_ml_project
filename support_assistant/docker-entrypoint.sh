#!/bin/sh
set -e

# docker-entrypoint.sh
# If RUN_INGEST=1, build the Chroma index at container start (writes to /app/support_assistant/chroma_db)
# If PERSIST_DIR is set, use it as the chroma persistence directory (mounted from host)

cd /app/support_assistant

if [ "${RUN_INGEST:-0}" = "1" ]; then
  echo "RUN_INGEST=1 -> running ingest.py to build the index (this may download the embedding model)..."
  python3 ingest.py
  echo "Ingest finished"
fi

# Exec the final command (uvicorn API by default)
exec "$@"
