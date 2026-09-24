# Multi-module repository

This repository is organized into three modules:

- `data_pipeline/` — scrapes, cleans, stores, and queries book data from books.toscrape.com.
- `analytics/` — exploratory data analysis and modeling notebooks built around the Titanic dataset.
- `support_assistant/` — a local document-retrieval support assistant with a FastAPI endpoint.

The repository also includes a single consolidated `requirements.txt` for Python package installation.

## Current status

The `data_pipeline` module is implemented and validated first, as requested. Analytics and support assistant work will proceed only after approval to continue.
