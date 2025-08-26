"""
Utility functions for interacting with the backend API, including PDF upload
and question submission.
"""

import os
import requests


def get_backend_url() -> str:
    """
    Return the backend API URL.

    Uses the BACKEND_URL environment variable or a default.
    """
    # In the Docker, let's set BACKEND_URL=http://rag-api:8000
    return os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")


def upload_pdfs(files):
    """Send the PDFs to the backend "/documents" (multipart/form-data)."""
    url = f"{get_backend_url()}/documents"
    multipart = []
    for f in files:
        multipart.append(
            ("files", (f.name, f.getvalue(), "application/pdf"))
        )
    resp = requests.post(url, files=multipart, timeout=120)
    resp.raise_for_status()
    return resp.json()


def ask_question(question: str):
    """Send the question to the backend "/question"."""
    url = f"{get_backend_url()}/question"
    payload = {"question": question}
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()
