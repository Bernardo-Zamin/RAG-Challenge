"""Utility functions for interacting with the RAG API backend."""

import os
import time
import requests

# In container: http://rag-api:8000 ; outside the container you can export BACKEND_URL  # noqa: E501
API_BASE = os.getenv("BACKEND_URL", "http://rag-api:8000")


def _post(url, **kwargs):
    """Send a POST request to the specified URL.

    - For convention we use "_" prefix for private functions.
    """
    attempts = 12
    last_err = None
    for i in range(attempts):
        try:
            return requests.post(url, timeout=120, **kwargs)
        except requests.RequestException as e:
            last_err = e
            time.sleep(1 + i)  # backoff
    raise last_err


def start_chat():
    """Start a new chat session and return the session ID."""
    r = _post(f"{API_BASE}/start_chat")
    r.raise_for_status()
    return r.json()["session_id"]


def upload_documents(files, session_id: str):
    """files: list of st.uploadedfile.UploadedFile."""
    files_payload = [
        ("files", (f.name, f.getvalue(), "application/pdf")) for f in files
    ]
    data = {"session_id": session_id}
    r = _post(f"{API_BASE}/documents", files=files_payload, data=data)
    r.raise_for_status()
    return r.json()


def ask_question(question: str, session_id: str):
    """Send a question to the backend and return the response.

    Args:
        question (str): The question to ask.
        session_id (str): The session identifier.

    Returns:
        dict: The response from the backend.
    """
    r = _post(
        f"{API_BASE}/question",
        json={
            "question": question,
            "session_id": session_id,
        },
    )
    r.raise_for_status()
    return r.json()
