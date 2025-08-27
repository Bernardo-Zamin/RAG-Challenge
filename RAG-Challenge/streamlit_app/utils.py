# RAG-Challenge/streamlit_app/utils.py
import os
import time
import requests

# Em container: http://rag-api:8000 ; fora do container você pode exportar BACKEND_URL
API_BASE = os.getenv("BACKEND_URL", "http://rag-api:8000")

def _post(url, **kwargs):
    attempts = 12
    last_err = None
    for i in range(attempts):
        try:
            return requests.post(url, timeout=30, **kwargs)
        except requests.RequestException as e:
            last_err = e
            time.sleep(1 + i)  # backoff
    raise last_err

def start_chat():
    r = _post(f"{API_BASE}/start_chat")
    r.raise_for_status()
    return r.json()["session_id"]

def upload_documents(files, session_id: str):
    """
    files: lista de st.uploadedfile.UploadedFile
    """
    files_payload = [
        ("files", (f.name, f.getvalue(), "application/pdf"))
        for f in files
    ]
    data = {"session_id": session_id}
    r = _post(f"{API_BASE}/documents", files=files_payload, data=data)
    r.raise_for_status()
    return r.json()

def ask_question(question: str, session_id: str):
    r = _post(f"{API_BASE}/question", json={"question": question, "session_id": session_id})
    r.raise_for_status()
    return r.json()
