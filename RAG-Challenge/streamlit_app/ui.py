"""Streamlit UI for the chatbot."""

from __future__ import annotations
import os
from pathlib import Path
import time
import concurrent.futures as cf
import requests
import streamlit as st
from streamlit_app import utils

API_BASE_URL = "http://rag-api:8000"

try:
    from dotenv import load_dotenv, find_dotenv, dotenv_values

    HERE = Path(__file__).resolve().parent
    PROJECT_DIR = HERE.parent  # one level above ui.py
    CWD = Path.cwd()

    auto_env = find_dotenv(usecwd=True)

    candidates = [
        HERE / ".env",
        PROJECT_DIR / ".env",
        CWD / ".env",
        Path(auto_env) if auto_env else None,
    ]
    candidates = [p for p in candidates if p and p.exists()]

    loaded_files = []
    for p in candidates:
        try:
            if load_dotenv(p, override=True):
                loaded_files.append(str(p))
        except Exception:
            pass

    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "").strip()

    if not OLLAMA_MODEL and candidates:
        for p in candidates:
            try:
                vals = dotenv_values(p)
                v = (vals.get("OLLAMA_MODEL") or "").strip()
                if v:
                    OLLAMA_MODEL = v
                    break
            except Exception:
                pass

except ModuleNotFoundError:
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "").strip()
    loaded_files = []

model_badge = OLLAMA_MODEL if OLLAMA_MODEL else "not set"


st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="💬",
    layout="wide",
)


if "session_id" not in st.session_state:
    st.session_state.session_id = f"ssn-{int(time.time())}"
if "messages" not in st.session_state:
    st.session_state.messages = []


OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "").strip()
model_badge = OLLAMA_MODEL if OLLAMA_MODEL else "not set"


st.markdown(
    """
<style>
:root{
  --bg: #0b0f19;
  --panel: #11172a;
  --text: #e8edf7;
  --muted: #a6b2c5;
  --primary: #5b9dff;
  --accent: #62e0a1;
  --warning: #ffcc66;
  --bubble-user: #1a2236;
  --bubble-bot: #0f1a2b;
  --border: #1f2a40;
}
.stApp { background: linear-gradient(180deg, #0b0f19 0%, #0b0f19 60%, #0e1425 100%) !important; }
h1, h2, h3, h4, h5, h6, p, span, li, div { color: var(--text) !important; }
small, .stMarkdown p small { color: var(--muted) !important; }

.block-container { padding-top: 2rem !important; }

/* chat bubbles */
.chat-bubble {
  padding: 0.9rem 1.05rem;
  border-radius: 14px;
  border: 1px solid var(--border);
  box-shadow: 0 6px 18px rgba(0,0,0,0.25);
  line-height: 1.55;
}
.user-bubble { background: var(--bubble-user); }
.bot-bubble { background: var(--bubble-bot); }

/* remove default black box around user chat */
.stChatMessage[data-testid="stChatMessageUser"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* hero */
.hero {
  width: 100%;
  background: radial-gradient(1200px 600px at 10% -20%, rgba(91,157,255,0.20), transparent 60%),
              radial-gradient(800px 500px at 90% -10%, rgba(98,224,161,0.18), transparent 60%);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 24px;
  margin-bottom: 18px;
}
.hero h1 { margin: 0; font-size: 2.1rem; letter-spacing: 0.3px; }
.subtle { color: var(--muted); margin-top: 6px; }

/* badges */
.badges { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 12px; }
.badge {
  display: inline-flex; align-items: center; gap: 6px;
  font-size: 0.85rem; padding: 6px 10px; border-radius: 999px;
  border: 1px solid var(--border); background: #0e1528;
}

/* file uploader */
.stFileUploader { background: #0f162a !important; border: 1px dashed var(--border) !important; border-radius: 14px !important; padding: 10px !important; }

/* expander */
details { background: #0f162a !important; border: 1px solid var(--border) !important; border-radius: 12px !important; }

/* footer */
.footer { opacity: 0.7; font-size: 0.85rem; margin-top: 8px; }
.ok { color: #8be28b; }
.warn { color: #ffd78a; }
</style>
""",
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.caption("Manage documents and preferences.")

    if st.button("🆕 New Chat"):
        try:
            response = requests.post(f"{API_BASE_URL}/start_chat")
            response.raise_for_status()
            new_session_id = response.json().get("session_id")
            if new_session_id:
                st.session_state.session_id = new_session_id
                st.session_state.messages = []
                st.success("New chat session started!")
            else:
                st.error("Failed to start a new chat session.")
        except Exception as e:
            st.error(f"Error starting new chat: {e}")

    uploaded = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Select one or more PDFs",
    )

    if st.button("📤 Index PDFs"):
        if not uploaded:
            st.warning("No files selected.")
        else:
            with st.spinner("Uploading and indexing documents..."):
                resp = utils.upload_documents(uploaded, st.session_state.session_id)
            st.success("Documents indexed!")

    st.divider()
    st.markdown("### ℹ️ About")
    st.caption(
        "This app is a **RAG Chatbot** running **CPU-only**, powered by **Ollama** for local model serving. "
        "Ask questions and get answers grounded in your uploaded PDFs."
    )
    st.markdown(
        "<div class='badges'>"
        "<span class='badge'>💻 CPU-only</span>"
        "<span class='badge'>🔎 Retrieval-Augmented Generation</span>"
        f"<span class='badge'>🧰 Ollama</span>"
        f"<span class='badge'>🧠 Model: <code>{model_badge}</code></span>"
        "</div>",
        unsafe_allow_html=True,
    )
    sid_footer = f"v1 • session: <code>{st.session_state.session_id}</code>"
    st.markdown(f"<div class='footer'>{sid_footer}</div>", unsafe_allow_html=True)


st.markdown(
    f"""
    <div class="hero">
      <h1>RAG Chatbot</h1>
      <div class="subtle">Ask questions and receive answers grounded in the content of your uploaded PDFs. Runs 100% on CPU.</div>
      <div class="badges" style="margin-top:10px;">
        <span class="badge">🧠 LLM + RAG</span>
        <span class="badge">Model: <code>{model_badge}</code></span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


if not OLLAMA_MODEL:
    st.info(
        "Ollama model is not set. Define it in your environment (e.g., in `.env`) using `OLLAMA_MODEL=<model_name>`.",
        icon="⚠️",
    )


for m in st.session_state.messages:
    role = m["role"]
    content = m["content"]
    refs = m.get("references", []) if role == "assistant" else []
    with st.chat_message(role):
        st.markdown(
            f"<div class='chat-bubble {'bot-bubble' if role=='assistant' else 'user-bubble'}'>{content}</div>",
            unsafe_allow_html=True,
        )
        if refs:
            with st.expander("📎 References used (context)"):
                for i, ref in enumerate(refs, start=1):
                    st.markdown(f"**{i}.** {ref}")


placeholder = "Ask something about the PDFs… (e.g., 'Summarize the methods section of the first document')"
user_msg = st.chat_input(placeholder)

if user_msg:

    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(
            f"<div class='chat-bubble user-bubble'>{user_msg}</div>",
            unsafe_allow_html=True,
        )

    with st.chat_message("assistant"):
        status = st.empty()
        t0 = time.time()
        try:
            with st.status(
                "Working… fetching context and generating an answer…", expanded=False
            ) as st_status:
                with cf.ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(
                        utils.ask_question, user_msg, st.session_state.session_id
                    )
                    resp = fut.result()

                total = time.time() - t0
                answer = resp.get("answer", "No answer.")
                refs = resp.get("references", [])

                st_status.update(label="Answer ready", state="complete", expanded=False)

            st.markdown(
                f"<div class='chat-bubble bot-bubble'>{answer}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='exec-time'>Execution time: {total:.2f}s</div>",
                unsafe_allow_html=True,
            )

            if refs:
                with st.expander("📎 References used (context)"):
                    for i, ref in enumerate(refs, start=1):
                        st.markdown(f"**{i}.** {ref}")

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "references": refs}
            )

        except Exception:
            with st.spinner("Working… fetching context and generating an answer…"):
                with cf.ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(
                        utils.ask_question, user_msg, st.session_state.session_id
                    )
                    resp = fut.result()

            total = time.time() - t0
            answer = resp.get("answer", "No answer.")
            refs = resp.get("references", [])

            st.markdown(
                f"<div class='chat-bubble bot-bubble'>{answer}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='exec-time'>Execution time: {total:.2f}s</div>",
                unsafe_allow_html=True,
            )

            if refs:
                with st.expander("📎 References used (context)"):
                    for i, ref in enumerate(refs, start=1):
                        st.markdown(f"**{i}.** {ref}")

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "references": refs}
            )

        except Exception as e:
            st.error(f"Error while processing your question: {e}")
