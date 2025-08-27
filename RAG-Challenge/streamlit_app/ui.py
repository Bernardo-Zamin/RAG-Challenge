# RAG-Challenge/streamlit_app/ui.py
import streamlit as st
from streamlit_app import utils

st.set_page_config(page_title="RAG Challenge", page_icon="🧠", layout="centered")

# --- session boot ---
if "session_id" not in st.session_state:
    try:
        st.session_state.session_id = utils.start_chat()
    except Exception:
        st.error("Backend ainda iniciando… tente novamente em alguns segundos.")
        st.stop()
if "messages" not in st.session_state:
    st.session_state.messages = []

st.sidebar.title("📄 Documents")
new_chat = st.sidebar.button("🧹 New chat (reset vectors)")
if new_chat:
    st.session_state.session_id = utils.start_chat()
    st.session_state.messages = []
    st.sidebar.success("Session restarted. Vector DB cleared.")

uploaded = st.sidebar.file_uploader(
    "Upload PDF(s)",
    type=["pdf"],
    accept_multiple_files=True,
    help="The PDFs will be indexed in the current session",
)

if uploaded and st.sidebar.button("Index PDFs"):
    with st.spinner("Processing and indexing..."):
        resp = utils.upload_documents(uploaded, st.session_state.session_id)
    st.sidebar.success(
        f"OK! {resp.get('documents_indexed',0)} doc(s), "
        f"{resp.get('total_chunks',0)} chunks, "
        f"{resp.get('indexed_points',0)} vectors."
    )

st.title("💬 Chat RAG")
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant" and m.get("references"):
            with st.expander("🔎 Used parts (context)"):
                for i, ref in enumerate(m["references"], start=1):
                    st.markdown(f"**{i}.** {ref}")

user_msg = st.chat_input("Ask something about the PDFs…")
if user_msg:
    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    with st.chat_message("assistant"):
        with st.spinner("Consulting…"):
            resp = utils.ask_question(user_msg, st.session_state.session_id)
        answer = resp.get("answer", "No answer.")
        refs = resp.get("references", [])
        st.markdown(answer)
        if refs:
            with st.expander("🔎 Used parts (context)"):
                for i, ref in enumerate(refs, start=1):
                    st.markdown(f"**{i}.** {ref}")
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "references": refs
        })
