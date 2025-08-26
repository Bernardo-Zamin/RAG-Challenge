import streamlit as st
from utils import upload_pdfs, ask_question

st.set_page_config(page_title="RAG Challenge", page_icon="📄", layout="centered")

# --- Sidebar: Upload de PDFs ---
with st.sidebar:
    st.header("📄 Upload de documentos")
    st.caption("Faça upload de um ou mais PDFs para indexação.")
    uploaded = st.file_uploader(
        "Selecionar PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Os PDFs serão parseados e indexados para RAG."
    )
    if uploaded:
        if st.button("Indexar PDFs"):
            with st.spinner("Processando e indexando PDFs..."):
                try:
                    res = upload_pdfs(uploaded)
                    st.success(
                        f"✅ {res.get('documents_indexed', 0)} documento(s) indexado(s), "
                        f"{res.get('total_chunks', 0)} chunk(s) adicionados."
                    )
                except Exception as e:
                    st.error(f"Falha ao indexar: {e}")

st.title("🤖 RAG Chatbot")
st.caption("Pergunte sobre os documentos que você enviou.")

# Estado da conversa
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar histórico
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("references"):
            with st.expander("🔎 Referências"):
                for r in m["references"]:
                    st.write(f"- {r}")

# Entrada do usuário
prompt = st.chat_input("Digite sua pergunta...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando o backend..."):
            try:
                result = ask_question(prompt)
                answer = result.get("answer", "Sem resposta.")
                refs = result.get("references") or []

                st.markdown(answer)
                if refs:
                    with st.expander("🔎 Referências"):
                        for r in refs:
                            st.write(f"- {r}")

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "references": refs}
                )
            except Exception as e:
                msg = f"Erro consultando backend: {e}"
                st.error(msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": msg}
                )
