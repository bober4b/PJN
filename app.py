import streamlit as st
import functions
import service.document_service as document_service
import os
import sys
import subprocess
from datetime import datetime


st.set_page_config(layout="wide", page_icon="📄")

# =====================
# Session state init
# =====================

if "doc_service" not in st.session_state:
    st.session_state.doc_service = document_service.DocumentService()

if "mod_service" not in st.session_state:
    st.session_state.mod_service = None

if "need_retrain" not in st.session_state:
    st.session_state.need_retrain = False

if "retrain_decision" not in st.session_state:
    st.session_state.retrain_decision = None

# =====================
# Cross-platform open
# =====================

def open_doc(doc_name):
    path = os.path.join(
        os.getcwd(),
        document_service.DocumentService.DOCS_DIR_PATH,
        doc_name
    )

    if sys.platform.startswith("darwin"):
        subprocess.call(["open", path])
    elif os.name == "nt":
        os.startfile(path)
    else:
        subprocess.call(["xdg-open", path])

# =====================
# Retrain prompt (INLINE)
# =====================

def retrain_prompt():
    st.warning(
        "Wykryto zmiany w dokumentach. "
        "Czy chcesz ponownie wytrenować model? (może zająć kilka minut)"
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Tak"):
            st.session_state.retrain_decision = True
            st.session_state.need_retrain = False
            st.session_state.mod_service = None
            st.rerun()

    with col2:
        if st.button("Nie"):
            st.session_state.retrain_decision = False
            st.session_state.need_retrain = False
            st.session_state.mod_service = None
            st.rerun()

# =====================
# Model init (ONE PLACE)
# =====================

def init_model():
    if st.session_state.mod_service is not None:
        return

    if st.session_state.doc_service.has_changes():
        if st.session_state.retrain_decision is None:
            st.session_state.need_retrain = True
            return

        if st.session_state.retrain_decision:
            with st.spinner("Trenowanie modelu..."):
                st.session_state.mod_service = (
                    functions.get_updated_model_service(
                        st.session_state.doc_service
                    )
                )
        else:
            st.session_state.mod_service = (
                functions.get_model_service(
                    st.session_state.doc_service
                )
        )
    else:
        st.session_state.mod_service = (
            functions.get_model_service(
                st.session_state.doc_service
            )
        )

def documents_view():
    st.divider()
    st.header("📄 Dokumenty")

    uploaded = st.file_uploader(
        "Dodaj dokument",
        type=["txt", "docx"]
    )

    if uploaded:
        path = os.path.join(
            os.getcwd(),
            document_service.DocumentService.DOCS_DIR_PATH,
            uploaded.name
        )

        with open(path, "wb") as f:
            f.write(uploaded.getbuffer())

        st.success(f"Plik {uploaded.name} zapisany")

        st.session_state.retrain_decision = None
        st.session_state.need_retrain = True
        st.session_state.mod_service = None

        st.rerun()

    docs = st.session_state.doc_service.documents

    if not docs:
        st.info("Brak dokumentów.")
        return

    for doc in docs:
        col1, col2, col3 = st.columns([4, 3, 1])

        with col1:
            st.write(doc.name)

        with col2:
            st.write(
                datetime.fromtimestamp(doc.mod_date)
                .strftime("%Y-%m-%d %H:%M")
            )

        with col3:
            st.button(
                "Otwórz",
                key=f"{doc.name}_open",
                on_click=open_doc,
                args=[doc.name]
            )


# =====================
# Search UI (GUARDED)
# =====================

st.header("🔍 Wyszukiwanie dokumentów")

init_model()

if st.session_state.need_retrain:
    retrain_prompt()
    st.stop()

if st.session_state.mod_service is None:
    st.info("Model nie jest gotowy.")
    st.stop()

query = st.text_input("Wprowadź słowa kluczowe")

if st.button("Zatwierdź") and query.strip():
    st.session_state.search_doc2vec = (
        st.session_state.mod_service.search_doc2vec(query, 5)
    )
    st.session_state.search_tfidf = (
        st.session_state.mod_service.search_tfidf(query, 5)
    )

# =====================
# Results
# =====================

if st.session_state.get("search_doc2vec"):
    st.subheader("Doc2Vec")
    for name, score in st.session_state.search_doc2vec:
        c1, c2, c3 = st.columns([3, 2, 1])
        c1.write(name)
        c2.write(score)
        c3.button(
            "Open",
            key=f"{name}_d2v",
            on_click=open_doc,
            args=[name]
        )

if st.session_state.get("search_tfidf"):
    st.subheader("TF-IDF")
    for i, (name, score) in enumerate(st.session_state.search_tfidf):
        c1, c2, c3 = st.columns([3, 2, 1])
        c1.write(name)
        c2.write(score)
        c3.button(
            "Open",
            key=f"{name}_tfidf_{i}",
            on_click=open_doc,
            args=[name]
        )

documents_view()
