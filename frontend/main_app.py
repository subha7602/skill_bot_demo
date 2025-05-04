import streamlit as st
from backend.vector_store import ResumeVectorStore

FAISS_INDEX_PATH = "faiss_index"

# Initialize session state
if "vector_store" not in st.session_state:
    resume_store = ResumeVectorStore()
    vector_store = resume_store.load_vector_store(FAISS_INDEX_PATH)
    st.session_state.vector_store = vector_store
    st.session_state.resume_store = resume_store

def render_main_app():
    """
    Renders the main UI for resume analysis.
    """


if __name__ == "__main__":
    render_main_app()