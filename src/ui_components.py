import streamlit as st
from typing import List, Optional
from langchain_core.documents import Document

def setup_page_config():
    st.set_page_config(
        page_title="Intelligent Document Query System",
        page_icon="📚",
        layout="centered",
        initial_sidebar_state="expanded",
    )

def render_header():
    st.sidebar.markdown('<h3 class="sub-header">📄 Document Upload</h3>', unsafe_allow_html=True)

def render_document_upload():
    uploaded_file = st.sidebar.file_uploader(
        "Choose a text document (.txt)",
        type=['txt'],
        help="Upload a large text document to build the knowledge base"
    )
    return uploaded_file

def render_clear_chat_button():
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

def render_document_stats(stats: dict):
     if not stats:
        return
    
    st.sidebar.success(f"✅ Document processed successfully! ({stats.get('total_chunks', 0)} chunks created)")

    # Display detailed statistics
    with st.sidebar.expander("📊 Document Statistics"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Chunks", stats.get('total_chunks', 0))
            st.metric("Characters", f"{stats.get('total_characters', 0):,}")
        
        with col2:
            st.metric("Avg Chunk Size", f"{stats.get('avg_chunk_size', 0):.0f}")
            st.metric("Max Chunk Size", stats.get('max_chunk_size', 0))

def render_getting_started():
    st.markdown("""
    <div class="info-box">
        <h4>Getting Started</h4>
        <p>1. Upload a text document (.txt) using the sidebar</p>
        <p>2. Wait for the document to be processed</p>
        <p>3. Start asking questions about your document!</p>
    </div>
    """, unsafe_allow_html=True)

def render_chat_interface():
    st.markdown('<h3 class="sub-header">💬 Ask Questions About Your Document</h3>', unsafe_allow_html=True)
    # Chat input
    user_question = st.chat_input("Ask a question about your document...")
    return user_question

def render_chat_history(chat_history: List[dict]):
    for message in chat_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
                
                # Show source documents if available
                if "sources" in message and message["sources"]:
                    with st.expander("📄 View Source Documents"):
                        for i, doc in enumerate(message["sources"][:3]):  # Show top 3 sources
                            st.markdown(f"**Source {i+1}:**")
                            st.markdown(f'<div class="source-document">{doc.page_content[:300]}{"..." if len(doc.page_content) > 300 else ""}</div>', unsafe_allow_html=True)
                            st.divider()

# def render_system_info(system_info: dict):

def render_error_message(message:  str):
    st.error(f"❌ {message}")

def render_success_message(message:str):
    st.success(f"✅ {message}")

def render_warning_message(message: str):
    st.warning(f"⚠️ {message}")


def render_info_message(message: str):
    st.info(f"ℹ️ {message}")

def render_processing_spinner(message: str = "Processing..."):
    return st.spinner(message)

def render_progress_bar(progress: float, message: str = "Processing"):
    return st.progress(progress, text=message)

def doc_upload_section():
    uploaded_files = st.file_uploader(
        "📄 Upload a text document (.txt only, max 200MB)",
        type=["txt"],
        accept_multiple_files=True,
        key="rag_docs",
    )
    return uploaded_files

def render_documents_in_db(documents: List[str]):
    with st.expander(f"📚 Documents in DB ({len(documents)})"):
        if documents:
            for doc in documents:
                st.write(f"• {doc}")
        else:
            st.write("No documents in database")

def render_chat_messages(messages: List[dict]):
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
        
def render_source_documents(source_docs: List[Document], max_sources: int = 3):
    if source_docs and isinstance(source_docs, list) and len(source_docs) > 0:
        with st.expander("📄 View Source Documents"):
            for i, doc in enumerate(source_docs[:max_sources]):
                st.markdown(f"**Source {i+1}:**")
                st.markdown(f'<div class="source-document">{doc.page_content[:300]}{"..." if len(doc.page_content) > 300 else ""}</div>', unsafe_allow_html=True)
                st.divider() 








