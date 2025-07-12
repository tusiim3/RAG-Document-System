import streamlit as st
from typing import List, Optional
from langchain_core.documents import Document

def setup_page_config():
    st.set_page_config(
        page_title="RAG Document System",
        page_icon="📚",
        layout="centered"
    )

def load_custom_css():
    st.markdown("""
    <style>
        .info-box {
            background-color: #f0f2f6;
            color: #000; 
            padding: 0.5rem;
            margin: 0.5rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
        }
    </style>
    """, unsafe_allow_html=True)

def render_header():
    st.markdown('<h1 class="main-header">📚 RAG Document System</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">Upload and interact with your documents</h2>', unsafe_allow_html=True)

def render_document_upload():
    uploaded_file = st.file_uploader(
        "Choose a text document (.txt)",
        type=['txt'],
        help="Upload a large text document to build the knowledge base"
    )
    return uploaded_file

def render_clear_chat_button():
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

def render_document_stats(stats: dict):
    if not stats:
        return
    
    st.markdown('<div class="success-box">✅ Document processed successfully!</div>', unsafe_allow_html=True)
    
    # Display detailed statistics
    with st.expander("📊 Document Statistics"):
        st.markdown(f"""
        <div class="stats-container">
            <div class="stat-item">
                <strong>Chunks</strong><br>
                {stats.get('total_chunks', 0)}
            </div>
            <div class="stat-item">
                <strong>Characters</strong><br>
                {stats.get('total_characters', 0):,}
            </div>
            <div class="stat-item">
                <strong>Avg Size</strong><br>
                {stats.get('avg_chunk_size', 0):.0f}
            </div>
            <div class="stat-item">
                <strong>Max Size</strong><br>
                {stats.get('max_chunk_size', 0)}
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_getting_started():
    st.markdown("""
    <div class="info-box">
        <h4>Getting Started</h4>
        <p>1. Upload a text document (.txt) using the file uploader above</p>
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

def render_system_info(system_info: dict):
    """Render system information"""
    with st.expander("🔧 System Information"):
        if not system_info:
            st.info("System information not available")
            return
        
        # Basic configuration
        st.markdown("**Configuration:**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"• Chunk Size: {system_info.get('chunk_size', 'N/A')}")
            st.write(f"• Chunk Overlap: {system_info.get('chunk_overlap', 'N/A')}")
            st.write(f"• Temperature: {system_info.get('temperature', 'N/A')}")
        
        with col2:
            st.write(f"• Embedding Model: {system_info.get('embedding_model', 'N/A')}")
            st.write(f"• Persist Directory: {system_info.get('persist_directory', 'N/A')}")
        
        # Component status
        st.markdown("**Component Status:**")
        components = system_info.get('components_initialized', {})
        for component, status in components.items():
            status_icon = "✅" if status else "❌"
            st.write(f"{status_icon} {component.replace('_', ' ').title()}")
        
        # Embedding info
        if 'embedding_info' in system_info:
            st.markdown("**Embedding Model Info:**")
            embedding_info = system_info['embedding_info']
            st.write(f"• Model: {embedding_info.get('model_name', 'N/A')}")
            st.write(f"• Device: {embedding_info.get('device', 'N/A')}")
            st.write(f"• Dimensions: {embedding_info.get('dimension', 'N/A')}")
        
        # Vector store stats
        if 'vector_store_stats' in system_info:
            st.markdown("**Vector Store Stats:**")
            vector_stats = system_info['vector_store_stats']
            st.write(f"• Total Documents: {vector_stats.get('total_documents', 0)}")
            st.write(f"• Collection: {vector_stats.get('collection_name', 'N/A')}")


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








