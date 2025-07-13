import streamlit as st

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

def render_getting_started():
    st.markdown("""
    <div class="info-box">
        <h4>Getting Started</h4>
        <p>1. Upload a text document (.txt) using the file uploader above</p>
        <p>2. Wait for the document to be processed</p>
        <p>3. Start asking questions about your document!</p>
    </div>
    """, unsafe_allow_html=True)

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

def render_processing_spinner(message: str = "Processing..."):
    return st.spinner(message) 








