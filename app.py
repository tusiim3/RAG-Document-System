import streamlit as st
import os
import tempfile
import logging
from dotenv import load_dotenv
import uuid

# UI Components moved to src/ui_components.py for easier debugging and maintenance

from src.ui_components import (
    setup_page_config, load_custom_css, render_header, 
    render_getting_started, render_system_info, 
    render_processing_spinner
)
from src.rag_pipeline import RAGPipeline

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_session_state():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = None

    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'rag_sources' not in st.session_state:
        st.session_state.rag_sources = []
    
    if 'document_loaded' not in st.session_state:
        st.session_state.document_loaded = False
    
    if 'document_stats' not in st.session_state:
        st.session_state.document_stats = None

def process_uploaded_document(uploaded_file):
    try:
        st.info(f"Starting to process: {uploaded_file.name}")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt', mode='w', encoding='utf-8') as tmp_file:
            content = uploaded_file.getvalue().decode('utf-8')
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        st.info(f"File saved temporarily at: {tmp_file_path}")
        st.info(f"File content length: {len(content)} characters")

        # Initialize RAG pipeline if not already done
        if st.session_state.rag_pipeline is None:
            st.info("Initializing RAG pipeline...")
            st.session_state.rag_pipeline = RAGPipeline()

        # Process document
        st.info("Processing document through RAG pipeline...")
        success = st.session_state.rag_pipeline.process_document(tmp_file_path)
        
        if success:
            st.info("Document processed successfully, getting statistics...")
            # Get document statistics
            chunks = st.session_state.rag_pipeline.document_processor.process_document(tmp_file_path)
            stats = st.session_state.rag_pipeline.document_processor.get_document_stats(chunks)
            
            # Update session state
            st.session_state.document_loaded = True
            st.session_state.document_stats = stats
            
            st.info(f"Document processed successfully: {stats['total_chunks']} chunks")
        else:
            st.error("Failed to process document")
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return success
        
    except Exception as e:
        st.error(f"Error processing uploaded document: {e}")
        logger.error(f"Error processing uploaded document: {e}")
        return False
    
def handle_user_query(user_question):
    try:
        if not st.session_state.rag_pipeline or not st.session_state.document_loaded:
            return "Please upload a document first before asking questions.", []
        
        # Add user question to messages
        st.session_state.messages.append({"role": "user", "content": user_question})
        
        # Get response from RAG pipeline
        with render_processing_spinner("Thinking..."):
            answer, source_docs = st.session_state.rag_pipeline.query(user_question)

         # Add assistant response to messages
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer, 
            "sources": source_docs
        })

        logger.info(f"Query processed: '{user_question[:50]}...'")
        return answer, source_docs
        
    except Exception as e:
        logger.error(f"Error handling user query: {e}")
        error_message = f"Error processing query: {str(e)}"
        st.session_state.messages.append({"role": "assistant", "content": error_message, "sources": []})
        return error_message, []
    
def clear_all_documents():
    st.session_state.rag_sources = []
    st.session_state.document_loaded = False
    st.session_state.document_stats = None
    st.session_state.rag_pipeline = None
    st.session_state.uploaded_files = []

    # Clear the vector store as well
    if st.session_state.rag_pipeline and st.session_state.rag_pipeline.vector_store_manager:
        st.session_state.rag_pipeline.vector_store_manager.clear_vector_store()
    
    # Increment uploader key to reset file uploader
    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key = 0
    st.session_state.uploader_key += 1
    st.rerun()

def process_uploaded_files():
    if 'uploaded_files' in st.session_state and st.session_state.uploaded_files:
        for uploaded_file in st.session_state.uploaded_files:
            if uploaded_file.name not in st.session_state.rag_sources:
                # Simple test - just read the file content first
                try:
                    content = uploaded_file.getvalue().decode('utf-8')
                    st.success(f"✅ {uploaded_file.name} uploaded successfully! Content length: {len(content)} characters")
                    st.session_state.rag_sources.append(uploaded_file.name)
                    
                    # Set document_loaded to True when we have files
                    st.session_state.document_loaded = True
                    
                    # Now try to process with RAG pipeline
                    with st.spinner(f"Processing {uploaded_file.name} with RAG..."):
                        success = process_uploaded_document(uploaded_file)
                        if success:
                            st.success(f"✅ {uploaded_file.name} RAG processing completed!")
                        else:
                            st.error(f"❌ RAG processing failed for {uploaded_file.name}")
                            
                except Exception as e:
                    st.error(f"❌ Error reading {uploaded_file.name}: {e}")
        
        # Clear the uploaded files from session state to prevent reprocessing
        st.session_state.uploaded_files = []

def main():
    # Setup page configuration and styling
    setup_page_config()
    load_custom_css()
    
    # Initialize session state
    initialize_session_state()
    
    # Render main header
    render_header()

    # Add getting started section
    if not st.session_state.document_loaded:
        render_getting_started()

    # Clear buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat", type="primary"):
            st.session_state.messages.clear()
            st.rerun()
    with col2:
        if st.button("Clear All Documents", type="secondary"):
            clear_all_documents()

    # Initialize uploader key
    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key = 0
    
    # File upload input 
    uploaded_files = st.file_uploader(
        "📄 Upload a text document (.txt only, max 200MB)", 
        type=["txt"],
        accept_multiple_files=True,
       key=f"rag_docs_{st.session_state.uploader_key}"
    )
    
    # Store uploaded files in session state and process them
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        st.info(f"Files uploaded: {[f.name for f in uploaded_files]}")
        process_uploaded_files()
    
    # Show documents in DB with individual remove buttons
    with st.expander(f"📚 Documents in DB ({len(st.session_state.rag_sources)})"):
        if st.session_state.rag_sources:
            for i, doc in enumerate(st.session_state.rag_sources):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"• {doc}")
                with col2:
                    if st.button("🗑️", key=f"remove_doc_{i}_{doc}"):
                        # Remove the document
                        st.session_state.rag_sources.pop(i)
                        # Reset document_loaded if no documents left
                        if len(st.session_state.rag_sources) == 0:
                            st.session_state.document_loaded = False
                            st.session_state.document_stats = None
                            st.session_state.rag_pipeline = None
                        st.rerun()
        else:
            st.write("No documents in database")
    

    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Your message"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # RAG response
            answer, source_docs = handle_user_query(prompt)
            st.write(answer)
            
            # Show source documents if available
            if source_docs and isinstance(source_docs, list) and len(source_docs) > 0:
                with st.expander("📄 View Source Documents"):
                    for i, doc in enumerate(source_docs[:3]):  # Show top 3 sources
                        st.markdown(f"**Source {i+1}:**")
                        st.markdown(f'{doc.page_content[:300]}{"..." if len(doc.page_content) > 300 else ""}')
                        st.divider()
    
    # System information
    if st.session_state.rag_pipeline:
        system_info = st.session_state.rag_pipeline.get_system_info()
        render_system_info(system_info)


if __name__ == "__main__":
    main() 