import streamlit as st
import os
import tempfile
import logging
from dotenv import load_dotenv

import streamlit as st
import os
import dotenv
import uuid

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from rag_methods import stream_llm_response, stream_llm_rag_response, load_doc_to_db

dotenv.load_dotenv()

MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
]

st.set_page_config(
    page_title="Large Document Interaction RAG System",
    page_icon=":book: 📚",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- Header ---
st.html("""<h2 style="text-align: center;">📚🔍 <i> Do your LLM even RAG bro? </i> 🤖💬</h2>""")


# --- Initial Setup ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there! How can I assist you today?"}
]
    
cols0 = st.columns(2)
with cols0[0]:
    is_vector_db_loaded = ("vector_db" in st.session_state and st.session_state.vector_db is not None)            st.toggle(
        "Use RAG", 
        value=is_vector_db_loaded, 
        key="use_rag", 
        disabled=not is_vector_db_loaded,
    )

with cols0[1]:
    st.button("Clear Chat", on_click=lambda: st.session_state.messages.clear(), type="primary")

st.header("RAG Sources:")
            
# File upload input for RAG with documents
st.file_uploader(
    "📄 Upload a document", 
    type=["pdf", "txt", "docx", "md"],
    accept_multiple_files=True,
    on_change=load_doc_to_db,
    key="rag_docs",
)

with st.expander(f"📚 Documents in DB ({0 if not is_vector_db_loaded else len(st.session_state.rag_sources)})"):
    st.write([] if not is_vector_db_loaded else [source for source in st.session_state.rag_sources])

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Your message"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        messages = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in st.session_state.messages]

        if not st.session_state.use_rag:
            st.write_stream(stream_llm_response(llm_stream, messages))
        else:
            st.write_stream(stream_llm_rag_response(llm_stream, messages))



