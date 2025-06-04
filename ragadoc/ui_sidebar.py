"""
Sidebar UI Components

Handles rendering of the sidebar including model selection, chat history,
and RAG configuration settings.
"""

import streamlit as st
from loguru import logger

from .ui_config import is_running_in_docker, get_ollama_base_url
from .ui_session import init_rag_system


def render_sidebar():
    """Render the sidebar with chat history and RAG configuration"""
    with st.sidebar:
        st.header("Settings")
        
        # Global Model Selection
        try:
            available_models = st.session_state.model_manager.get_available_models()
            if available_models:
                previous_model = st.session_state.selected_model
                st.session_state.selected_model = st.selectbox(
                    "🤖 Chat Model (Global):",
                    available_models,
                    index=0 if not st.session_state.selected_model else 
                          (available_models.index(st.session_state.selected_model) 
                           if st.session_state.selected_model in available_models else 0),
                    key="global_model_selector",
                    help="This model will be used for all chats"
                )
                
                # Reinitialize RAG system if model changed
                if previous_model != st.session_state.selected_model and previous_model is not None:
                    logger.info(f"Model changed from {previous_model} to {st.session_state.selected_model}")
                    if "rag_system" in st.session_state:
                        del st.session_state["rag_system"]
                        init_rag_system()
                    st.rerun()
            else:
                st.error("❌ No Ollama models found")
                st.caption("Please ensure Ollama is running")
                return
        except Exception as e:
            st.error(f"❌ Error connecting to Ollama: {e}")
            return
        
        # Global Embedding Model Selection
        previous_embedding_model = st.session_state.rag_config["embedding_model"]
        embedding_model = st.selectbox(
            "🔍 Embedding Model (Global):", 
            ["nomic-embed-text", "mxbai-embed-large", "all-minilm"], 
            index=0 if st.session_state.rag_config["embedding_model"] == "nomic-embed-text" else 
                  (1 if st.session_state.rag_config["embedding_model"] == "mxbai-embed-large" else 2),
            key="global_embedding_selector",
            help="This model will be used for document embedding and semantic search"
        )
        
        # Update embedding model if changed
        if embedding_model != previous_embedding_model:
            st.session_state.rag_config["embedding_model"] = embedding_model
            logger.info(f"Embedding model changed from {previous_embedding_model} to {embedding_model}")
            if "rag_system" in st.session_state:
                del st.session_state["rag_system"]
                init_rag_system()
            st.info("⚠️ Embedding model changed. Upload a new document to apply changes.")
            st.rerun()
        
        st.divider()
        
        st.header("Chat History")
        
        # Connection info
        ollama_base_url = get_ollama_base_url()
        if is_running_in_docker():
            st.caption(f"🐳 Docker → {ollama_base_url}")
        else:
            st.caption(f"💻 Direct → localhost:11434")
        
        # New chat button
        if st.button("New Chat", use_container_width=True, type="primary"):
            # Stop any ongoing generation before creating new chat
            if st.session_state.get('generating', False):
                st.session_state.stop_generation = True
                st.session_state.generating = False
                logger.info("Stopped ongoing generation due to new chat creation")
            
            st.session_state.chat_manager.create_new_chat()
            st.rerun()
        
        st.divider()
        
        # RAG Configuration
        with st.expander("🔍 RAG Settings", expanded=False):
            st.info("🔍 **Smart Retrieval**: The system first finds ALL chunks above the similarity threshold, then limits to the max number.")
            
            # RAG parameters (excluding embedding model which is now global)
            chunk_size = st.slider("Chunk Size (tokens)", 32, 1024, st.session_state.rag_config["chunk_size"], 64)
            chunk_overlap = st.slider("Chunk Overlap (tokens)", 0, 200, st.session_state.rag_config["chunk_overlap"], 10)
            similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, st.session_state.rag_config["similarity_threshold"], 0.05)
            top_k = st.slider("Max Retrieved Chunks", 1, 20, st.session_state.rag_config["top_k"], 1)
            
            # Update configuration if changed (excluding embedding model)
            new_config = {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap, 
                "similarity_threshold": similarity_threshold,
                "top_k": top_k,
                "embedding_model": st.session_state.rag_config["embedding_model"]  # Keep current embedding model
            }
            
            if new_config != st.session_state.rag_config:
                st.session_state.rag_config = new_config
                st.info("⚠️ RAG settings changed. Upload a new document to apply changes.")
            
            # RAG system status
            if st.session_state.rag_system:
                st.success("✅ RAG System Ready")
                
                # Show available documents
                available_docs = st.session_state.rag_system.get_available_documents()
                if available_docs:
                    st.info(f"📊 {len(available_docs)} document(s) in system")
                    
                    # Show current document for current chat
                    current_chat = st.session_state.chat_manager.get_current_chat()
                    if current_chat and current_chat.rag_processed:
                        stats = current_chat.rag_stats or {}
                        st.info(f"📄 Current: {stats.get('total_chunks', 0)} chunks")
                    else:
                        st.warning("📄 No document in current chat")
                else:
                    st.warning("📄 No documents processed yet")
            else:
                st.error("❌ RAG System Not Available")
                if st.button("🔄 Retry RAG Initialization"):
                    if "rag_system" in st.session_state:
                        del st.session_state["rag_system"]
                    init_rag_system()
                    st.rerun()
        
        st.divider()
        
        # Chat history
        sorted_chats = st.session_state.chat_manager.get_sorted_chats()
        if sorted_chats:
            for chat_id, chat_session in sorted_chats:
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    is_current = chat_id == st.session_state.chat_manager.current_chat_id
                    button_type = "primary" if is_current else "secondary"
                    
                    if st.button(chat_session.title, key=f"chat-{chat_id}", 
                               use_container_width=True, type=button_type):
                        # Stop any ongoing generation before switching
                        if st.session_state.get('generating', False):
                            st.session_state.stop_generation = True
                            st.session_state.generating = False
                            logger.info("Stopped ongoing generation due to chat switch")
                        
                        # Switch to the chat
                        st.session_state.chat_manager.switch_to_chat(chat_id)
                        
                        # Load the appropriate document for this chat if it has one
                        if chat_session.document_id and st.session_state.rag_system:
                            try:
                                success = st.session_state.rag_system.load_document(chat_session.document_id)
                                if success:
                                    logger.info(f"Loaded document {chat_session.document_id} for chat {chat_id}")
                                else:
                                    logger.warning(f"Could not load document {chat_session.document_id} for chat {chat_id}")
                            except Exception as e:
                                logger.error(f"Error loading document for chat {chat_id}: {e}")
                        
                        st.rerun()
                
                with col2:
                    if st.button("×", key=f"del-{chat_id}", help="Delete"):
                        # Stop any ongoing generation before deleting chat
                        if st.session_state.get('generating', False):
                            st.session_state.stop_generation = True
                            st.session_state.generating = False
                            logger.info("Stopped ongoing generation due to chat deletion")
                        
                        st.session_state.chat_manager.delete_chat(chat_id)
                        st.rerun() 