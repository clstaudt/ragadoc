"""
Sidebar UI Components

Handles rendering of the sidebar including model selection, chat history,
and RAG configuration settings.
"""

import streamlit as st
from loguru import logger

from .ui_config import is_running_in_docker, get_ollama_base_url
from .ui_session import init_rag_system
from .config import (
    CHUNK_SIZE_RANGE, CHUNK_SIZE_STEP,
    CHUNK_OVERLAP_RANGE, CHUNK_OVERLAP_STEP,
    SIMILARITY_THRESHOLD_RANGE, SIMILARITY_THRESHOLD_STEP,
    TOP_K_RANGE, TOP_K_STEP
)


def render_sidebar():
    """Render the sidebar with chat history and RAG configuration"""
    with st.sidebar:
        st.header("Settings")
        
        # Chat Model Selection
        try:
            available_models = st.session_state.model_manager.get_available_models()
            if available_models:
                previous_model = st.session_state.selected_model
                st.session_state.selected_model = st.selectbox(
                    "🤖 Chat Model:",
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
        
        # Embedding Model Selection
        previous_embedding_model = st.session_state.rag_config["embedding_model"]
        embedding_model = st.selectbox(
            "🔍 Embedding Model:", 
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
        
        # Expert Mode Toggle
        expert_mode = st.toggle(
            "🔧 Expert Mode", 
            value=st.session_state.get('expert_mode', False),
            help="Show advanced RAG configuration settings"
        )
        st.session_state.expert_mode = expert_mode
        
        # RAG Configuration (only shown in expert mode)
        if expert_mode:
            with st.expander("🔍 RAG Settings", expanded=False):
                # RAG parameters (excluding embedding model which is now global)
                chunk_size = st.slider("Chunk Size (tokens)", CHUNK_SIZE_RANGE[0], CHUNK_SIZE_RANGE[1], st.session_state.rag_config["chunk_size"], CHUNK_SIZE_STEP)
                chunk_overlap = st.slider("Chunk Overlap (tokens)", CHUNK_OVERLAP_RANGE[0], CHUNK_OVERLAP_RANGE[1], st.session_state.rag_config["chunk_overlap"], CHUNK_OVERLAP_STEP)
                similarity_threshold = st.slider("Similarity Threshold", SIMILARITY_THRESHOLD_RANGE[0], SIMILARITY_THRESHOLD_RANGE[1], st.session_state.rag_config["similarity_threshold"], SIMILARITY_THRESHOLD_STEP)
                top_k = st.slider("Max Retrieved Chunks", TOP_K_RANGE[0], TOP_K_RANGE[1], st.session_state.rag_config["top_k"], TOP_K_STEP)
                
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
        
        st.header("Chat History")
        
        # New chat button - prominent golden button
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            # Stop any ongoing generation before creating new chat
            if st.session_state.get('generating', False):
                st.session_state.stop_generation = True
                st.session_state.generating = False
                logger.info("Stopped ongoing generation due to new chat creation")
            
            st.session_state.chat_manager.create_new_chat()
            st.rerun()
        
        st.divider()
        
        # Chat history using container with custom styling
        sorted_chats = st.session_state.chat_manager.get_sorted_chats()
        if sorted_chats:
            for chat_id, chat_session in sorted_chats:
                is_current = chat_id == st.session_state.chat_manager.current_chat_id
                
                # Create a container for each chat item
                chat_container = st.container()
                with chat_container:
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        # All chats use the same emoji, different styling for selection
                        if is_current:
                            # Current chat - highlighted with primary styling
                            st.button(
                                f"💬 {chat_session.title}",
                                key=f"chat-{chat_id}",
                                use_container_width=True,
                                type="primary",
                                help="Current chat",
                                disabled=True  # Disabled to show it's selected
                            )
                        else:
                            # Inactive chat - secondary styling
                            if st.button(
                                f"💬 {chat_session.title}",
                                key=f"chat-{chat_id}",
                                use_container_width=True,
                                type="secondary",
                                help="Click to switch to this chat"
                            ):
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
                        if st.button("×", key=f"del-{chat_id}", help="Delete", type="secondary"):
                            # Stop any ongoing generation before deleting chat
                            if st.session_state.get('generating', False):
                                st.session_state.stop_generation = True
                                st.session_state.generating = False
                                logger.info("Stopped ongoing generation due to chat deletion")
                            
                            st.session_state.chat_manager.delete_chat(chat_id)
                            st.rerun()
        else:
            st.write("*No chat history yet*") 