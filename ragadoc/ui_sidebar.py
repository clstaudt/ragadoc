"""
Sidebar UI Components

Handles rendering of the sidebar including model selection, chat history,
and RAG configuration settings.
"""

import streamlit as st
from loguru import logger

from .ui_config import is_running_in_docker, get_ollama_base_url
from .ui_session import init_rag_system, switch_ollama_instance
from .config import (
    RAG_BACKENDS, DEFAULT_RAG_BACKEND,
    CHUNK_SIZE_RANGE, CHUNK_SIZE_STEP,
    CHUNK_OVERLAP_RANGE, CHUNK_OVERLAP_STEP,
    SIMILARITY_THRESHOLD_RANGE, SIMILARITY_THRESHOLD_STEP,
    TOP_K_RANGE, TOP_K_STEP
)


def _render_rag_status():
    """Render minimal RAG system status at the bottom of the sidebar"""
    if st.session_state.rag_system:
        backend_type = getattr(st.session_state.rag_system, 'backend_type', 'vector')
        backend_label = RAG_BACKENDS.get(backend_type, backend_type)
        st.caption(f"✅ {backend_label}")
        
        # Document count with clear option
        available_docs = st.session_state.rag_system.get_available_documents()
        if available_docs:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.caption(f"📊 {len(available_docs)} document(s) in system")
            with col2:
                if st.button("🗑️", key="clear_all_docs", help="Clear all stored documents"):
                    st.session_state.rag_system.clear_all_documents()
                    st.session_state.chat_manager.clear_all_chats()
                    st.rerun()
        
        # Current document stats for this chat (backend-aware)
        current_chat = st.session_state.chat_manager.get_current_chat()
        if current_chat and current_chat.rag_processed:
            stats = current_chat.rag_stats or {}
            if backend_type == "pageindex":
                st.caption(f"🌳 Current: {stats.get('total_nodes', 0)} tree nodes")
            else:
                st.caption(f"📄 Current: {stats.get('total_chunks', 0)} chunks")
    else:
        # RAG system not available - show retry option
        st.caption("❌ RAG System Not Available")
        if st.button("🔄 Retry", key="retry_rag_init", help="Retry RAG initialization"):
            if "rag_system" in st.session_state:
                del st.session_state["rag_system"]
            init_rag_system()
            st.rerun()


def render_sidebar():
    """Render the sidebar with chat history and RAG configuration"""
    with st.sidebar:
        st.header("Settings")
        
        # Ollama Instance Selection (only if multiple instances)
        instances = st.session_state.ollama_instances
        if len(instances) > 1:
            names = [i["name"] for i in instances]
            current = st.session_state.selected_ollama_instance
            idx = names.index(current) if current in names else 0
            
            selected = st.selectbox("🌐 Ollama Instance:", names, index=idx, key="ollama_instance_selector")
            if selected != current:
                logger.info(f"Instance changed: {current} -> {selected}")
                switch_ollama_instance(selected)
                st.rerun()
        
        # Debug: show which URL we're using
        current_url = st.session_state.model_manager.ollama_base_url
        st.caption(f"📡 {current_url}")
        
        # Key includes instance name to force widget refresh when instance changes
        instance_key = st.session_state.selected_ollama_instance
        
        # Chat Model Selection
        try:
            available_models = st.session_state.model_manager.get_available_models()
            # Filter: chat models exclude embedding models
            chat_models = [m for m in available_models if not any(x in m.lower() for x in ['embed', 'minilm'])]
            # Filter: embedding models only
            embedding_models = [m for m in available_models if any(x in m.lower() for x in ['embed', 'minilm'])]
            
            if chat_models:
                previous_model = st.session_state.selected_model
                st.session_state.selected_model = st.selectbox(
                    "🤖 Chat Model:",
                    chat_models,
                    index=0 if not st.session_state.selected_model else 
                          (chat_models.index(st.session_state.selected_model) 
                           if st.session_state.selected_model in chat_models else 0),
                    key=f"model_selector_{instance_key}",
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
                st.error("❌ No chat models found")
                st.caption("Please ensure Ollama has chat models installed")
                return
        except Exception as e:
            st.error(f"❌ Error connecting to Ollama: {e}")
            return
        
        # ── RAG Backend Selection ──────────────────────────────────
        # Disabled once a document is uploaded in the current chat
        current_chat = st.session_state.chat_manager.get_current_chat()
        has_document = current_chat and current_chat.document_text

        backend_options = list(RAG_BACKENDS.keys())
        current_backend = st.session_state.get("rag_backend_type", DEFAULT_RAG_BACKEND)
        current_backend_idx = backend_options.index(current_backend) if current_backend in backend_options else 0

        selected_backend = st.selectbox(
            "📐 RAG Approach:",
            backend_options,
            index=current_backend_idx,
            format_func=lambda x: RAG_BACKENDS[x],
            disabled=bool(has_document),
            key=f"backend_selector_{instance_key}",
            help="Choose retrieval approach before uploading a document. "
                 "Vector RAG uses embeddings + similarity search. "
                 "PageIndex RAG uses tree-based reasoning (fully local)."
        )

        if selected_backend != st.session_state.rag_backend_type:
            st.session_state.rag_backend_type = selected_backend
            if "rag_system" in st.session_state:
                del st.session_state["rag_system"]
            init_rag_system()
            st.rerun()

        is_pageindex = st.session_state.rag_backend_type == "pageindex"

        # ── Vector RAG: Embedding Model Selection ─────────────────
        if not is_pageindex:
            if not embedding_models:
                embedding_models = ["nomic-embed-text"]  # fallback
            
            previous_embedding_model = st.session_state.rag_config["embedding_model"]
            current_embed_idx = 0
            if previous_embedding_model in embedding_models:
                current_embed_idx = embedding_models.index(previous_embedding_model)
            
            embedding_model = st.selectbox(
                "🔍 Embedding Model:", 
                embedding_models,
                index=current_embed_idx,
                key=f"embedding_selector_{instance_key}",
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
            if is_pageindex:
                # ── PageIndex: no chunk/similarity settings ───────
                with st.expander("🌳 PageIndex Settings", expanded=False):
                    st.caption("PageIndex uses the selected chat model for tree "
                               "generation and reasoning-based retrieval. No "
                               "embeddings, chunks, or similarity thresholds.")
                    st.caption("Tree generation may take several minutes for "
                               "long documents, depending on model speed.")
            else:
                # ── Vector RAG: existing settings ─────────────────
                with st.expander("🔍 RAG Settings", expanded=False):
                    # === Query-time parameters (apply immediately) ===
                    st.caption("**Query Settings** — apply immediately")
                    similarity_threshold = st.slider(
                        "Similarity Threshold", 
                        SIMILARITY_THRESHOLD_RANGE[0], SIMILARITY_THRESHOLD_RANGE[1], 
                        st.session_state.rag_config["similarity_threshold"], 
                        SIMILARITY_THRESHOLD_STEP,
                        help="Minimum similarity score for retrieved chunks"
                    )
                    top_k = st.slider(
                        "Max Retrieved Chunks", 
                        TOP_K_RANGE[0], TOP_K_RANGE[1], 
                        st.session_state.rag_config["top_k"], 
                        TOP_K_STEP,
                        help="Maximum number of chunks to retrieve per query"
                    )
                    
                    st.divider()
                    
                    # === Document-time parameters (require re-indexing) ===
                    st.caption("**Indexing Settings** — require document re-upload")
                    chunk_size = st.slider(
                        "Chunk Size (tokens)", 
                        CHUNK_SIZE_RANGE[0], CHUNK_SIZE_RANGE[1], 
                        st.session_state.rag_config["chunk_size"], 
                        CHUNK_SIZE_STEP,
                        help="Size of text chunks when indexing documents"
                    )
                    chunk_overlap = st.slider(
                        "Chunk Overlap (tokens)", 
                        CHUNK_OVERLAP_RANGE[0], CHUNK_OVERLAP_RANGE[1], 
                        st.session_state.rag_config["chunk_overlap"], 
                        CHUNK_OVERLAP_STEP,
                        help="Overlap between consecutive chunks"
                    )
                    
                    # Update configuration
                    new_config = {
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap, 
                        "similarity_threshold": similarity_threshold,
                        "top_k": top_k,
                        "embedding_model": st.session_state.rag_config["embedding_model"]
                    }
                    
                    # Check if indexing settings changed (requires re-indexing)
                    indexing_changed = (
                        new_config["chunk_size"] != st.session_state.rag_config["chunk_size"] or
                        new_config["chunk_overlap"] != st.session_state.rag_config["chunk_overlap"]
                    )
                    
                    if new_config != st.session_state.rag_config:
                        st.session_state.rag_config = new_config
                        if indexing_changed:
                            st.warning("↑ Re-upload document to apply indexing changes")
        
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
        
        # Minimal RAG status at the bottom of sidebar
        st.divider()
        _render_rag_status() 