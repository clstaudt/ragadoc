"""
Session State Management

Handles initialization and management of Streamlit session state including
backend managers and UI state variables.
"""

import streamlit as st
from loguru import logger

from .model_manager import ModelManager
from .chat_manager import ChatManager
from .llm_interface import LLMInterface
from .rag_system import create_rag_system
from .ui_config import get_ollama_base_url, is_running_in_docker
from .config import DEFAULT_RAG_CONFIG


def init_session_state():
    """Initialize Streamlit session state with backend managers"""
    
    # Get environment configuration
    ollama_base_url = get_ollama_base_url()
    in_docker = is_running_in_docker()
    
    # Initialize backend managers
    if "model_manager" not in st.session_state:
        st.session_state.model_manager = ModelManager(ollama_base_url, in_docker)
    
    if "chat_manager" not in st.session_state:
        st.session_state.chat_manager = ChatManager()
        # Create first chat
        st.session_state.chat_manager.create_new_chat()
    
    if "llm_interface" not in st.session_state:
        st.session_state.llm_interface = LLMInterface(ollama_base_url, in_docker)
    
    # UI state
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None
    
    if "generating" not in st.session_state:
        st.session_state.generating = False
    
    if "stop_generation" not in st.session_state:
        st.session_state.stop_generation = False
    
    # RAG configuration
    if "rag_config" not in st.session_state:
        st.session_state.rag_config = DEFAULT_RAG_CONFIG.copy()
    
    # Initialize RAG system
    if "rag_system" not in st.session_state:
        init_rag_system()


def init_rag_system():
    """Initialize the RAG system"""
    try:
        ollama_base_url = get_ollama_base_url()
        
        # Get available models for embedding model check
        available_models = st.session_state.model_manager.get_available_models()
        embedding_model = st.session_state.rag_config["embedding_model"]
        
        # Check if we need to add :latest suffix
        if embedding_model not in available_models:
            embedding_model_with_suffix = f"{embedding_model}:latest"
            if embedding_model_with_suffix in available_models:
                embedding_model = embedding_model_with_suffix
                logger.info(f"Using embedding model: {embedding_model}")
        
        # Create RAG system with current configuration
        rag_config = st.session_state.rag_config.copy()
        rag_config["embedding_model"] = embedding_model
        
        # Use the selected model from the UI
        if st.session_state.selected_model:
            rag_config["llm_model"] = st.session_state.selected_model
        else:
            # Try to get the first available model
            if available_models:
                llm_models = [m for m in available_models 
                             if not any(embed in m.lower() for embed in ['embed', 'minilm'])]
                if llm_models:
                    rag_config["llm_model"] = llm_models[0]
        
        st.session_state.rag_system = create_rag_system(
            ollama_base_url=ollama_base_url,
            **rag_config
        )
        logger.info("RAG system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        st.session_state.rag_system = None 