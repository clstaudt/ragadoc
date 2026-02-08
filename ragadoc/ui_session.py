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
from .ui_config import get_ollama_instances, is_running_in_docker
from .config import DEFAULT_RAG_CONFIG, DEFAULT_RAG_BACKEND


def init_session_state():
    """Initialize Streamlit session state with backend managers"""
    
    # Initialize Ollama instances
    if "ollama_instances" not in st.session_state:
        st.session_state.ollama_instances = get_ollama_instances()
    if "selected_ollama_instance" not in st.session_state:
        st.session_state.selected_ollama_instance = st.session_state.ollama_instances[0]["name"]
    
    # Get environment configuration
    ollama_base_url = next(
        (i["url"] for i in st.session_state.ollama_instances 
         if i["name"] == st.session_state.selected_ollama_instance),
        st.session_state.ollama_instances[0]["url"]
    )
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
    
    # RAG backend selection
    if "rag_backend_type" not in st.session_state:
        st.session_state.rag_backend_type = DEFAULT_RAG_BACKEND
    
    # RAG configuration
    if "rag_config" not in st.session_state:
        st.session_state.rag_config = DEFAULT_RAG_CONFIG.copy()
    
    # Initialize RAG system
    if "rag_system" not in st.session_state:
        init_rag_system()


def get_current_ollama_url() -> str:
    """Get the URL for the currently selected Ollama instance"""
    return next(
        (i["url"] for i in st.session_state.ollama_instances 
         if i["name"] == st.session_state.selected_ollama_instance),
        st.session_state.ollama_instances[0]["url"]
    )


def init_rag_system():
    """Initialize the RAG system based on the selected backend type"""
    backend_type = st.session_state.get("rag_backend_type", DEFAULT_RAG_BACKEND)

    if backend_type == "pageindex":
        _init_pageindex_rag_system()
    else:
        _init_vector_rag_system()


def _init_vector_rag_system():
    """Initialize the vector-based RAG system (ChromaDB + embeddings)"""
    try:
        ollama_base_url = get_current_ollama_url()
        
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
        logger.info("Vector RAG system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize vector RAG system: {e}")
        st.session_state.rag_system = None


def _init_pageindex_rag_system():
    """Initialize the PageIndex tree-based RAG system (fully local via Ollama)"""
    try:
        from .pageindex_rag import PageIndexRAGSystem

        ollama_base_url = get_current_ollama_url()

        # Determine which LLM model to use for tree generation and search
        llm_model = st.session_state.selected_model
        if not llm_model:
            available_models = st.session_state.model_manager.get_available_models()
            chat_models = [m for m in available_models
                          if not any(embed in m.lower() for embed in ['embed', 'minilm'])]
            if chat_models:
                llm_model = chat_models[0]

        if not llm_model:
            logger.error("No chat model available for PageIndex RAG system")
            st.session_state.rag_system = None
            return

        st.session_state.rag_system = PageIndexRAGSystem(
            ollama_base_url=ollama_base_url,
            llm_model=llm_model,
        )
        logger.info(f"PageIndex RAG system initialized with model: {llm_model}")
    except ImportError:
        logger.error("PageIndex package not installed. Install with: pip install pageindex")
        st.session_state.rag_system = None
    except Exception as e:
        logger.error(f"Failed to initialize PageIndex RAG system: {e}")
        st.session_state.rag_system = None


def switch_ollama_instance(instance_name: str):
    """Switch to a different Ollama instance"""
    logger.info(f"switch_ollama_instance called with: {instance_name}")
    logger.info(f"Available instances: {st.session_state.ollama_instances}")
    
    url = next((i["url"] for i in st.session_state.ollama_instances if i["name"] == instance_name), None)
    if not url:
        logger.error(f"No URL found for instance: {instance_name}")
        return
    
    logger.info(f"Setting model_manager URL to: {url}")
    st.session_state.selected_ollama_instance = instance_name
    in_docker = is_running_in_docker()
    
    st.session_state.model_manager = ModelManager(url, in_docker)
    st.session_state.llm_interface = LLMInterface(url, in_docker)
    st.session_state.selected_model = None
    
    if "rag_system" in st.session_state:
        del st.session_state["rag_system"]
    init_rag_system()
    
    logger.info(f"Switched to Ollama instance: {instance_name} ({url})") 