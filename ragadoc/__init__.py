"""
Ragadoc - AI-powered PDF processing and highlighting system with RAG capabilities
"""

from .enhanced_pdf_processor import (
    EnhancedPDFProcessor,
)

from .rag_system import (
    RAGSystem,
    create_rag_system
)

from .model_manager import (
    ModelManager,
    ContextChecker
)

from .chat_manager import (
    ChatManager,
    ChatSession,
    ChatMessage
)

from .llm_interface import (
    LLMInterface,
    PromptBuilder,
    ReasoningParser
)

# UI Components
from .ui_config import (
    is_running_in_docker,
    get_ollama_base_url,
    setup_streamlit_config
)

from .ui_session import (
    init_session_state,
    init_rag_system
)

from .ui_sidebar import (
    render_sidebar
)

from .ui_document import (
    render_document_upload
)

from .ui_chat import (
    show_citations,
    generate_response_with_ui,
    render_chat_interface
)

__version__ = "0.1.0"
__all__ = [
    "EnhancedPDFProcessor",
    "RAGSystem",
    "create_rag_system",
    "ModelManager",
    "ContextChecker",
    "ChatManager",
    "ChatSession",
    "ChatMessage",
    "LLMInterface",
    "PromptBuilder",
    "ReasoningParser",
    # UI Components
    "is_running_in_docker",
    "get_ollama_base_url",
    "setup_streamlit_config",
    "init_session_state",
    "init_rag_system",
    "render_sidebar",
    "render_document_upload",
    "show_citations",
    "generate_response_with_ui",
    "render_chat_interface"
]  