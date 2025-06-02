"""
Ragnarok - AI-powered PDF processing and highlighting system with RAG capabilities
"""

from .enhanced_pdf_processor import (
    EnhancedPDFProcessor,
    highlight_ai_referenced_text,
    process_pdf_with_highlighting
)

from .rag_system import (
    RAGSystem,
    create_rag_system
)

__version__ = "0.1.0"
__all__ = [
    "EnhancedPDFProcessor",
    "highlight_ai_referenced_text", 
    "process_pdf_with_highlighting",
    "RAGSystem",
    "create_rag_system"
]  