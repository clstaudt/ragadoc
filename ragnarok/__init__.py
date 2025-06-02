"""
Ragnarok - AI-powered PDF processing and highlighting system with RAG capabilities
"""

from .enhanced_pdf_processor import (
    EnhancedPDFProcessor,
)

from .rag_system import (
    RAGSystem,
    create_rag_system
)

__version__ = "0.1.0"
__all__ = [
    "EnhancedPDFProcessor",
    "RAGSystem",
    "create_rag_system"
]  