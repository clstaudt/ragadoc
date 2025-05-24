"""
Ragnarok - AI-powered PDF processing and highlighting system
"""

from .enhanced_pdf_processor import (
    EnhancedPDFProcessor,
    highlight_ai_referenced_text,
    process_pdf_with_highlighting
)

__version__ = "0.1.0"
__all__ = [
    "EnhancedPDFProcessor",
    "highlight_ai_referenced_text", 
    "process_pdf_with_highlighting"
]  