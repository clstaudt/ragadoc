"""
Configuration constants for Ragadoc

This module contains all default configuration values used throughout the application.
"""

# RAG Backend Options
RAG_BACKENDS = {
    "vector": "Vector RAG (Embeddings + ChromaDB)",
    "pageindex": "PageIndex RAG (Tree-based Reasoning)",
}
DEFAULT_RAG_BACKEND = "vector"

# Default RAG Configuration (vector backend)
DEFAULT_RAG_CONFIG = {
    "chunk_size": 128,
    "chunk_overlap": 64,
    "similarity_threshold": 0.7,
    "top_k": 10,
    "embedding_model": "nomic-embed-text",
    "llm_model": None
}

# RAG System Constructor Defaults
DEFAULT_CHUNK_SIZE = DEFAULT_RAG_CONFIG["chunk_size"]
DEFAULT_CHUNK_OVERLAP = DEFAULT_RAG_CONFIG["chunk_overlap"]
DEFAULT_SIMILARITY_THRESHOLD = DEFAULT_RAG_CONFIG["similarity_threshold"]
DEFAULT_TOP_K = DEFAULT_RAG_CONFIG["top_k"]
DEFAULT_EMBEDDING_MODEL = DEFAULT_RAG_CONFIG["embedding_model"]

# UI Slider Configuration
CHUNK_SIZE_RANGE = (32, 1024)
CHUNK_SIZE_STEP = 64
CHUNK_OVERLAP_RANGE = (0, 200)
CHUNK_OVERLAP_STEP = 10
SIMILARITY_THRESHOLD_RANGE = (0.0, 1.0)
SIMILARITY_THRESHOLD_STEP = 0.05
TOP_K_RANGE = (1, 20)
TOP_K_STEP = 1 