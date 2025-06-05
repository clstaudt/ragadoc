"""
Fast integration tests for RAG system - uses specific known models
"""
import pytest
import sys
import os
import uuid
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragadoc import RAGSystem, create_rag_system
from ragadoc.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

# Known available models
EMBEDDING_MODEL = "nomic-embed-text:latest"
LLM_MODEL = "tinyllama:latest"


def is_ollama_available():
    """Check if Ollama is available"""
    try:
        import ollama
        ollama.list()
        return True
    except:
        return False


@pytest.fixture
def temp_rag_dir():
    """Create temporary directory for RAG system"""
    temp_dir = Path(tempfile.mkdtemp(prefix="test_rag_"))
    yield temp_dir
    # Cleanup after test
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def clean_rag_system(temp_rag_dir):
    """Create a clean RAG system for each test"""
    # Use unique storage location for each test
    rag = RAGSystem(
        embedding_model=EMBEDDING_MODEL,
        llm_model=LLM_MODEL,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        top_k=2
    )
    
    # Override storage directory to use temp location
    rag.storage_dir = temp_rag_dir
    rag.chroma_dir = temp_rag_dir / "chroma_db"
    
    # Re-setup ChromaDB with new location
    rag._setup_chroma_client()
    
    yield rag
    
    # Cleanup after test
    try:
        rag.clear_all_documents()
        rag.cleanup()
    except Exception as e:
        print(f"Cleanup warning: {e}")


@pytest.mark.skipif(not is_ollama_available(), reason="Ollama not available")
class TestRAGSystemFast:
    """Fast integration tests with minimal content"""
    
    def test_rag_initialization(self, clean_rag_system):
        """Test RAG system initializes quickly"""
        rag = clean_rag_system
        
        assert rag.embedding_model == EMBEDDING_MODEL
        assert rag.llm_model == LLM_MODEL
        assert rag.chunk_size == DEFAULT_CHUNK_SIZE
        assert rag.current_document_id is None
    
    def test_document_processing_minimal(self, clean_rag_system):
        """Test processing very small document"""
        rag = clean_rag_system
        
        # Minimal test content
        doc = "Cat sits. Dog runs. Birds fly in the sky."
        
        # Use unique document ID for each test
        doc_id = f"tiny_doc_{uuid.uuid4().hex[:8]}"
        result = rag.process_document(doc, doc_id)
        
        assert result['document_id'] == doc_id
        assert result['total_chunks'] > 0
        assert rag.current_document_id == doc_id
    
    def test_retrieval_works(self, clean_rag_system):
        """Test retrieval finds relevant chunks"""
        rag = clean_rag_system
        
        doc = "Python programming language. Fast code execution. Machine learning algorithms."
        doc_id = f"lang_doc_{uuid.uuid4().hex[:8]}"
        rag.process_document(doc, doc_id)
        
        # Test retrieval
        results = rag.get_retrieval_info("Python")
        
        assert isinstance(results, list)
        assert len(results) > 0
        # Should find something related to Python
        found_python = any('python' in chunk.get('text', '').lower() for chunk in results)
        assert found_python
    
    def test_end_to_end_query(self, clean_rag_system):
        """Test complete pipeline"""
        rag = clean_rag_system
        
        # Very simple document
        doc = "Bob likes cats. Cats are pets. Animals need care."
        doc_id = f"pet_doc_{uuid.uuid4().hex[:8]}"
        rag.process_document(doc, doc_id)
        
        # Simple query
        response = rag.query_document("What does Bob like?")
        
        assert isinstance(response, dict)
        assert 'response' in response
        assert isinstance(response['response'], str)
        assert len(response['response']) > 5  # Should generate something
    
    def test_document_clearing(self, clean_rag_system):
        """Test clearing documents"""
        rag = clean_rag_system
        
        doc_id = f"clear_me_{uuid.uuid4().hex[:8]}"
        rag.process_document("Test content for clearing documents", doc_id)
        assert rag.current_document_id == doc_id
        
        rag.clear_document(doc_id)
        
        # Should fail to query after clearing
        with pytest.raises(Exception):
            rag.query_document("test")


@pytest.mark.skipif(not is_ollama_available(), reason="Ollama not available")
class TestRAGErrorHandling:
    """Test error handling quickly"""
    
    def test_empty_document(self, clean_rag_system):
        """Test empty document handling - should process gracefully"""
        rag = clean_rag_system
        
        # Empty documents should be processed without error
        doc_id = f"empty_{uuid.uuid4().hex[:8]}"
        result = rag.process_document("", doc_id)
        
        # Should succeed but create minimal chunks
        assert result['document_id'] == doc_id
        assert result['total_chunks'] >= 1  # At least one empty chunk
        assert rag.current_document_id == doc_id
    
    def test_query_without_document(self, clean_rag_system):
        """Test querying without document"""
        rag = clean_rag_system
        
        with pytest.raises(Exception):
            rag.query_document("test")


@pytest.mark.skipif(not is_ollama_available(), reason="Ollama not available")
class TestRAGFactory:
    """Test factory function quickly"""
    
    def test_create_rag_system(self, temp_rag_dir):
        """Test factory creates working system"""
        rag = create_rag_system(
            embedding_model=EMBEDDING_MODEL,
            llm_model=LLM_MODEL,
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP
        )
        
        # Override storage directory
        rag.storage_dir = temp_rag_dir
        rag.chroma_dir = temp_rag_dir / "chroma_db"
        rag._setup_chroma_client()
        
        assert isinstance(rag, RAGSystem)
        assert rag.chunk_size == DEFAULT_CHUNK_SIZE
        
        # Quick functionality test
        doc_id = f"factory_test_{uuid.uuid4().hex[:8]}"
        rag.process_document("Quick test document for factory", doc_id)
        assert rag.current_document_id == doc_id
        
        # Cleanup
        rag.clear_all_documents()
        rag.cleanup() 