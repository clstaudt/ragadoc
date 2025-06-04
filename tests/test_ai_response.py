"""
Simple tests for AI response generation - focuses on core logic
"""
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app
from ragadoc.llm_interface import PromptBuilder
from ragadoc.model_manager import ModelManager

# Known available models
EMBEDDING_MODEL = "nomic-embed-text:latest"
LLM_MODEL = "olmo2:13b"


def is_ollama_available():
    """Check if Ollama is available"""
    try:
        import ollama
        ollama.list()
        return True
    except:
        return False


@pytest.mark.skipif(not is_ollama_available(), reason="Ollama not available")
class TestModelManager:
    """Test ModelManager with real Ollama connection"""
    
    def test_get_available_models_real(self):
        """Test getting real model list from Ollama"""
        model_manager = ModelManager()
        models = model_manager.get_available_models()
        
        assert isinstance(models, list)
        assert len(models) > 0  # Should have at least one model
        
        # Should contain our known models
        assert EMBEDDING_MODEL in models
        assert LLM_MODEL in models
    
    def test_get_model_info_real(self):
        """Test getting model info for known model"""
        model_manager = ModelManager()
        info = model_manager.get_model_info(LLM_MODEL)
        
        # The function returns various types, just check it doesn't crash
        assert info is not None
    
    def test_get_context_length_real(self):
        """Test getting context length for known model"""
        model_manager = ModelManager()
        context_length = model_manager.get_context_length(LLM_MODEL)
        
        # May return None for unknown models, just check it doesn't crash
        assert context_length is None or isinstance(context_length, int)


class TestContextChecker:
    """Test context checking functionality"""
    
    def test_estimate_token_count(self):
        """Test token count estimation"""
        test_text = "This is a test sentence with several words."
        
        from ragadoc.model_manager import ContextChecker
        token_count = ContextChecker.estimate_token_count(test_text)
        
        assert isinstance(token_count, int)
        assert token_count > 0
        assert token_count < len(test_text)  # Should be less than character count
    
    def test_check_document_fits_context_basic(self):
        """Test document context checking basic functionality"""
        short_text = "Short text."
        
        from ragadoc.model_manager import ContextChecker
        model_manager = ModelManager()
        result = ContextChecker.check_document_fits_context(
            short_text, model_manager, LLM_MODEL, "Test prompt"
        )
        
        # Function may return tuple or dict depending on model support
        assert result is not None
        # Just verify it doesn't crash and returns something
        assert len(result) > 0


class TestEnvironmentDetection:
    """Test environment detection functions"""
    
    def test_docker_detection(self):
        """Test Docker environment detection"""
        from ragadoc.ui_config import is_running_in_docker
        result = is_running_in_docker()
        
        assert isinstance(result, bool)
    
    def test_ollama_url_configuration(self):
        """Test Ollama URL is properly configured"""
        from ragadoc.ui_config import get_ollama_base_url
        url = get_ollama_base_url()
        assert url.startswith('http')
        assert ':11434' in url 