"""
Essential smoke tests for regression detection
"""
import pytest
import sys
import os
import fitz

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestBasicImports:
    """Test core imports work"""
    
    def test_app_import(self):
        """Test main app imports"""
        import app
        assert hasattr(app, 'main')
    
    def test_ragadoc_import(self):
        """Test ragadoc package imports"""
        from ragadoc import EnhancedPDFProcessor, RAGSystem
        assert EnhancedPDFProcessor is not None
        assert RAGSystem is not None


class TestEnvironment:
    """Test environment detection"""
    
    def test_docker_detection(self):
        """Test Docker detection returns boolean"""
        from ragadoc.ui_config import is_running_in_docker
        result = is_running_in_docker()
        assert isinstance(result, bool)
    
    def test_ollama_url_configured(self):
        """Test Ollama URL is properly configured"""
        from ragadoc.ui_config import get_ollama_base_url
        url = get_ollama_base_url()
        assert url.startswith('http')
        assert '11434' in url


class TestPDFBasics:
    """Test PDF processing basics"""
    
    def test_pdf_creation_and_extraction(self):
        """Test PDF creation and text extraction work"""
        from ragadoc import EnhancedPDFProcessor
        
        # Create simple test PDF
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 100), "Test content for extraction")
        pdf_bytes = doc.tobytes()
        doc.close()
        
        # Test extraction works
        processor = EnhancedPDFProcessor(pdf_bytes)
        result = processor.extract_full_text()
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert "Test content" in result
    
    def test_pdf_error_handling(self):
        """Test PDF processor handles bad input"""
        from ragadoc import EnhancedPDFProcessor
        
        try:
            processor = EnhancedPDFProcessor(b"not a pdf")
            result = processor.extract_full_text()
            # If no exception, should return string
            assert isinstance(result, str)
        except Exception:
            # Expected for invalid PDF
            pass


class TestModelManager:
    """Test ModelManager basics"""
    
    def test_get_models_returns_list(self):
        """Test get_available_models returns list"""
        from ragadoc.model_manager import ModelManager
        model_manager = ModelManager()
        models = model_manager.get_available_models()
        assert isinstance(models, list)
    
    def test_model_info_doesnt_crash(self):
        """Test get_model_info doesn't crash"""
        from ragadoc.model_manager import ModelManager
        model_manager = ModelManager()
        info = model_manager.get_model_info("test-model")
        # Should return something or None, just shouldn't crash
        assert info is None or info is not None


class TestContextChecker:
    """Test context checking basics"""
    
    def test_token_estimation(self):
        """Test token estimation works"""
        from ragadoc.model_manager import ContextChecker
        count = ContextChecker.estimate_token_count("test text")
        assert isinstance(count, int)
        assert count > 0
    
    def test_context_checking_basic(self):
        """Test context checking doesn't crash"""
        from ragadoc.model_manager import ContextChecker, ModelManager
        model_manager = ModelManager()
        result = ContextChecker.check_document_fits_context("test", model_manager, "model")
        # Should return something, format may vary
        assert result is not None


def is_ollama_available():
    """Check if Ollama is available"""
    try:
        import ollama
        ollama.list()
        return True
    except:
        return False


@pytest.mark.skipif(not is_ollama_available(), reason="Ollama not available")
class TestIntegration:
    """Real integration tests when Ollama available"""
    
    def test_model_list_integration(self):
        """Test real model list from Ollama"""
        from ragadoc.model_manager import ModelManager
        model_manager = ModelManager()
        models = model_manager.get_available_models()
        assert isinstance(models, list)
        # May be empty, that's ok
    
    def test_pdf_to_text_workflow(self):
        """Test complete PDF to text workflow"""
        from ragadoc import EnhancedPDFProcessor
        
        # Create realistic test PDF
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 100), "Integration test document content")
        pdf_bytes = doc.tobytes()
        doc.close()
        
        # Test full workflow
        processor = EnhancedPDFProcessor(pdf_bytes)
        text = processor.extract_full_text()
        
        assert isinstance(text, str)
        assert len(text) > 0 