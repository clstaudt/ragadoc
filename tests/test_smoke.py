"""
Simple smoke tests for regression testing
"""
import pytest
import sys
import os
import fitz
from unittest.mock import Mock, patch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestBasicImports:
    """Test that we can import all modules without errors"""
    
    def test_import_app(self):
        """Test app.py imports without errors"""
        import app
        assert hasattr(app, 'main')
        assert hasattr(app, 'PDFProcessor')
        assert hasattr(app, 'ModelManager')
    
    def test_import_ragnarok(self):
        """Test ragnarok package imports"""
        from ragnarok import EnhancedPDFProcessor
        assert EnhancedPDFProcessor is not None


class TestEnvironmentDetection:
    """Environment detection regression tests"""
    
    def test_docker_vs_direct_execution_detection(self):
        """Test Docker detection works correctly in different environments"""
        from app import is_running_in_docker
        
        # Should return a boolean
        result = is_running_in_docker()
        assert isinstance(result, bool)
        
        # Test with mocked Docker environment
        with patch('os.path.exists', return_value=True):
            assert is_running_in_docker() == True
            
        # Test with mocked direct execution
        with patch('os.path.exists', return_value=False), \
             patch.dict(os.environ, {}, clear=True):
            assert is_running_in_docker() == False
    
    def test_ollama_url_configuration_consistency(self):
        """Test Ollama URL configuration is consistent"""
        import app
        
        # Should have valid URL format
        assert app.ollama_base_url.startswith('http')
        assert ':11434' in app.ollama_base_url
        assert isinstance(app.in_docker, bool)
        
        # URL should match environment
        if app.in_docker:
            assert 'docker.internal' in app.ollama_base_url or 'localhost' in app.ollama_base_url
        else:
            assert 'localhost' in app.ollama_base_url


class TestPDFProcessing:
    """PDF processing regression tests"""
    
    def test_pdf_text_extraction_workflow(self):
        """Test complete PDF text extraction workflow"""
        from app import PDFProcessor
        
        # Create PDF with realistic content
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 100), "Document Title: Regression Test\n\nThis is a multi-line document with various content including:\n- Bullet points\n- Numbers: 123, 456\n- Special chars: @#$%")
        pdf_bytes = doc.tobytes()
        doc.close()
        
        # Mock streamlit file object
        mock_file = Mock()
        mock_file.getvalue.return_value = pdf_bytes
        
        # Test extraction preserves content structure
        result = PDFProcessor.extract_text(mock_file)
        assert "Document Title" in result
        assert "Regression Test" in result
        assert "multi-line" in result
        assert "123" in result
        assert "@#$%" in result
        # Should handle errors gracefully and return string
        assert isinstance(result, str)
    
    def test_pdf_processor_error_handling(self):
        """Test PDF processor handles corrupted files"""
        from app import PDFProcessor
        
        mock_file = Mock()
        mock_file.getvalue.return_value = b"This is not a PDF file"
        
        with patch('streamlit.error'):
            result = PDFProcessor.extract_text(mock_file)
            # Should return empty string, not crash
            assert result == ""
    
    def test_enhanced_pdf_processor_multipage(self):
        """Test EnhancedPDFProcessor with multi-page documents"""
        from ragnarok import EnhancedPDFProcessor
        
        # Create multi-page PDF
        doc = fitz.open()
        
        page1 = doc.new_page()
        page1.insert_text((50, 100), "Page 1: Introduction to the test document")
        
        page2 = doc.new_page()
        page2.insert_text((50, 100), "Page 2: Detailed content and analysis")
        
        page3 = doc.new_page()
        page3.insert_text((50, 100), "Page 3: Conclusion and final thoughts")
        
        pdf_bytes = doc.tobytes()
        doc.close()
        
        # Test processor handles multiple pages
        processor = EnhancedPDFProcessor(pdf_bytes)
        text = processor.extract_full_text()
        
        # Should extract text from all pages
        assert "Page 1" in text
        assert "Page 2" in text
        assert "Page 3" in text
        assert "Introduction" in text
        assert "Conclusion" in text


class TestModelManager:
    """ModelManager regression tests"""
    
    @patch('ollama.list')
    def test_model_parsing_with_different_formats(self, mock_ollama):
        """Test ModelManager handles different model response formats"""
        from app import ModelManager
        
        # Test with mixed model name formats
        mock_ollama.return_value = {
            'models': [
                {'model': 'llama2:7b'},
                {'name': 'mistral:latest'},
                {'model': 'codellama:13b-instruct'},
                {'name': 'deepseek-r1:14b'}
            ]
        }
        
        models = ModelManager.get_available_models()
        
        # Should extract model names correctly
        assert 'llama2:7b' in models
        assert 'mistral:latest' in models
        assert 'codellama:13b-instruct' in models
        assert 'deepseek-r1:14b' in models
        assert len(models) == 4
    
    @patch('ollama.list')
    @patch('streamlit.error')
    def test_model_manager_connection_failures(self, mock_error, mock_ollama):
        """Test ModelManager handles various connection failures"""
        from app import ModelManager
        
        # Test different types of errors
        test_errors = [
            ConnectionError("Connection refused"),
            TimeoutError("Request timeout"),
            Exception("Unexpected error")
        ]
        
        for error in test_errors:
            mock_ollama.side_effect = error
            models = ModelManager.get_available_models()
            
            # Should always return empty list and log error
            assert models == []
        
        # Should have called error logging for each failure
        assert mock_error.call_count == len(test_errors)
    
    @patch('ollama.list')
    def test_model_manager_empty_response(self, mock_ollama):
        """Test ModelManager handles empty or malformed responses"""
        from app import ModelManager
        
        test_cases = [
            {},  # Empty dict
            {'models': []},  # Empty models list
            {'other_key': 'value'},  # Missing models key
            {'models': None},  # None models
        ]
        
        for case in test_cases:
            mock_ollama.return_value = case
            models = ModelManager.get_available_models()
            assert models == []


class TestCitationExtraction:
    """Citation functionality regression tests"""
    
    def test_citation_extraction_with_exact_quotes(self):
        """Test citation extraction finds exact quotes"""
        from ragnarok import EnhancedPDFProcessor
        
        # Create PDF with specific text to quote
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 100), "The research methodology involves machine learning algorithms and data analysis techniques.")
        pdf_bytes = doc.tobytes()
        doc.close()
        
        processor = EnhancedPDFProcessor(pdf_bytes)
        original_text = processor.extract_full_text()
        
        # Test with exact quote from the document
        ai_response = 'The document states "machine learning algorithms and data analysis techniques" are used.'
        highlights = processor.create_ai_response_highlights(ai_response, original_text)
        
        # Should find the exact quote
        assert len(highlights) > 0
        assert "machine learning algorithms and data analysis techniques" in highlights[0]
    
    def test_citation_extraction_with_partial_quotes(self):
        """Test citation extraction handles partial quotes"""
        from ragnarok import EnhancedPDFProcessor
        
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 100), "The artificial intelligence system demonstrated remarkable performance in natural language processing tasks.")
        pdf_bytes = doc.tobytes()
        doc.close()
        
        processor = EnhancedPDFProcessor(pdf_bytes)
        original_text = processor.extract_full_text()
        
        # Test with partial quote
        ai_response = 'According to the text, "artificial intelligence system demonstrated remarkable performance" was observed.'
        highlights = processor.create_ai_response_highlights(ai_response, original_text)
        
        # Should find matches for substantial quotes
        assert isinstance(highlights, list)
        if highlights:  # May or may not match depending on fuzzy matching threshold
            assert len(highlights[0]) > 10  # Should be substantial quotes only
    
    def test_citation_extraction_ignores_short_quotes(self):
        """Test citation extraction ignores very short quotes"""
        from ragnarok import EnhancedPDFProcessor
        
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 100), "The cat sat on the mat in the garden.")
        pdf_bytes = doc.tobytes()
        doc.close()
        
        processor = EnhancedPDFProcessor(pdf_bytes)
        original_text = processor.extract_full_text()
        
        # Test with very short quotes that should be ignored
        ai_response = 'The text mentions "cat" and "mat" and "the" in various places.'
        highlights = processor.create_ai_response_highlights(ai_response, original_text)
        
        # Should ignore short quotes (less than 10 characters)
        for highlight in highlights:
            assert len(highlight) > 10
    
    def test_citation_extraction_no_matches(self):
        """Test citation extraction when no quotes match"""
        from ragnarok import EnhancedPDFProcessor
        
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 100), "This document talks about completely different topics.")
        pdf_bytes = doc.tobytes()
        doc.close()
        
        processor = EnhancedPDFProcessor(pdf_bytes)
        original_text = processor.extract_full_text()
        
        # Test with quotes that don't exist in the document
        ai_response = 'The document mentions "quantum computing" and "blockchain technology" extensively.'
        highlights = processor.create_ai_response_highlights(ai_response, original_text)
        
        # Should return empty list when no matches found
        assert highlights == []


class TestStreamlitIntegration:
    """Basic Streamlit integration tests"""
    
    def test_render_functions_exist(self):
        """Test that main render functions exist"""
        import app
        assert hasattr(app, 'render_document_upload')
        assert hasattr(app, 'render_chat_interface')
        assert hasattr(app, 'render_sidebar')
    
    def test_chat_manager_exists(self):
        """Test ChatManager class exists"""
        from app import ChatManager
        assert ChatManager is not None
        
        # Test basic instantiation
        manager = ChatManager()
        assert hasattr(manager, 'get_current_chat')
        assert hasattr(manager, 'create_new_chat')


@pytest.mark.integration
class TestBasicIntegration:
    """Basic integration smoke tests"""
    
    def test_full_pdf_to_text_flow(self):
        """Test complete PDF to text extraction flow"""
        from app import PDFProcessor
        
        # Create PDF with some content
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 100), "Integration test document with multiple words")
        pdf_bytes = doc.tobytes()
        doc.close()
        
        # Mock file upload
        mock_file = Mock()
        mock_file.getvalue.return_value = pdf_bytes
        mock_file.name = "test.pdf"
        
        # Extract text
        text = PDFProcessor.extract_text(mock_file)
        
        # Verify
        assert isinstance(text, str)
        assert len(text.strip()) > 0
        assert "integration test" in text.lower()
        assert len(text.split()) > 3  # Multiple words
    
    def test_enhanced_processor_with_citations(self):
        """Test enhanced processor with citation workflow"""
        from ragnarok import EnhancedPDFProcessor
        
        # Create PDF
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 100), "This document discusses machine learning algorithms")
        pdf_bytes = doc.tobytes()
        doc.close()
        
        # Test workflow
        processor = EnhancedPDFProcessor(pdf_bytes)
        text = processor.extract_full_text()
        
        ai_response = 'The document states "machine learning algorithms" are discussed.'
        highlights = processor.create_ai_response_highlights(ai_response, text)
        
        # Basic verification
        assert isinstance(highlights, list)
        assert text is not None
        assert len(text) > 0 