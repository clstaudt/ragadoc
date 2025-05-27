"""
Unit tests for PDF extraction functionality
"""
import pytest
import sys
import os
import fitz

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragnarok import EnhancedPDFProcessor


class TestBasicExtraction:
    """Test basic PDF extraction functionality"""
    
    def test_extract_full_text_basic(self):
        """Test the main text extraction method"""
        # Create a test PDF
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 100), "This is a test document with some content.")
        pdf_bytes = doc.tobytes()
        doc.close()
        
        processor = EnhancedPDFProcessor(pdf_bytes)
        text = processor.extract_full_text()
        
        assert isinstance(text, str)
        assert len(text) > 0
        assert "test document" in text
    
    def test_extract_full_text_multipage(self):
        """Test extraction from multi-page documents"""
        # Create a multi-page PDF
        doc = fitz.open()
        
        for i in range(3):
            page = doc.new_page()
            page.insert_text((50, 100), f"Page {i+1} content with some text.")
        
        pdf_bytes = doc.tobytes()
        doc.close()
        
        processor = EnhancedPDFProcessor(pdf_bytes)
        text = processor.extract_full_text()
        
        assert isinstance(text, str)
        assert len(text) > 0
        assert "Page 1" in text
        assert "Page 3" in text
    
    def test_extract_full_text_with_structure(self):
        """Test extraction preserves basic structure"""
        # Create a PDF with structured content
        doc = fitz.open()
        page = doc.new_page()
        
        structured_content = """Title: Test Document

Section 1: Introduction
This is the introduction.

Section 2: Methods
- Method 1
- Method 2

Numbers: 123, 456"""
        
        page.insert_text((50, 100), structured_content)
        pdf_bytes = doc.tobytes()
        doc.close()
        
        processor = EnhancedPDFProcessor(pdf_bytes)
        text = processor.extract_full_text()
        
        # Check that content is preserved
        assert "Test Document" in text
        assert "Section 1" in text
        assert "Section 2" in text
        assert "123" in text


class TestDocumentMetadata:
    """Test document metadata extraction"""
    
    def test_get_document_metadata_basic(self):
        """Test basic metadata extraction"""
        # Create a test PDF with metadata
        doc = fitz.open()
        doc.set_metadata({
            "title": "Test Document Title",
            "author": "Test Author",
            "subject": "Test Subject"
        })
        
        page = doc.new_page()
        page.insert_text((50, 100), "Document content")
        pdf_bytes = doc.tobytes()
        doc.close()
        
        processor = EnhancedPDFProcessor(pdf_bytes)
        metadata = processor.get_document_metadata()
        
        assert isinstance(metadata, dict)
        assert "page_count" in metadata
        assert metadata["page_count"] == 1
    
    def test_multipage_document_metadata(self):
        """Test metadata for multi-page documents"""
        # Create a multi-page PDF
        doc = fitz.open()
        
        for i in range(3):
            page = doc.new_page()
            page.insert_text((50, 100), f"Page {i+1} content")
        
        pdf_bytes = doc.tobytes()
        doc.close()
        
        processor = EnhancedPDFProcessor(pdf_bytes)
        metadata = processor.get_document_metadata()
        
        assert metadata["page_count"] == 3


class TestSectionExtraction:
    """Test document section extraction"""
    
    def test_extract_sections_basic(self):
        """Test basic section extraction"""
        # Create a test PDF
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 100), "Document with some content for section extraction.")
        pdf_bytes = doc.tobytes()
        doc.close()
        
        processor = EnhancedPDFProcessor(pdf_bytes)
        sections = processor.extract_sections()
        
        assert isinstance(sections, dict)
        # Sections might be empty for simple documents, that's okay
    
    def test_extract_table_of_contents(self):
        """Test table of contents extraction"""
        # Create a test PDF
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 100), "Document with potential TOC structure")
        pdf_bytes = doc.tobytes()
        doc.close()
        
        processor = EnhancedPDFProcessor(pdf_bytes)
        toc = processor.extract_table_of_contents()
        
        assert isinstance(toc, list)
        # TOC might be empty for simple documents, that's okay


class TestErrorHandling:
    """Test error handling in extraction"""
    
    def test_corrupted_pdf_handling(self):
        """Test handling of corrupted PDF data"""
        # Test with invalid PDF data
        invalid_pdf_data = b"This is not a PDF file"
        
        with pytest.raises(Exception):
            # Should raise an exception for invalid PDF
            EnhancedPDFProcessor(invalid_pdf_data)
    
    def test_empty_pdf_handling(self):
        """Test handling of empty PDF documents"""
        # Create an empty PDF
        doc = fitz.open()
        page = doc.new_page()
        # Don't add any content
        pdf_bytes = doc.tobytes()
        doc.close()
        
        processor = EnhancedPDFProcessor(pdf_bytes)
        text = processor.extract_full_text()
        
        # Should handle empty documents gracefully
        assert isinstance(text, str)
        # Text might be empty or contain minimal content


class TestSpecialCharacters:
    """Test handling of special characters and encoding"""
    
    def test_character_encoding_handling(self):
        """Test handling of various character encodings"""
        # Create a PDF with special characters
        doc = fitz.open()
        page = doc.new_page()
        
        # Include various Unicode characters
        special_content = "Special chars: áéíóú ñ ü ß"
        page.insert_text((50, 100), special_content)
        pdf_bytes = doc.tobytes()
        doc.close()
        
        processor = EnhancedPDFProcessor(pdf_bytes)
        text = processor.extract_full_text()
        
        assert isinstance(text, str)
        assert len(text) > 0
        # Should handle special characters gracefully
        assert "Special chars" in text 