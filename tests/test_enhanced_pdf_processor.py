"""
Test EnhancedPDFProcessor functionality
"""
import pytest
import io
import fitz
from ragnarok import EnhancedPDFProcessor


class TestEnhancedPDFProcessor:
    """Test the EnhancedPDFProcessor class"""

    def test_processor_initialization(self, sample_pdf_bytes):
        """Test that processor can be initialized with PDF bytes"""
        processor = EnhancedPDFProcessor(sample_pdf_bytes)
        assert processor is not None
        assert processor.doc is not None
        assert processor.doc.page_count >= 1

    def test_extract_full_text(self, pdf_processor):
        """Test full text extraction"""
        full_text = pdf_processor.extract_full_text()
        assert isinstance(full_text, str)
        assert len(full_text) > 0
        assert "sample document" in full_text.lower()
        assert "highlighting functionality" in full_text.lower()

    def test_extract_text_with_positions(self, pdf_processor):
        """Test text extraction with position information"""
        text_data = pdf_processor.extract_text_with_positions()
        assert isinstance(text_data, dict)
        assert len(text_data) >= 1  # At least one page
        
        # Check first page data
        page_0_data = text_data.get(0, [])
        assert len(page_0_data) > 0
        
        # Check structure of text data
        first_text_block = page_0_data[0]
        assert "text" in first_text_block
        assert "bbox" in first_text_block
        assert "page" in first_text_block
        assert isinstance(first_text_block["bbox"], (list, tuple))

    def test_search_and_highlight_text(self, pdf_processor):
        """Test searching and highlighting text"""
        search_terms = ["sample", "highlighting"]
        highlighted_pdf_bytes = pdf_processor.search_and_highlight_text(search_terms)
        
        assert isinstance(highlighted_pdf_bytes, bytes)
        assert len(highlighted_pdf_bytes) > 0
        
        # Should be able to create a new document from highlighted bytes
        import fitz
        highlighted_doc = fitz.open(stream=highlighted_pdf_bytes, filetype="pdf")
        assert highlighted_doc.page_count >= 1
        highlighted_doc.close()

    def test_create_ai_response_highlights_basic(self, pdf_processor, sample_ai_response):
        """Test extracting highlights from AI response"""
        original_text = pdf_processor.extract_full_text()
        highlight_terms = pdf_processor.create_ai_response_highlights(sample_ai_response, original_text)
        
        assert isinstance(highlight_terms, list)
        # Should find "quoted text" from the sample response
        assert len(highlight_terms) >= 1
        assert any("quoted text" in term.lower() for term in highlight_terms)

    def test_create_ai_response_highlights_german(self, german_pdf_processor, german_ai_response):
        """Test extracting highlights from German AI response"""
        original_text = german_pdf_processor.extract_full_text()
        highlight_terms = german_pdf_processor.create_ai_response_highlights(german_ai_response, original_text)
        
        assert isinstance(highlight_terms, list)
        # Should find the German quote
        assert len(highlight_terms) >= 1
        # Check for partial matches of the German text
        found_match = any("kindheitspädagogische" in term for term in highlight_terms)
        assert found_match

    def test_fuzzy_text_match(self, pdf_processor):
        """Test the fuzzy text matching functionality"""
        document_text = "This is a sample document with various text content."
        
        # Exact match
        assert pdf_processor._fuzzy_text_match("sample document", document_text)
        
        # Case insensitive match
        assert pdf_processor._fuzzy_text_match("SAMPLE DOCUMENT", document_text)
        
        # Partial match with normalization
        assert pdf_processor._fuzzy_text_match("sample document with", document_text)
        
        # Non-matching text
        assert not pdf_processor._fuzzy_text_match("completely different text", document_text)

    def test_get_highlighted_snippets(self, pdf_processor):
        """Test getting highlighted snippets"""
        search_terms = ["sample", "document"]
        snippets = pdf_processor.get_highlighted_snippets(search_terms, context_chars=100)
        
        assert isinstance(snippets, list)
        if snippets:  # May be empty if terms aren't found in this specific test
            snippet = snippets[0]
            assert "term" in snippet
            assert "page" in snippet
            assert "context" in snippet
            assert "bbox" in snippet
            assert "page_image" in snippet

    def test_render_page_with_highlights(self, pdf_processor):
        """Test rendering a page with highlights"""
        # First create some highlights
        search_terms = ["sample"]
        pdf_processor.search_and_highlight_text(search_terms)
        
        # Render the page
        page_image_bytes = pdf_processor.render_page_with_highlights(0, dpi=100)
        
        assert isinstance(page_image_bytes, bytes)
        assert len(page_image_bytes) > 0
        
        # Should be valid PNG
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(page_image_bytes))
        assert img.format == "PNG"

    def test_get_context_around_highlights(self, pdf_processor):
        """Test getting context around highlighted terms"""
        search_terms = ["sample", "document"]
        contexts = pdf_processor.get_context_around_highlights(search_terms, context_chars=50)
        
        assert isinstance(contexts, list)
        if contexts:  # May be empty if terms aren't found
            context = contexts[0]
            assert "term" in context
            assert "context" in context
            assert "position" in context

    def test_processor_cleanup(self, sample_pdf_bytes):
        """Test that processor properly cleans up resources"""
        processor = EnhancedPDFProcessor(sample_pdf_bytes)
        assert processor.doc is not None
        
        # Manual cleanup
        del processor
        
        # Should be able to create a new processor
        processor2 = EnhancedPDFProcessor(sample_pdf_bytes)
        assert processor2.doc is not None

    def test_normalize_for_search(self, pdf_processor):
        """Test text normalization for search"""
        # Test various text normalization scenarios
        test_cases = [
            ("Hello World!", "hello world"),
            ("  Multiple   Spaces  ", "multiple spaces"),
            ("Ümlauts änd spëcial chars!", "umlauts and special chars"),
            ("Mixed    Case\n\nWith\tTabs", "mixed case with tabs"),
        ]
        
        for input_text, expected_normalized in test_cases:
            normalized = pdf_processor._normalize_for_search(input_text)
            assert expected_normalized in normalized.lower()

    def test_empty_search_terms(self, pdf_processor):
        """Test handling of empty search terms"""
        # Empty list
        highlighted_pdf = pdf_processor.search_and_highlight_text([])
        assert isinstance(highlighted_pdf, bytes)
        
        # List with empty strings
        highlighted_pdf = pdf_processor.search_and_highlight_text(["", "  ", None])
        assert isinstance(highlighted_pdf, bytes)

    def test_multiple_page_highlighting(self):
        """Test highlighting across multiple pages"""
        import fitz
        
        # Create multi-page PDF
        doc = fitz.open()
        
        # Page 1
        page1 = doc.new_page()
        page1.insert_text((50, 100), "This is page one with important content.")
        
        # Page 2
        page2 = doc.new_page()
        page2.insert_text((50, 100), "This is page two with important information.")
        
        pdf_bytes = doc.tobytes()
        doc.close()
        
        # Test highlighting
        processor = EnhancedPDFProcessor(pdf_bytes)
        highlighted_pdf = processor.search_and_highlight_text(["important"])
        
        assert isinstance(highlighted_pdf, bytes)
        assert len(highlighted_pdf) > 0
        
        # Verify both pages exist in highlighted PDF
        highlighted_doc = fitz.open(stream=highlighted_pdf, filetype="pdf")
        assert highlighted_doc.page_count == 2
        highlighted_doc.close()

    def test_special_characters_in_search(self, pdf_processor):
        """Test searching for text with special characters"""
        # Create PDF with special characters
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        
        special_text = "Text with special chars: @#$%^&*()_+-=[]{}|;':\",./<>?"
        page.insert_text((50, 100), special_text)
        
        pdf_bytes = doc.tobytes()
        doc.close()
        
        processor = EnhancedPDFProcessor(pdf_bytes)
        
        # Test highlighting special characters (should not crash)
        search_terms = ["@#$%", "special chars"]
        highlighted_pdf = processor.search_and_highlight_text(search_terms)
        assert isinstance(highlighted_pdf, bytes) 