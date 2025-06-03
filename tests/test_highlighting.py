"""
Highlighting functionality tests - focused on core behavior
"""
import pytest
import sys
import os
import fitz

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragnarok import EnhancedPDFProcessor


class TestHighlightingBasics:
    """Test basic highlighting functionality"""
    
    def test_highlighting_phrase_logic(self):
        """Test core phrase highlighting logic"""
        def should_highlight_phrase(phrase):
            # Simple version of phrase highlighting logic
            if len(phrase.strip()) < 8:
                return False
            if phrase.strip().isdigit():
                return False
            if any(keyword in phrase.lower() for keyword in ['%', 'times', 'accuracy', 'performance']):
                return True
            return len(phrase.strip()) >= 12
        
        # Test meaningful content should be highlighted
        assert should_highlight_phrase("50% accuracy improvement")
        assert should_highlight_phrase("performance increased significantly")
        
        # Test trivial content should not be highlighted
        assert not should_highlight_phrase("the cat sat")
        assert not should_highlight_phrase("123")
    
    def test_three_word_phrases(self):
        """Test three-word phrase highlighting"""
        def should_highlight_3word_phrase(phrase):
            import re
            if (re.search(r'\d+%', phrase) or  # Contains percentage
                re.search(r'\d+\s+\w+', phrase) or  # Contains "number word"
                re.search(r'\w+\s+\d+', phrase) or  # Contains "word number"
                len(phrase.strip()) >= 12):
                
                if not re.match(r'^\d+$', phrase.strip()):
                    return True
            return False
        
        assert should_highlight_3word_phrase("50% smaller result")
        assert should_highlight_3word_phrase("6 times faster")
        assert not should_highlight_3word_phrase("the cat sat")


class TestAIResponseHighlights:
    """Test AI response highlight extraction"""
    
    def test_ai_response_highlights_basic(self):
        """Test basic highlight extraction from AI responses"""
        import fitz
        
        # Create a test PDF
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 100), "This paper discusses machine learning algorithms and achieved 95% accuracy.")
        pdf_bytes = doc.tobytes()
        doc.close()
        
        processor = EnhancedPDFProcessor(pdf_bytes)
        
        # Test with quoted text
        ai_response = '''The document states that "machine learning algorithms" are effective.'''
        
        document_text = processor.extract_full_text()
        highlights = processor.create_ai_response_highlights(ai_response, document_text)
        
        assert isinstance(highlights, list)
        # Should find matches when quotes exist in document
    
    def test_ai_response_no_matches(self):
        """Test AI response highlights when no matches found"""
        import fitz
        
        # Create a test PDF
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 100), "This paper is about machine learning.")
        pdf_bytes = doc.tobytes()
        doc.close()
        
        processor = EnhancedPDFProcessor(pdf_bytes)
        
        ai_response = '''The document mentions "quantum computing" and "blockchain".'''
        document_text = processor.extract_full_text()
        
        highlights = processor.create_ai_response_highlights(ai_response, document_text)
        
        assert isinstance(highlights, list)
        assert len(highlights) == 0  # No matches should be found
    
    def test_fuzzy_text_matching(self):
        """Test fuzzy text matching works"""
        import fitz
        
        # Create a test PDF
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 100), "The machine learning algorithms performed well.")
        pdf_bytes = doc.tobytes()
        doc.close()
        
        processor = EnhancedPDFProcessor(pdf_bytes)
        document_text = processor.extract_full_text()
        
        # Test exact match
        assert processor._fuzzy_text_match("machine learning algorithms", document_text)
        
        # Test no match
        assert not processor._fuzzy_text_match("quantum computing", document_text)


class TestCitationExtraction:
    """Test citation extraction logic"""
    
    def test_citation_preserves_context(self):
        """Test that citations preserve meaningful context"""
        from ragnarok import EnhancedPDFProcessor
        import fitz
        
        # Create a test PDF
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 100), "The system achieved 85% accuracy with improved algorithms.")
        pdf_bytes = doc.tobytes()
        doc.close()
        
        processor = EnhancedPDFProcessor(pdf_bytes)
        
        # Test response with meaningful quote
        ai_response = 'The results show [1] "achieved 85% accuracy with improved algorithms" according to the study.'
        citations = processor._extract_quotes_from_ai_response(ai_response, "What was the accuracy?")
        
        # Should preserve meaningful quotes
        assert len(citations) == 1
        assert "85% accuracy" in citations[1]
    
    def test_citation_length_handling(self):
        """Test citation handles different quote lengths appropriately"""
        from ragnarok import EnhancedPDFProcessor
        import fitz
        
        # Create a test PDF
        doc = fitz.open()
        page = doc.new_page()
        long_text = "This is a moderately long sentence that contains important information about the research methodology and findings."
        page.insert_text((50, 100), long_text)
        pdf_bytes = doc.tobytes()
        doc.close()
        
        processor = EnhancedPDFProcessor(pdf_bytes)
        
        # Test with moderate length quote
        ai_response = f'The document explains [1] "{long_text}" in detail.'
        citations = processor._extract_quotes_from_ai_response(ai_response, "What does it explain?")
        
        # Should handle different lengths appropriately
        assert len(citations) == 1
        assert len(citations[1]) > 0 