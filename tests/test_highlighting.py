"""
Unit tests for highlighting functionality
"""
import pytest
import re
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragnarok import EnhancedPDFProcessor


class TestHighlightingLogic:
    """Test highlighting logic and phrase filtering"""
    
    def test_should_highlight_phrase_rejects_standalone_numbers(self):
        """Test that standalone numbers are not highlighted"""
        # Simulate the improved logic from _highlight_partial_matches
        def should_highlight_phrase(phrase):
            """Test if a phrase should be highlighted based on generic rules"""
            if (re.match(r'^\d+$', phrase.strip()) or  # Just a number
                phrase.strip().lower() in ['the', 'and', 'of', 'to', 'in', 'for', 'is', 'on', 'that', 'by'] or
                len(phrase.strip()) < 8):  # Too short
                return False
            return True
        
        # Test cases that should NOT be highlighted
        bad_phrases = ["50", "6", "1", "the", "and", "of"]
        
        for phrase in bad_phrases:
            assert not should_highlight_phrase(phrase), f"'{phrase}' should not be highlighted"
    
    def test_should_highlight_phrase_accepts_meaningful_content(self):
        """Test that meaningful phrases are highlighted"""
        def should_highlight_phrase(phrase):
            """Test if a phrase should be highlighted based on generic rules"""
            if (re.match(r'^\d+$', phrase.strip()) or  # Just a number
                phrase.strip().lower() in ['the', 'and', 'of', 'to', 'in', 'for', 'is', 'on', 'that', 'by'] or
                len(phrase.strip()) < 8):  # Too short
                return False
            return True
        
        # Test cases that SHOULD be highlighted
        good_phrases = [
            "performance improved",
            "significantly better",
            "machine learning algorithms",
            "neural networks achieved"
        ]
        
        for phrase in good_phrases:
            assert should_highlight_phrase(phrase), f"'{phrase}' should be highlighted"
    
    def test_three_word_contextual_phrases(self):
        """Test highlighting logic for 3-word phrases with context"""
        def should_highlight_3word_phrase(phrase):
            """Test if a 3-word phrase should be highlighted"""
            if (re.search(r'\d+%', phrase) or  # Contains percentage
                re.search(r'\d+\s+\w+', phrase) or  # Contains "number word"
                re.search(r'\w+\s+\d+', phrase) or  # Contains "word number"
                re.search(r'\d{1,2}:\d{2}', phrase) or  # Contains time
                len(phrase.strip()) >= 12):  # Or is substantial enough
                
                # Additional check: avoid standalone numbers
                if not re.match(r'^\d+$', phrase.strip()):
                    return True
            return False
        
        test_cases = [
            ("50% smaller", True),    # Should highlight - has percentage
            ("6 times faster", True), # Should highlight - has number + word
            ("times faster than", True), # Should highlight - has context
            ("the cat sat", False),   # Should not highlight - no meaningful info
            ("50 is big", True),      # Should highlight - has number + word
            ("big 50 number", True),  # Should highlight - has word + number
            ("14:30 departure", True), # Should highlight - has time
        ]
        
        for phrase, expected in test_cases:
            result = should_highlight_3word_phrase(phrase)
            assert result == expected, f"'{phrase}' should {'be' if expected else 'not be'} highlighted"


class TestAIResponseHighlights:
    """Test AI response highlight extraction"""
    
    def test_ai_response_highlights_with_quotes(self):
        """Test extracting highlights from AI responses with quoted text"""
        import fitz
        
        # Create a test PDF
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 100), "This paper discusses machine learning algorithms and how neural networks achieved 95% accuracy in our experiments.")
        pdf_bytes = doc.tobytes()
        doc.close()
        
        processor = EnhancedPDFProcessor(pdf_bytes)
        
        # Test with quoted text
        ai_response = '''The document states that "machine learning algorithms" are effective. It also mentions "neural networks achieved 95% accuracy" in the results.'''
        
        document_text = processor.extract_full_text()
        
        # Test the highlight extraction
        highlights = processor.create_ai_response_highlights(ai_response, document_text)
        
        assert isinstance(highlights, list)
        assert len(highlights) >= 1  # Should find at least one match
    
    def test_ai_response_highlights_no_matches(self):
        """Test AI response highlights when no matches are found"""
        import fitz
        
        # Create a test PDF
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 100), "This paper is about machine learning and artificial intelligence.")
        pdf_bytes = doc.tobytes()
        doc.close()
        
        processor = EnhancedPDFProcessor(pdf_bytes)
        
        ai_response = '''The document mentions "quantum computing" and "blockchain technology".'''
        document_text = processor.extract_full_text()
        
        highlights = processor.create_ai_response_highlights(ai_response, document_text)
        
        # Should return empty list when no matches found
        assert isinstance(highlights, list)
        assert len(highlights) == 0
    
    def test_fuzzy_text_matching(self):
        """Test the fuzzy text matching functionality"""
        import fitz
        
        # Create a test PDF
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 100), "The machine learning algorithms performed well in the experiments.")
        pdf_bytes = doc.tobytes()
        doc.close()
        
        processor = EnhancedPDFProcessor(pdf_bytes)
        document_text = processor.extract_full_text()
        
        # Test exact match
        assert processor._fuzzy_text_match("machine learning algorithms", document_text)
        
        # Test partial match with different punctuation
        assert processor._fuzzy_text_match("machine learning algorithms performed", document_text)
        
        # Test no match
        assert not processor._fuzzy_text_match("quantum computing", document_text) 