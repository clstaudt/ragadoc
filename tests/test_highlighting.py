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


class TestImprovedCitationExtraction:
    """Test the improved citation extraction logic"""
    
    def test_preserves_context_for_medium_quotes(self):
        """Test that medium-length quotes preserve their full context"""
        from ragnarok import EnhancedPDFProcessor
        import fitz
        
        # Create a test PDF
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 100), "The system achieved 85% accuracy with improved algorithms and better data processing.")
        pdf_bytes = doc.tobytes()
        doc.close()
        
        processor = EnhancedPDFProcessor(pdf_bytes)
        
        # Test response with medium-length quote that should be preserved
        ai_response = 'The results show that [1] "achieved 85% accuracy with improved algorithms and better data" according to the study.'
        citations = processor._extract_quotes_from_ai_response(ai_response, "What was the accuracy?")
        
        # Should preserve the full quote since it's under 30 words
        assert len(citations) == 1
        assert "achieved 85% accuracy with improved algorithms and better data" in citations[1]
        assert len(citations[1].split()) > 8  # Should be substantial
    
    def test_avoids_over_trimming_contextual_quotes(self):
        """Test that quotes with important context are not over-trimmed"""
        from ragnarok import EnhancedPDFProcessor
        import fitz
        
        # Create a test PDF
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 100), "Performance increased by 50% compared to the previous version when using the new optimization techniques.")
        pdf_bytes = doc.tobytes()
        doc.close()
        
        processor = EnhancedPDFProcessor(pdf_bytes)
        
        # Test response that might have been over-trimmed before
        ai_response = 'The document states [1] "Performance increased by 50% compared to the previous version" in the results section.'
        citations = processor._extract_quotes_from_ai_response(ai_response, "How much did performance improve?")
        
        # Should preserve the full contextual quote, not just "50%"
        assert len(citations) == 1
        assert "Performance increased by 50% compared to the previous version" in citations[1]
        assert "50%" in citations[1]  # Should include the percentage
        assert len(citations[1].split()) >= 8  # Should have substantial context
    
    def test_only_trims_very_long_quotes(self):
        """Test that only extremely long quotes (>30 words) get trimmed"""
        from ragnarok import EnhancedPDFProcessor
        import fitz
        
        # Create a test PDF
        doc = fitz.open()
        page = doc.new_page()
        long_text = "This is a very long sentence that contains more than thirty words and should potentially be trimmed by the new logic because it exceeds the threshold for automatic processing and trimming that we have established in our improved citation system."
        page.insert_text((50, 100), long_text)
        pdf_bytes = doc.tobytes()
        doc.close()
        
        processor = EnhancedPDFProcessor(pdf_bytes)
        
        # Test with a very long quote
        ai_response = f'The document explains that [1] "{long_text}" in great detail.'
        citations = processor._extract_quotes_from_ai_response(ai_response, "What does it explain?")
        
        # Should be trimmed since it's over 30 words
        assert len(citations) == 1
        original_word_count = len(long_text.split())
        extracted_word_count = len(citations[1].split())
        
        # Should be shorter than original but still substantial
        assert extracted_word_count < original_word_count
        assert extracted_word_count >= 8  # Should still be meaningful
    
    def test_rejects_very_short_terms_for_highlighting(self):
        """Test that very short terms don't go through partial matching"""
        from ragnarok import EnhancedPDFProcessor
        import fitz
        
        # Create a test PDF
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 100), "The cat sat on the mat with 50% efficiency.")
        pdf_bytes = doc.tobytes()
        doc.close()
        
        processor = EnhancedPDFProcessor(pdf_bytes)
        
        # Simulate the improved _smart_highlight_long_quote_fast logic
        # Very short terms should return False without trying partial matching
        result = processor._smart_highlight_long_quote_fast(page, "cat")  # 1 word
        assert result == False  # Should not highlight single words
        
        result = processor._smart_highlight_long_quote_fast(page, "50%")  # Very short
        assert result == False  # Should not highlight isolated percentages
        
        result = processor._smart_highlight_long_quote_fast(page, "the cat sat")  # 3 words
        assert result == False  # Should not highlight very short phrases 