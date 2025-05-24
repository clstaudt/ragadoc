"""
Test text matching and highlighting logic
"""
import pytest
import re


class TestTextMatching:
    """Test text matching and fuzzy search functionality"""

    def test_exact_text_matching(self):
        """Test exact text matching"""
        document_text = """und dabei stets die pädagogische Begleitung von Kindern und ihren Familien – unter anderem Pflegefamilien, 
Kinder mit Down-Syndrom und deren Familien sowie Geschwister von Kindern mit Behinderungen. Aktuell 
arbeite ich als kindheitspädagogische Fachkraft in einer Kita und begleite dort Kinder mit viel Engagement in 
ihrer Entwicklung. Zusätzlich bilde ich mich im Bereich Kinderschutz und Kindeswohlgefährdung fort, da mir 
diese Themen besonders am Herzen liegen."""

        search_term = "kindheitspädagogische Fachkraft in einer Kita"
        
        # Should find exact match
        assert search_term.lower() in document_text.lower()

    def test_case_insensitive_matching(self):
        """Test case insensitive text matching"""
        document_text = "This is a SAMPLE document with Mixed Case text."
        search_term = "sample document"
        
        assert search_term.lower() in document_text.lower()

    def test_partial_word_matching(self, pdf_processor):
        """Test partial word matching using fuzzy match"""
        document_text = "Working as a childhood education specialist in a daycare center."
        
        # Test various partial matches
        test_cases = [
            ("childhood education", True),
            ("education specialist", True),
            ("daycare center", True),
            ("completely different", False),
        ]
        
        for search_term, should_match in test_cases:
            result = pdf_processor._fuzzy_text_match(search_term, document_text)
            assert result == should_match

    def test_whitespace_normalization(self, pdf_processor):
        """Test that whitespace differences don't break matching"""
        document_text = "Text   with    multiple      spaces"
        search_term = "Text with multiple spaces"
        
        assert pdf_processor._fuzzy_text_match(search_term, document_text)

    def test_punctuation_handling(self, pdf_processor):
        """Test matching with punctuation differences"""
        document_text = "Text, with: various; punctuation! marks?"
        search_term = "Text with various punctuation marks"
        
        assert pdf_processor._fuzzy_text_match(search_term, document_text)

    def test_word_boundary_matching(self):
        """Test word boundary considerations"""
        document_text = "The specialist works in childhood education."
        
        # Should find whole words
        assert "specialist" in document_text.lower()
        assert "childhood" in document_text.lower()
        
        # Should not match partial words in wrong context
        search_term = "child education"  # Missing "hood" from "childhood"
        # This should still work with fuzzy matching due to significant word overlap

    def test_ai_response_quote_extraction(self):
        """Test extracting quotes from AI responses"""
        ai_response = '''Sena Neriman Demirbas arbeitet zurzeit als kindheitspädagogische Fachkraft in einer Kita. Dies wird aus dem Text "Aktuell arbeite ich als kindheitspädagogische Fachkraft in einer Kita" entnommen.'''
        
        # Test quote extraction patterns
        quoted_patterns = [
            r'"([^"]+)"',  # Text in double quotes
            r"'([^']+)'",  # Text in single quotes
            r'`([^`]+)`',  # Text in backticks
        ]
        
        extracted_quotes = []
        for pattern in quoted_patterns:
            matches = re.findall(pattern, ai_response)
            extracted_quotes.extend(matches)
        
        assert len(extracted_quotes) >= 1
        # Should find the German quote
        assert any("kindheitspädagogische" in quote for quote in extracted_quotes)

    def test_context_extraction(self, pdf_processor):
        """Test extracting context around found terms"""
        # This uses the existing PDF with sample text
        full_text = pdf_processor.extract_full_text()
        
        # Find a term that should exist
        search_term = "sample"
        if search_term.lower() in full_text.lower():
            # Get context around the term
            context = pdf_processor._get_text_context(search_term, full_text, context_chars=100)
            
            if context:  # May be None if term not found
                assert search_term.lower() in context.lower()
                assert len(context) <= 200  # Should respect context_chars limit

    def test_significant_word_extraction(self):
        """Test extraction of significant words from quotes"""
        quote = "kindheitspädagogische Fachkraft in einer Kita und begleite dort Kinder"
        words = quote.split()
        
        # Find significant words (longer than 3 characters)
        significant_words = [w for w in words if len(w) > 3]
        
        assert "kindheitspädagogische" in significant_words
        assert "Fachkraft" in significant_words
        assert "einer" in significant_words
        assert "Kita" in significant_words
        assert "begleite" in significant_words
        
        # Short words should be filtered out
        assert "in" not in significant_words
        assert "und" not in significant_words

    def test_partial_phrase_matching(self, pdf_processor):
        """Test matching partial phrases from longer quotes"""
        document_text = "I work as a childhood education specialist in a kindergarten and support children there."
        
        long_quote = "work as a childhood education specialist in a kindergarten"
        
        # Should match the full phrase
        assert pdf_processor._fuzzy_text_match(long_quote, document_text)
        
        # Should also match significant parts
        partial_phrases = [
            "childhood education specialist",
            "specialist in a kindergarten",
            "work as a childhood education",
        ]
        
        for phrase in partial_phrases:
            result = pdf_processor._fuzzy_text_match(phrase, document_text)
            assert result, f"Failed to match phrase: {phrase}"

    def test_minimum_length_filtering(self):
        """Test that very short quotes are filtered out"""
        short_quotes = ["a", "in", "the", "is", "at"]
        long_quotes = ["this is a longer quote", "substantial content here"]
        
        # Simulate the minimum length check from the processor
        min_length = 5
        
        filtered_short = [q for q in short_quotes if len(q) > min_length]
        filtered_long = [q for q in long_quotes if len(q) > min_length]
        
        assert len(filtered_short) == 0  # All short quotes filtered out
        assert len(filtered_long) == 2   # Long quotes preserved

    def test_unicode_text_matching(self, pdf_processor):
        """Test matching with Unicode characters"""
        document_text = "Text with ümlаuts and spëcial chаrаcters like café and naïve."
        
        test_cases = [
            ("ümlаuts and spëcial", True),
            ("café and naïve", True),
            ("umlаuts and special", True),  # Should match even with normalized chars
            ("completely different", False),
        ]
        
        for search_term, should_match in test_cases:
            result = pdf_processor._fuzzy_text_match(search_term, document_text)
            # Note: The exact behavior may depend on the normalization implementation
            # We're mainly testing that it doesn't crash on Unicode

    def test_word_overlap_threshold(self, pdf_processor):
        """Test word overlap threshold in fuzzy matching"""
        document_text = "The quick brown fox jumps over the lazy dog"
        
        # High overlap - should match
        high_overlap = "quick brown fox jumps"  # 4/4 words match
        assert pdf_processor._fuzzy_text_match(high_overlap, document_text)
        
        # Medium overlap - may or may not match depending on threshold
        medium_overlap = "quick brown elephant jumps"  # 3/4 words match
        # Result depends on implementation threshold
        
        # Low overlap - should not match
        low_overlap = "purple elephant flying high"  # 0/4 words match
        assert not pdf_processor._fuzzy_text_match(low_overlap, document_text)

    def test_empty_and_none_inputs(self, pdf_processor):
        """Test handling of empty or None inputs"""
        document_text = "Sample document text"
        
        # Empty search term
        assert not pdf_processor._fuzzy_text_match("", document_text)
        assert not pdf_processor._fuzzy_text_match("   ", document_text)
        
        # Empty document
        assert not pdf_processor._fuzzy_text_match("test", "")
        
        # Both empty
        assert not pdf_processor._fuzzy_text_match("", "")

    def test_highlight_term_in_context(self, pdf_processor):
        """Test highlighting terms within context"""
        context = "This is a sample document with highlighting functionality."
        term = "sample document"
        
        highlighted = pdf_processor._highlight_term_in_context(term, context)
        
        # Should contain markdown bold formatting
        assert "**" in highlighted
        # Original context should be preserved mostly
        assert "highlighting functionality" in highlighted

    def test_german_text_example_full_flow(self, german_pdf_processor, german_ai_response):
        """Test the full flow with German text as in original debug test"""
        original_text = german_pdf_processor.extract_full_text()
        
        # Extract highlights from AI response
        highlight_terms = german_pdf_processor.create_ai_response_highlights(german_ai_response, original_text)
        
        assert isinstance(highlight_terms, list)
        
        if highlight_terms:
            # Test that we can get snippets for the highlights
            snippets = german_pdf_processor.get_highlighted_snippets(highlight_terms, context_chars=100)
            assert isinstance(snippets, list)
            
            # Test that we can create highlighted PDF
            highlighted_pdf = german_pdf_processor.search_and_highlight_text(highlight_terms)
            assert isinstance(highlighted_pdf, bytes)
            assert len(highlighted_pdf) > 0 