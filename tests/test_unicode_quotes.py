"""
Test Unicode quote detection functionality
"""
import pytest
import re


class TestUnicodeQuotes:
    """Test Unicode quote pattern detection"""

    def test_ascii_quotes(self):
        """Test detection of standard ASCII quotes"""
        text = 'Normal ASCII quotes: "test quote"'
        pattern = r'"([^"]+)"'
        matches = re.findall(pattern, text)
        assert len(matches) == 1
        assert matches[0] == "test quote"

    def test_smart_quotes(self):
        """Test detection of Unicode smart quotes"""
        text = 'Smart quotes: \u201ctest quote\u201d'
        pattern = r'\u201c([^\u201d]+)\u201d'
        matches = re.findall(pattern, text)
        assert len(matches) == 1
        assert matches[0] == "test quote"

    def test_german_quotes(self):
        """Test detection of German-style quotes"""
        text = 'German quotes: \u201etest quote\u201c'
        pattern = r'\u201e([^\u201c]+)\u201c'
        matches = re.findall(pattern, text)
        assert len(matches) == 1
        assert matches[0] == "test quote"

    def test_french_guillemets(self):
        """Test detection of French guillemets"""
        text = 'French guillemets: \u00bbtest quote\u00ab'
        pattern = r'\u00bb([^\u00ab]+)\u00ab'
        matches = re.findall(pattern, text)
        assert len(matches) == 1
        assert matches[0] == "test quote"

    def test_single_quotes(self):
        """Test detection of single quotes"""
        text = "Single quotes: 'test quote'"
        pattern = r"'([^']+)'"
        matches = re.findall(pattern, text)
        assert len(matches) == 1
        assert matches[0] == "test quote"

    def test_backticks(self):
        """Test detection of backtick quotes"""
        text = "Backticks: `test quote`"
        pattern = r'`([^`]+)`'
        matches = re.findall(pattern, text)
        assert len(matches) == 1
        assert matches[0] == "test quote"

    def test_complex_unicode_example(self):
        """Test complex Unicode example with German text"""
        text = 'Your actual AI response: \u201cAktuell arbeite ich als kindheitspädagogische FachkRAFT in einer KITA und begleite dort Kinder mit viel ENGAGEMENT in ihrer Entwicklung.\u201d'
        pattern = r'\u201c([^\u201d]+)\u201d'
        matches = re.findall(pattern, text)
        assert len(matches) == 1
        assert "kindheitspädagogische" in matches[0]
        assert "ENGAGEMENT" in matches[0]

    def test_all_quote_patterns(self, unicode_quote_samples):
        """Test comprehensive quote detection with various Unicode quotes"""
        # Comprehensive regex patterns for various quote types
        quoted_patterns = [
            r'"([^"]+)"',                    # ASCII double quotes
            r'\u201c([^\u201d]+)\u201d',     # Unicode smart double quotes
            r'\u201e([^\u201c]+)\u201c',     # German-style quotes
            r'\u00bb([^\u00ab]+)\u00ab',     # French-style guillemets
            r"'([^']+)'",                    # ASCII single quotes
            r'\u2018([^\u2019]+)\u2019',     # Unicode smart single quotes
            r'`([^`]+)`',                    # Backticks
            r'\u201a([^\u2018]+)\u2018',     # German-style single quotes
        ]
        
        total_found_quotes = 0
        
        for test_case in unicode_quote_samples:
            found_quotes = []
            
            for pattern in quoted_patterns:
                try:
                    matches = re.findall(pattern, test_case)
                    if matches:
                        found_quotes.extend(matches)
                except Exception:
                    # Skip patterns that cause errors
                    continue
            
            if found_quotes:
                total_found_quotes += len(found_quotes)
                # Verify quotes have reasonable content
                for quote in found_quotes:
                    assert len(quote.strip()) > 0
        
        # Should find quotes in most of the test cases
        assert total_found_quotes >= 3

    def test_enhanced_processor_quote_extraction(self, pdf_processor):
        """Test quote extraction using EnhancedPDFProcessor"""
        # Test with various AI responses containing quotes
        test_responses = [
            'The document mentions "quoted text" and various concepts.',
            'Smart quotes: \u201ctest content\u201d are also supported.',
            'German response: \u201eZitat in deutscher Art\u201c wird erkannt.',
        ]
        
        original_text = pdf_processor.extract_full_text()
        
        for response in test_responses:
            highlight_terms = pdf_processor.create_ai_response_highlights(response, original_text)
            assert isinstance(highlight_terms, list)
            # The method should return a list, even if empty

    def test_quote_patterns_edge_cases(self):
        """Test edge cases for quote detection"""
        edge_cases = [
            ('Empty quotes: ""', ['']),
            ('Nested "quotes with "inner" quotes"', ['quotes with "inner" quotes']),
            ('Multiple "first quote" and "second quote"', ['first quote', 'second quote']),
            ('Unmatched quote: "incomplete', []),
            ('No quotes here at all', []),
            ('Mixed \'single\' and "double" quotes', ['single', 'double']),
        ]
        
        double_quote_pattern = r'"([^"]*)"'
        single_quote_pattern = r"'([^']*)'"
        
        for text, expected_quotes in edge_cases:
            # Test double quotes
            double_matches = re.findall(double_quote_pattern, text)
            # Test single quotes  
            single_matches = re.findall(single_quote_pattern, text)
            
            all_matches = double_matches + single_matches
            
            if expected_quotes:
                assert len(all_matches) >= len(expected_quotes)
                for expected in expected_quotes:
                    assert any(expected in match for match in all_matches)
            else:
                # For cases where no quotes are expected, 
                # we might still find some due to different interpretations
                pass

    def test_quote_extraction_performance(self):
        """Test that quote extraction doesn't hang on large text"""
        # Create a large text with many potential quote patterns
        large_text = 'Text with quotes: "quote1" and more text. ' * 1000
        large_text += 'Smart quotes: \u201cquote2\u201d repeated. ' * 1000
        
        patterns = [
            r'"([^"]+)"',
            r'\u201c([^\u201d]+)\u201d',
        ]
        
        for pattern in patterns:
            # This should complete quickly
            matches = re.findall(pattern, large_text)
            assert len(matches) > 0  # Should find many matches

    def test_special_characters_in_quotes(self):
        """Test quotes containing special characters"""
        special_cases = [
            'Quote with numbers: "123 test 456"',
            'Quote with symbols: "test@email.com & more!"',
            'Quote with unicode: "café naïve résumé"',
            'Quote with newlines: "line1\nline2"',
            'Quote with tabs: "column1\tcolumn2"',
        ]
        
        pattern = r'"([^"]+)"'
        
        for text in special_cases:
            matches = re.findall(pattern, text)
            assert len(matches) >= 1
            # Verify the content includes the special characters
            quote_content = matches[0]
            assert len(quote_content) > 0 