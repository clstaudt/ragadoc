#!/usr/bin/env python3
"""
Test Unicode quote detection
"""

import re

def test_quote_patterns():
    """Test quote detection with various Unicode quotes"""
    
    # Test cases with different quote types
    test_cases = [
        'Normal ASCII quotes: "test quote"',
        'Smart quotes: \u201ctest quote\u201d',
        'German quotes: \u201etest quote\u201c',
        'French guillemets: \u00bbtest quote\u00ab',
        'Your actual AI response: \u201cAktuell arbeite ich als kindheitspädagogische FachkRAFT in einer KITA und begleite dort Kinder mit viel ENGAGEMENT in ihrer Entwicklung.\u201d'
    ]
    
    # Comprehensive regex patterns for various quote types including Unicode
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
    
    print("🔧 Testing Unicode Quote Detection")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases):
        print(f"\n📝 Test Case {i+1}:")
        print(f"Text: {test_case}")
        
        found_quotes = []
        
        for j, pattern in enumerate(quoted_patterns):
            try:
                matches = re.findall(pattern, test_case)
                if matches:
                    print(f"  Pattern {j+1}: Found: {matches}")
                    found_quotes.extend(matches)
            except Exception as e:
                print(f"  Pattern {j+1}: Error - {e}")
        
        if found_quotes:
            print(f"  ✅ Total quotes found: {len(found_quotes)}")
            for quote in found_quotes:
                print(f"    - '{quote}'")
        else:
            print(f"  ❌ No quotes found")

if __name__ == "__main__":
    test_quote_patterns() 