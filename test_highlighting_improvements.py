#!/usr/bin/env python3
"""
Test script to demonstrate the improved highlighting logic.
This tests the generic solution without domain-specific patterns.
"""

import re

def test_partial_match_filtering():
    """Test that our improved partial matching avoids standalone numbers"""
    
    # Simulate the improved logic from _highlight_partial_matches
    def should_highlight_phrase(phrase):
        """Test if a phrase should be highlighted based on our generic rules"""
        
        # Skip phrases that are just numbers or very generic
        if (re.match(r'^\d+$', phrase.strip()) or  # Just a number
            phrase.strip().lower() in ['the', 'and', 'of', 'to', 'in', 'for', 'is', 'on', 'that', 'by'] or
            len(phrase.strip()) < 8):  # Too short
            return False
            
        return True
    
    def should_highlight_3word_phrase(phrase):
        """Test if a 3-word phrase should be highlighted"""
        
        # Only highlight 3-word phrases if they contain meaningful information
        if (re.search(r'\d+%', phrase) or  # Contains percentage
            re.search(r'\d+\s+\w+', phrase) or  # Contains "number word"
            re.search(r'\w+\s+\d+', phrase) or  # Contains "word number"
            re.search(r'\d{1,2}:\d{2}', phrase) or  # Contains time
            len(phrase.strip()) >= 12):  # Or is substantial enough
            
            # Additional check: avoid standalone numbers
            if not re.match(r'^\d+$', phrase.strip()):
                return True
        return False
    
    # Test cases that should NOT be highlighted (the problem cases)
    bad_phrases = [
        "50",  # Standalone number
        "6",   # Standalone number  
        "1",   # Standalone number
        "the", # Generic word
        "and", # Generic word
        "of",  # Generic word
    ]
    
    # Test cases that SHOULD be highlighted (meaningful phrases)
    good_phrases = [
        "50% smaller model",     # Percentage with context
        "6 times faster",        # Number with context
        "within 1% error",       # Number with context
        "trained on 14kh",       # Technical specification
        "ca. 2% of original",    # Percentage with context
        "performance improved",   # Meaningful phrase
        "significantly better",   # Meaningful phrase
        "14:30 departure",       # Time with context
    ]
    
    print("Testing improved highlighting logic...")
    print("\n=== Phrases that should NOT be highlighted ===")
    for phrase in bad_phrases:
        result = should_highlight_phrase(phrase)
        status = "❌ CORRECTLY REJECTED" if not result else "⚠️  INCORRECTLY ACCEPTED"
        print(f"{phrase:20} -> {status}")
    
    print("\n=== Phrases that SHOULD be highlighted ===")
    for phrase in good_phrases:
        # Test both 4+ word logic and 3-word logic
        result = should_highlight_phrase(phrase) or should_highlight_3word_phrase(phrase)
        status = "✅ CORRECTLY ACCEPTED" if result else "❌ INCORRECTLY REJECTED"
        print(f"{phrase:20} -> {status}")
    
    print("\n=== Testing 3-word contextual phrases ===")
    three_word_tests = [
        ("50% smaller", True),    # Should highlight - has percentage
        ("6 times faster", True), # Should highlight - has number + word
        ("times faster than", True), # Should highlight - has context
        ("the cat sat", False),   # Should not highlight - no meaningful info
        ("50 is big", True),      # Should highlight - has number + word
        ("big 50 number", True),  # Should highlight - has word + number
    ]
    
    for phrase, expected in three_word_tests:
        result = should_highlight_3word_phrase(phrase)
        status = "✅ CORRECT" if result == expected else "❌ WRONG"
        print(f"{phrase:20} -> {result:5} (expected {expected:5}) {status}")

if __name__ == "__main__":
    test_partial_match_filtering() 