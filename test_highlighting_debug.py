#!/usr/bin/env python3
"""
Debug script to test highlighting text matching
"""

def test_text_matching():
    """Test the text matching logic with real examples"""
    
    # Example from the user's case
    ai_response = '''Sena Neriman Demirbas arbeitet zurzeit als kindheitspädagogische Fachkraft in einer Kita. Dies wird aus dem Text "Aktuell arbeite ich als kindheitspädagogische Fachkraft in einer Kita" entnommen.'''
    
    document_text = """und dabei stets die pädagogische Begleitung von Kindern und ihren Familien – unter anderem Pflegefamilien, 
Kinder mit Down-Syndrom und deren Familien sowie Geschwister von Kindern mit Behinderungen. Aktuell 
arbeite ich als kindheitspädagogische Fachkraft in einer Kita und begleite dort Kinder mit viel Engagement in 
ihrer Entwicklung. Zusätzlich bilde ich mich im Bereich Kinderschutz und Kindeswohlgefährdung fort, da mir 
diese Themen besonders am Herzen liegen. Auc"""
    
    print("🔧 Testing Text Matching Logic")
    print("=" * 50)
    
    # Test 1: Quote extraction
    import re
    quoted_patterns = [
        r'"([^"]+)"',  # Text in double quotes
        r"'([^']+)'",  # Text in single quotes
        r'`([^`]+)`',  # Text in backticks
    ]
    
    print(f"\n📝 AI Response:")
    print(f"'{ai_response}'")
    
    print(f"\n📄 Document snippet:")
    print(f"'{document_text}'")
    
    print(f"\n🔍 Extracting quoted text...")
    highlighted_terms = []
    
    for i, pattern in enumerate(quoted_patterns):
        matches = re.findall(pattern, ai_response)
        print(f"Pattern {i+1} ({pattern}): Found {len(matches)} matches")
        for match in matches:
            print(f"  - '{match}'")
            
            # Clean up the match
            cleaned_match = match.strip()
            
            # Check if it exists in document
            if len(cleaned_match) > 10:
                if cleaned_match.lower() in document_text.lower():
                    print(f"    ✅ Found in document (exact match)")
                    highlighted_terms.append(cleaned_match)
                else:
                    print(f"    ❌ Not found in document (exact match)")
                    
                    # Try partial matching
                    words = cleaned_match.split()
                    if len(words) >= 3:
                        partial_match = ' '.join(words[:3] + words[-2:]) if len(words) > 5 else cleaned_match
                        if partial_match.lower() in document_text.lower():
                            print(f"    ✅ Found partial match: '{partial_match}'")
                            highlighted_terms.append(partial_match)
                        else:
                            print(f"    ❌ No partial match found")
            else:
                print(f"    ⚠️ Quote too short (< 10 chars)")
    
    print(f"\n🎯 Final highlighted terms: {highlighted_terms}")
    
    # Test 2: Text highlighting in context
    print(f"\n💡 Testing context highlighting...")
    
    for term in highlighted_terms:
        print(f"\nTerm: '{term}'")
        
        # Find the term in context
        term_pos = document_text.lower().find(term.lower())
        if term_pos != -1:
            print(f"  Found at position {term_pos}")
            
            # Extract context
            context_start = max(0, term_pos - 50)
            context_end = min(len(document_text), term_pos + len(term) + 50)
            context = document_text[context_start:context_end]
            
            print(f"  Context: '{context}'")
            
            # Try highlighting
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            match = pattern.search(context)
            if match:
                actual_text = match.group()
                highlighted_context = context.replace(actual_text, f"**{actual_text}**")
                print(f"  Highlighted: '{highlighted_context}'")
            else:
                print(f"  ❌ Could not highlight in context")
        else:
            print(f"  ❌ Term not found in document")

def test_with_enhanced_processor():
    """Test with the actual EnhancedPDFProcessor"""
    try:
        from enhanced_pdf_processor import EnhancedPDFProcessor
        import fitz
        
        print("\n" + "=" * 50)
        print("🔧 Testing with EnhancedPDFProcessor")
        
        # Create a test PDF with German text
        doc = fitz.open()
        page = doc.new_page()
        
        test_text = """und dabei stets die pädagogische Begleitung von Kindern und ihren Familien – unter anderem Pflegefamilien, 
Kinder mit Down-Syndrom und deren Familien sowie Geschwister von Kindern mit Behinderungen. Aktuell 
arbeite ich als kindheitspädagogische Fachkraft in einer Kita und begleite dort Kinder mit viel Engagement in 
ihrer Entwicklung. Zusätzlich bilde ich mich im Bereich Kinderschutz und Kindeswohlgefährdung fort, da mir 
diese Themen besonders am Herzen liegen."""
        
        page.insert_text((50, 100), test_text)
        pdf_bytes = doc.tobytes()
        doc.close()
        
        # Test the processor
        processor = EnhancedPDFProcessor(pdf_bytes)
        
        # Test text extraction
        full_text = processor.extract_full_text()
        print(f"📄 Extracted text length: {len(full_text)} chars")
        print(f"First 200 chars: '{full_text[:200]}...'")
        
        # Test AI response highlighting
        ai_response = '''Sena Neriman Demirbas arbeitet zurzeit als kindheitspädagogische Fachkraft in einer Kita. Dies wird aus dem Text "Aktuell arbeite ich als kindheitspädagogische Fachkraft in einer Kita" entnommen.'''
        
        highlight_terms = processor.create_ai_response_highlights(ai_response, full_text)
        print(f"\n🎯 Highlight terms found: {highlight_terms}")
        
        # Test snippet generation
        snippets = processor.get_highlighted_snippets(highlight_terms, context_chars=100)
        print(f"\n📝 Generated {len(snippets)} snippets")
        
        for i, snippet in enumerate(snippets):
            print(f"\nSnippet {i+1}:")
            print(f"  Term: '{snippet['term']}'")
            print(f"  Page: {snippet['page']}")
            print(f"  Context: '{snippet['context']}'")
        
        print("✅ EnhancedPDFProcessor test completed")
        
    except Exception as e:
        print(f"❌ EnhancedPDFProcessor test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_text_matching()
    test_with_enhanced_processor() 