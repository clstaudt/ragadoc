#!/usr/bin/env python3
"""
Debug script to test why text matching is failing
"""

def debug_text_matching():
    """Debug the specific case from the user's example"""
    
    # The quoted text from the AI response (from the screenshot)
    quoted_text = "Aktuell arbeite ich als kindheitspädagogische Fachkraft in einer Kita und begleite dort Kinder mit viel Engagement in ihrer Entwicklung."
    
    # Sample document text (we need to see what the actual extracted text looks like)
    # This is just a sample - we need to check the actual extracted text
    document_sample = """
    und dabei stets die pädagogische Begleitung von Kindern und ihren Familien – unter anderem Pflegefamilien, 
    Kinder mit Down-Syndrom und deren Familien sowie Geschwister von Kindern mit Behinderungen. Aktuell 
    arbeite ich als kindheitspädagogische Fachkraft in einer Kita und begleite dort Kinder mit viel Engagement in 
    ihrer Entwicklung. Zusätzlich bilde ich mich im Bereich Kinderschutz und Kindeswohlgefährdung fort, da mir 
    diese Themen besonders am Herzen liegen.
    """
    
    print("🔧 Debug Text Matching")
    print("=" * 60)
    
    print(f"\n📝 Quoted text (AI response):")
    print(f"'{quoted_text}'")
    print(f"Length: {len(quoted_text)} chars")
    
    print(f"\n📄 Document sample:")
    print(f"'{document_sample}'")
    print(f"Length: {len(document_sample)} chars")
    
    # Test 1: Exact match
    print(f"\n🔍 Test 1: Exact match")
    if quoted_text in document_sample:
        print("✅ Exact match found!")
    else:
        print("❌ No exact match")
    
    # Test 2: Case insensitive match
    print(f"\n🔍 Test 2: Case insensitive match")
    if quoted_text.lower() in document_sample.lower():
        print("✅ Case insensitive match found!")
    else:
        print("❌ No case insensitive match")
    
    # Test 3: Character by character comparison
    print(f"\n🔍 Test 3: Character analysis")
    quoted_lower = quoted_text.lower()
    doc_lower = document_sample.lower()
    
    # Find the best partial match
    best_match_pos = -1
    best_match_len = 0
    
    for i in range(len(doc_lower) - 20):  # Need at least 20 chars
        for j in range(min(100, len(quoted_lower))):  # Check up to 100 chars
            if i + j < len(doc_lower) and quoted_lower[:j+1] in doc_lower[i:i+j+50]:
                if j > best_match_len:
                    best_match_len = j
                    best_match_pos = i
    
    if best_match_pos >= 0:
        print(f"✅ Best partial match: {best_match_len} chars at position {best_match_pos}")
        partial_match = doc_lower[best_match_pos:best_match_pos+best_match_len+1]
        print(f"Matched text: '{partial_match}'")
    else:
        print("❌ No partial match found")
    
    # Test 4: Word by word analysis
    print(f"\n🔍 Test 4: Word analysis")
    quoted_words = quoted_text.lower().split()
    doc_words = document_sample.lower().split()
    
    print(f"Quoted words: {len(quoted_words)}")
    print(f"Doc words: {len(doc_words)}")
    
    matching_words = []
    for word in quoted_words:
        if word in doc_words:
            matching_words.append(word)
    
    print(f"Matching words: {len(matching_words)}/{len(quoted_words)}")
    print(f"Match ratio: {len(matching_words)/len(quoted_words)*100:.1f}%")
    
    if len(matching_words) >= len(quoted_words) * 0.8:  # 80% match
        print("✅ Good word match - should be highlightable")
    else:
        print("❌ Poor word match")
        print(f"Non-matching words: {[w for w in quoted_words if w not in doc_words]}")

if __name__ == "__main__":
    debug_text_matching() 