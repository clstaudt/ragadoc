#!/usr/bin/env python3
"""
Simple test script to verify PyMuPDF highlighting functionality
"""

def test_imports():
    """Test that all required imports work"""
    try:
        import fitz  # PyMuPDF
        print("✅ PyMuPDF (fitz) imported successfully")
        
        import streamlit as st
        print("✅ Streamlit imported successfully")
        
        from enhanced_pdf_processor import EnhancedPDFProcessor
        print("✅ EnhancedPDFProcessor imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_pymupdf_basic():
    """Test basic PyMuPDF functionality"""
    try:
        import fitz
        
        # Create a simple test PDF in memory
        doc = fitz.open()  # New PDF
        page = doc.new_page()
        
        # Add some text
        text = "This is a test document for highlighting functionality."
        page.insert_text((50, 100), text)
        
        # Search for text
        text_instances = page.search_for("test document", quads=True)
        print(f"✅ Found {len(text_instances)} text instances")
        
        # Add highlight
        if text_instances:
            highlight = page.add_highlight_annot(text_instances[0])
            highlight.set_colors(stroke=(1, 1, 0))  # Yellow
            highlight.update()
            print("✅ Highlight annotation added successfully")
        
        # Get pixmap
        pix = page.get_pixmap(dpi=150)
        print(f"✅ Pixmap created: {pix.width}x{pix.height}")
        
        doc.close()
        return True
        
    except Exception as e:
        print(f"❌ PyMuPDF test failed: {e}")
        return False

def test_enhanced_processor():
    """Test the EnhancedPDFProcessor with a simple PDF"""
    try:
        import fitz
        from enhanced_pdf_processor import EnhancedPDFProcessor
        
        # Create a test PDF
        doc = fitz.open()
        page = doc.new_page()
        
        test_text = """
        This is a sample document for testing highlighting.
        It contains "quoted text" and various terms.
        Machine learning and artificial intelligence are mentioned here.
        """
        
        page.insert_text((50, 100), test_text)
        pdf_bytes = doc.tobytes()
        doc.close()
        
        # Test the processor
        processor = EnhancedPDFProcessor(pdf_bytes)
        
        # Test text extraction
        full_text = processor.extract_full_text()
        print(f"✅ Extracted {len(full_text)} characters of text")
        
        # Test AI response highlighting
        fake_ai_response = 'The document mentions "quoted text" and discusses various concepts.'
        highlight_terms = processor.create_ai_response_highlights(fake_ai_response, full_text)
        print(f"✅ Found {len(highlight_terms)} terms to highlight: {highlight_terms}")
        
        # Test highlighting
        if highlight_terms:
            highlighted_pdf = processor.search_and_highlight_text(highlight_terms)
            print(f"✅ Created highlighted PDF ({len(highlighted_pdf)} bytes)")
        
        return True
        
    except Exception as e:
        print(f"❌ EnhancedPDFProcessor test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🔧 Testing PyMuPDF Highlighting Setup")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("PyMuPDF Basic", test_pymupdf_basic),
        ("Enhanced Processor", test_enhanced_processor)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} test...")
        result = test_func()
        results.append(result)
        
        if result:
            print(f"🎉 {test_name} test PASSED")
        else:
            print(f"💥 {test_name} test FAILED")
    
    print("\n" + "=" * 50)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🌟 ALL TESTS PASSED ({passed}/{total})")
        print("\n✨ Your highlighting setup is ready to use!")
        print("\nNext steps:")
        print("1. Try: streamlit run demo_highlighting.py")
        print("2. Or: streamlit run app_enhanced.py")
    else:
        print(f"⚠️ SOME TESTS FAILED ({passed}/{total})")
        print("\n🔧 Check the error messages above and:")
        print("1. Ensure all dependencies are installed")
        print("2. Check for any import errors")
        print("3. Verify PyMuPDF installation")

if __name__ == "__main__":
    main() 