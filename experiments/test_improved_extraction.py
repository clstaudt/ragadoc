#!/usr/bin/env python3
"""
Test Improved PDF Extraction Quality
Tests the new multi-method approach with fallback strategies
"""

import sys
import os
from pathlib import Path

# Add the ragnarok directory to the path
sys.path.append(str(Path(__file__).parent / "ragnarok"))

from enhanced_pdf_processor import EnhancedPDFProcessor

def analyze_text_quality(text: str, method_name: str):
    """Analyze the quality of extracted text"""
    if not text or len(text.strip()) < 50:
        print(f"❌ {method_name}: Insufficient text extracted")
        return False
    
    # Structure indicators
    structure_indicators = {
        "Headers": text.count('#'),
        "Tables": text.count('|'),
        "Lists": text.count('- ') + text.count('* '),
        "Bold text": text.count('**'),
        "Code blocks": text.count('```'),
        "Paragraphs": text.count('\n\n'),
        "Total lines": text.count('\n')
    }
    
    print(f"✅ {method_name}: Success")
    print(f"   Length: {len(text)} characters")
    print(f"   Structure indicators:")
    for indicator, count in structure_indicators.items():
        print(f"     {indicator}: {count}")
    
    # Show sample text
    print(f"   Sample text (first 300 chars):")
    sample = text[:300].replace('\n', '\\n')
    print(f"     \"{sample}{'...' if len(text) > 300 else ''}\"")
    
    return True

def test_extraction_quality(pdf_path: str):
    """Test and compare extraction quality"""
    
    print("🔬 Enhanced PDF Extraction Quality Test")
    print("=" * 60)
    
    # Load PDF
    try:
        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()
        print(f"✅ Loaded PDF: {pdf_path}")
    except FileNotFoundError:
        print(f"❌ PDF file not found: {pdf_path}")
        print("Please provide a valid PDF file path")
        return
    
    # Create processor
    processor = EnhancedPDFProcessor(pdf_bytes)
    
    print(f"\n📄 Document Info:")
    print(f"   Pages: {processor.doc.page_count}")
    print(f"   Title: {processor.doc.metadata.get('title', 'Unknown')}")
    print(f"   Has TOC: {bool(processor.doc.get_toc())}")
    
    # Test the main extraction method (which tries multiple approaches)
    print(f"\n🚀 Testing Enhanced Multi-Method Extraction...")
    print("-" * 60)
    
    try:
        extracted_text = processor.extract_full_text()
        analyze_text_quality(extracted_text, "Enhanced Multi-Method")
        
        # Save the result for inspection
        output_file = "extracted_text_sample.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(extracted_text)
        print(f"\n💾 Full extracted text saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Enhanced extraction failed: {e}")
    
    # Test individual methods for comparison
    print(f"\n🔍 Testing Individual Methods...")
    print("-" * 60)
    
    # Test basic fallback
    try:
        basic_text = processor._extract_basic_fallback()
        analyze_text_quality(basic_text, "Basic Fallback")
    except Exception as e:
        print(f"❌ Basic extraction failed: {e}")
    
    # Test document sections
    print(f"\n📑 Testing Section Extraction...")
    print("-" * 60)
    
    try:
        sections = processor.extract_sections()
        print(f"✅ Sections extracted: {len(sections)}")
        for section_name, section_text in list(sections.items())[:3]:  # Show first 3 sections
            print(f"   Section: '{section_name}' ({len(section_text)} chars)")
    except Exception as e:
        print(f"❌ Section extraction failed: {e}")
    
    # Test TOC extraction
    print(f"\n📋 Testing Table of Contents...")
    print("-" * 60)
    
    try:
        toc = processor.extract_table_of_contents()
        print(f"✅ TOC entries found: {len(toc)}")
        for entry in toc[:5]:  # Show first 5 entries
            print(f"   Level {entry['level']}: {entry['title']}")
    except Exception as e:
        print(f"❌ TOC extraction failed: {e}")

def main():
    """Main function"""
    # You can specify a PDF file path here
    pdf_file = "sample.pdf"  # Change this to your PDF file
    
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
    
    if not os.path.exists(pdf_file):
        print(f"📁 Available files in current directory:")
        for file in os.listdir('.'):
            if file.endswith('.pdf'):
                print(f"   {file}")
        print(f"\nUsage: python {sys.argv[0]} <pdf_file>")
        return
    
    test_extraction_quality(pdf_file)

if __name__ == "__main__":
    main() 