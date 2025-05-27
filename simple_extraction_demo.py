#!/usr/bin/env python3
"""
Simple demo showing how PyMuPDF4LLM and Marker handle all structure detection automatically
No regex patterns or custom logic needed!
"""

import sys
from pathlib import Path
from ragnarok.enhanced_pdf_processor import EnhancedPDFProcessor

def simple_demo(pdf_path: str):
    """Demonstrate how simple the extraction is now"""
    
    # Read PDF
    with open(pdf_path, 'rb') as f:
        pdf_bytes = f.read()
    
    # Create processor
    processor = EnhancedPDFProcessor(pdf_bytes)
    
    print("🚀 Simple PDF Extraction Demo")
    print("=" * 50)
    print("Libraries handle ALL structure detection automatically!")
    print()
    
    # Method 1: PyMuPDF4LLM (recommended)
    print("📄 PyMuPDF4LLM Extraction:")
    print("-" * 30)
    
    markdown_text = processor.extract_high_quality_markdown()
    
    print(f"✅ Extracted {len(markdown_text)} characters")
    print(f"✅ Found {markdown_text.count('#')} headings automatically")
    print(f"✅ Found {markdown_text.count('|')} table elements automatically")
    print(f"✅ Found {markdown_text.count('- ')} list items automatically")
    
    # Show first few lines to demonstrate structure
    lines = markdown_text.split('\n')[:20]
    print("\n📋 First 20 lines (showing automatic structure detection):")
    for i, line in enumerate(lines, 1):
        if line.strip():
            print(f"{i:2d}: {line}")
    
    print("\n🎯 Key Point: No regex patterns needed!")
    print("   PyMuPDF4LLM automatically detects:")
    print("   • Headers (# ## ### ####)")
    print("   • Tables (| format)")
    print("   • Lists (- and numbered)")
    print("   • Text formatting (bold, italic)")
    print("   • Reading order and layout")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python simple_extraction_demo.py <pdf_file>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    if not Path(pdf_path).exists():
        print(f"Error: File {pdf_path} not found")
        sys.exit(1)
    
    simple_demo(pdf_path) 