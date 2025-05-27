#!/usr/bin/env python3
"""
Simplified PDF Extraction Demo
Shows the streamlined approach using only PyMuPDF4LLM for high-quality structure preservation
"""

import sys
import os
from pathlib import Path

# Add the ragnarok directory to the path
sys.path.append(str(Path(__file__).parent / "ragnarok"))

from enhanced_pdf_processor import EnhancedPDFProcessor

def demo_simplified_extraction(pdf_path: str):
    """Demonstrate the simplified PDF extraction approach"""
    
    print("🚀 Simplified PDF Extraction Demo")
    print("=" * 50)
    
    # Load PDF
    try:
        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()
        print(f"✅ Loaded PDF: {pdf_path}")
    except FileNotFoundError:
        print(f"❌ PDF file not found: {pdf_path}")
        return
    
    # Create processor
    processor = EnhancedPDFProcessor(pdf_bytes)
    
    print(f"📄 Document has {processor.doc.page_count} pages")
    print()
    
    # Extract text using the simplified approach
    print("🔍 Extracting text with PyMuPDF4LLM...")
    extracted_text = processor.extract_full_text()
    
    print(f"✅ Extraction complete!")
    print(f"📊 Extracted {len(extracted_text)} characters")
    print()
    
    # Show structure preview
    print("📋 Document Structure Preview:")
    print("-" * 30)
    
    lines = extracted_text.split('\n')
    header_count = 0
    
    for line in lines[:50]:  # First 50 lines
        line = line.strip()
        if line.startswith('#'):
            header_count += 1
            level = len(line) - len(line.lstrip('#'))
            title = line.lstrip('#').strip()
            indent = "  " * (level - 1)
            print(f"{indent}📌 {title}")
        elif line and not line.startswith('#') and len(line) > 50:
            # Show first substantial content line
            print(f"   📝 {line[:80]}...")
            break
    
    print()
    print(f"🎯 Found {header_count} headers in preview")
    
    # Show sections
    sections = processor.extract_sections()
    print(f"📚 Document has {len(sections)} sections:")
    for i, section_name in enumerate(list(sections.keys())[:5], 1):
        content_length = len(sections[section_name])
        print(f"  {i}. {section_name} ({content_length} chars)")
    
    if len(sections) > 5:
        print(f"  ... and {len(sections) - 5} more sections")
    
    print()
    print("✨ Key Benefits of Simplified Approach:")
    print("  • Single, reliable extraction method")
    print("  • Automatic structure detection")
    print("  • Optimized for LLM/RAG applications")
    print("  • Fast and accurate")
    print("  • No complex method selection needed")

if __name__ == "__main__":
    # Look for PDF files in the current directory
    pdf_files = list(Path(".").glob("*.pdf"))
    
    if not pdf_files:
        print("❌ No PDF files found in current directory")
        print("💡 Place a PDF file in this directory and run again")
        sys.exit(1)
    
    # Use the first PDF found
    pdf_path = str(pdf_files[0])
    demo_simplified_extraction(pdf_path) 