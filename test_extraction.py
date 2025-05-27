#!/usr/bin/env python3
"""
Test script to demonstrate improved PDF text extraction with structure preservation
All processing is done locally - no external services required
"""

import sys
from pathlib import Path
from ragnarok.enhanced_pdf_processor import EnhancedPDFProcessor

def test_extraction_methods(pdf_path: str):
    """Test different extraction methods on a PDF file (all local processing)"""
    
    # Read PDF file
    with open(pdf_path, 'rb') as f:
        pdf_bytes = f.read()
    
    # Create processor
    processor = EnhancedPDFProcessor(pdf_bytes)
    
    print(f"Testing LOCAL extraction methods on: {pdf_path}")
    print("=" * 60)
    print("All processing is done locally - no external services used")
    print()
    
    # Get available methods
    methods = processor.get_extraction_methods()
    print(f"Available extraction methods: {methods}")
    print()
    
    # Method descriptions
    method_descriptions = {
        "high_quality_markdown": "PyMuPDF4LLM - Best choice for LLM/RAG applications (fast, accurate, local)",
        "basic_text": "Simple - Basic text extraction without formatting (fallback only)"
    }
    
    # Test each method
    for method in methods:
        print(f"\n{'='*20} {method.upper()} {'='*20}")
        print(f"Description: {method_descriptions.get(method, 'Unknown method')}")
        
        try:
            text = processor.extract_with_method(method)
            
            # Show first 500 characters
            preview = text[:500] + "..." if len(text) > 500 else text
            print(f"Extracted {len(text)} characters")
            print(f"Preview:\n{preview}")
            
            # Count sections/headings
            lines = text.split('\n')
            headings = [line for line in lines if line.strip().startswith('#')]
            tables = [line for line in lines if '|' in line and line.count('|') >= 2]
            
            print(f"Found {len(headings)} headings")
            print(f"Found {len(tables)} potential table rows")
            
            if headings:
                print("Sample headings:")
                for heading in headings[:3]:  # Show first 3 headings
                    print(f"  {heading.strip()}")
            
        except Exception as e:
            print(f"Error with {method}: {e}")
    
    # Test metadata extraction
    print(f"\n{'='*20} METADATA {'='*20}")
    try:
        metadata = processor.get_document_metadata()
        print(f"Pages: {metadata['page_count']}")
        print(f"Title: {metadata['title']}")
        print(f"Author: {metadata['author']}")
        print(f"Has TOC: {metadata['has_outline']}")
        print(f"Sections found: {len(metadata['sections'])}")
        if metadata['sections']:
            print("Section names:")
            for section in metadata['sections'][:5]:  # Show first 5 sections
                print(f"  - {section}")
    except Exception as e:
        print(f"Error extracting metadata: {e}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python test_extraction.py <pdf_file>")
        print("\nThis script tests local PDF text extraction methods.")
        print("No external services or internet connection required.")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    if not Path(pdf_path).exists():
        print(f"Error: File {pdf_path} not found")
        sys.exit(1)
    
    test_extraction_methods(pdf_path)

if __name__ == "__main__":
    main() 