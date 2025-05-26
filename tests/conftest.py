"""
Simple test fixtures for smoke tests
"""
import pytest
import fitz


@pytest.fixture
def simple_pdf_bytes():
    """Create a simple test PDF"""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 100), "Simple test document for smoke testing.")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes 