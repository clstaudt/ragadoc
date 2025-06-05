"""
Simple test fixtures for smoke tests
"""
import pytest
import fitz
from .test_utils import check_test_environment


# Ensure test environment is ready when tests start
if not check_test_environment():
    pytest.exit("Test environment setup failed. Please check Ollama installation and service.", returncode=1)


@pytest.fixture
def simple_pdf_bytes():
    """Create a simple test PDF"""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 100), "Simple test document for smoke testing.")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes 