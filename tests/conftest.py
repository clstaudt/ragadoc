import pytest
import fitz  # PyMuPDF
import io
from ragnarok import EnhancedPDFProcessor


@pytest.fixture
def sample_pdf_bytes():
    """Create a simple test PDF with sample text"""
    doc = fitz.open()  # New PDF
    page = doc.new_page()
    
    test_text = """
    This is a sample document for testing highlighting functionality.
    It contains "quoted text" and various technical terms.
    Machine learning and artificial intelligence are mentioned here.
    The document also has some German text: "Aktuell arbeite ich als kindheitspädagogische Fachkraft in einer Kita."
    """
    
    page.insert_text((50, 100), test_text)
    pdf_bytes = doc.tobytes()
    doc.close()
    
    return pdf_bytes


@pytest.fixture
def german_pdf_bytes():
    """Create a test PDF with German text for testing Unicode quote detection"""
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
    
    return pdf_bytes


@pytest.fixture
def pdf_processor(sample_pdf_bytes):
    """Create an EnhancedPDFProcessor instance with sample PDF"""
    return EnhancedPDFProcessor(sample_pdf_bytes)


@pytest.fixture
def german_pdf_processor(german_pdf_bytes):
    """Create an EnhancedPDFProcessor instance with German text PDF"""
    return EnhancedPDFProcessor(german_pdf_bytes)


@pytest.fixture
def sample_ai_response():
    """Sample AI response with quoted text"""
    return 'The document mentions "quoted text" and discusses various technical concepts like machine learning.'


@pytest.fixture
def german_ai_response():
    """Sample AI response with German quoted text"""
    return '''Sena Neriman Demirbas arbeitet zurzeit als kindheitspädagogische Fachkraft in einer Kita. Dies wird aus dem Text "Aktuell arbeite ich als kindheitspädagogische Fachkraft in einer Kita" entnommen.'''


@pytest.fixture
def unicode_quote_samples():
    """Various Unicode quote samples for testing"""
    return [
        'Normal ASCII quotes: "test quote"',
        'Smart quotes: \u201ctest quote\u201d',
        'German quotes: \u201etest quote\u201c',
        'French guillemets: \u00bbtest quote\u00ab',
        'Unicode example: \u201cAktuell arbeite ich als kindheitspädagogische FachkRAFT in einer KITA und begleite dort Kinder mit viel ENGAGEMENT in ihrer Entwicklung.\u201d'
    ] 