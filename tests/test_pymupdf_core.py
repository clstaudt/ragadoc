"""
Test core PyMuPDF functionality
"""
import pytest
import fitz
from PIL import Image
import io


class TestPyMuPDFCore:
    """Test basic PyMuPDF functionality"""

    def test_import_pymupdf(self):
        """Test that PyMuPDF can be imported"""
        import fitz
        assert fitz is not None

    def test_create_empty_pdf(self):
        """Test creating a new empty PDF document"""
        doc = fitz.open()
        assert doc is not None
        assert doc.page_count == 0
        doc.close()

    def test_add_page_and_text(self):
        """Test adding a page and text to PDF"""
        doc = fitz.open()
        page = doc.new_page()
        
        text = "This is a test document for highlighting functionality."
        page.insert_text((50, 100), text)
        
        assert doc.page_count == 1
        extracted_text = page.get_text()
        assert "test document" in extracted_text
        
        doc.close()

    def test_text_search(self):
        """Test searching for text in PDF"""
        doc = fitz.open()
        page = doc.new_page()
        
        text = "This is a test document for highlighting functionality."
        page.insert_text((50, 100), text)
        
        # Search for text
        text_instances = page.search_for("test document", quads=True)
        assert len(text_instances) >= 1
        
        doc.close()

    def test_highlight_annotation(self):
        """Test adding highlight annotations"""
        doc = fitz.open()
        page = doc.new_page()
        
        text = "This is a test document for highlighting functionality."
        page.insert_text((50, 100), text)
        
        # Search and highlight
        text_instances = page.search_for("test document", quads=True)
        assert len(text_instances) >= 1
        
        highlight = page.add_highlight_annot(text_instances[0])
        highlight.set_colors(stroke=(1, 1, 0))  # Yellow
        highlight.update()
        
        # Check that annotation was added
        annotations = page.annots()
        annotation_count = len(list(annotations))
        assert annotation_count >= 1
        
        doc.close()

    def test_pixmap_creation(self):
        """Test creating pixmap from page"""
        doc = fitz.open()
        page = doc.new_page()
        
        text = "Test pixmap creation"
        page.insert_text((50, 100), text)
        
        # Create pixmap
        pix = page.get_pixmap(dpi=150)
        assert pix is not None
        assert pix.width > 0
        assert pix.height > 0
        
        # Test PNG conversion
        png_bytes = pix.tobytes("png")
        assert len(png_bytes) > 0
        
        # Verify it's a valid PNG
        img = Image.open(io.BytesIO(png_bytes))
        assert img.format == "PNG"
        
        doc.close()

    def test_pdf_serialization(self):
        """Test converting PDF to bytes and back"""
        doc = fitz.open()
        page = doc.new_page()
        
        text = "Test PDF serialization"
        page.insert_text((50, 100), text)
        
        # Convert to bytes
        pdf_bytes = doc.tobytes()
        assert len(pdf_bytes) > 0
        doc.close()
        
        # Recreate from bytes
        doc2 = fitz.open(stream=pdf_bytes, filetype="pdf")
        assert doc2.page_count == 1
        
        page2 = doc2[0]
        extracted_text = page2.get_text()
        assert "Test PDF serialization" in extracted_text
        
        doc2.close()

    def test_multiple_highlights(self):
        """Test adding multiple highlights to a page"""
        doc = fitz.open()
        page = doc.new_page()
        
        text = "This document contains multiple terms that need highlighting: test, document, highlighting."
        page.insert_text((50, 100), text)
        
        search_terms = ["test", "document", "highlighting"]
        highlight_count = 0
        
        for term in search_terms:
            instances = page.search_for(term, quads=True)
            for inst in instances:
                highlight = page.add_highlight_annot(inst)
                highlight.set_colors(stroke=(1, 1, 0))
                highlight.update()
                highlight_count += 1
        
        assert highlight_count >= 3  # Should find at least one instance of each term
        
        doc.close()

    def test_text_extraction_multipage(self):
        """Test text extraction from multiple pages"""
        doc = fitz.open()
        
        # Add multiple pages with different content
        page1 = doc.new_page()
        page1.insert_text((50, 100), "Content of page one")
        
        page2 = doc.new_page()
        page2.insert_text((50, 100), "Content of page two")
        
        assert doc.page_count == 2
        
        # Extract text from each page
        text1 = page1.get_text()
        text2 = page2.get_text()
        
        assert "page one" in text1
        assert "page two" in text2
        assert "page one" not in text2
        assert "page two" not in text1
        
        doc.close()

    def test_resource_cleanup(self):
        """Test that documents can be properly closed"""
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 100), "Test cleanup")
        
        # Should not raise exception
        doc.close()
        
        # Creating a new document should work
        doc2 = fitz.open()
        assert doc2 is not None
        doc2.close() 