# Ragnarok PDF Processing Tests

This directory contains the test suite for the Ragnarok PDF Processing application, specifically focusing on the Enhanced PDF Processor functionality with PyMuPDF highlighting capabilities.

## Test Structure

The tests are organized into the following modules:

### `test_pymupdf_core.py`
Tests the core PyMuPDF functionality:
- PDF creation and manipulation
- Text insertion and extraction
- Highlighting annotations
- Pixmap generation
- Resource cleanup

### `test_enhanced_pdf_processor.py`
Tests the main `EnhancedPDFProcessor` class:
- PDF processor initialization
- Text extraction with positions
- Search and highlight functionality
- AI response highlight extraction
- Fuzzy text matching
- Context extraction
- Snippet generation

### `test_unicode_quotes.py`
Tests Unicode quote detection:
- ASCII quotes (`"text"`)
- Smart quotes (`"text"`)
- German quotes (`„text"`)
- French guillemets (`»text«`)
- Single quotes (`'text'`)
- Backticks (`\`text\``)
- Complex Unicode examples

### `test_text_matching.py`
Tests text matching and search logic:
- Exact and fuzzy text matching
- Case insensitive matching
- Whitespace normalization
- Punctuation handling
- Partial phrase matching
- Context extraction
- German text processing

## Running Tests

### Basic Test Run
```bash
# Run all tests (uses configuration from pytest.ini)
pytest

# Or be explicit about the test directory
pytest tests/
```

### With Coverage Report
```bash
# Run tests with coverage
pytest --cov=enhanced_pdf_processor

# Generate HTML coverage report
pytest --cov=enhanced_pdf_processor --cov-report=html

# Coverage with missing lines shown
pytest --cov=enhanced_pdf_processor --cov-report=term-missing
```

### Run Specific Test Modules
```bash
# Test only PyMuPDF core functionality
pytest tests/test_pymupdf_core.py

# Test only the EnhancedPDFProcessor
pytest tests/test_enhanced_pdf_processor.py

# Test only Unicode quote detection
pytest tests/test_unicode_quotes.py

# Test only text matching logic
pytest tests/test_text_matching.py
```

### Useful Pytest Options
```bash
# Stop on first failure
pytest -x

# Run tests in parallel (requires pytest-xdist)
pytest -n auto

# Show local variables in tracebacks
pytest -l

# Run only failed tests from last run
pytest --lf

# Verbose output with full test names
pytest -vv
```

## Test Configuration

The project uses `pytest.ini` for configuration with these defaults:
- Verbose output (`-v`)
- Short traceback format (`--tb=short`)
- Colored output (`--color=yes`)
- Strict marker checking
- Warning suppression for cleaner output

## Test Fixtures

The tests use several fixtures defined in `conftest.py`:

- `sample_pdf_bytes`: Creates a simple test PDF with English text
- `german_pdf_bytes`: Creates a test PDF with German text
- `pdf_processor`: EnhancedPDFProcessor instance with sample PDF
- `german_pdf_processor`: EnhancedPDFProcessor instance with German PDF
- `sample_ai_response`: Sample AI response with quoted text
- `german_ai_response`: German AI response with quoted text
- `unicode_quote_samples`: Various Unicode quote examples

## Requirements

The tests require the following packages (install with `pip install -r requirements.txt`):

- `pytest`: Test framework
- `pytest-cov`: Coverage reporting
- `PyMuPDF (fitz)`: PDF processing
- `Pillow`: Image processing
- `enhanced_pdf_processor`: The main module being tested

## Test Coverage

The tests cover:

✅ **Core PyMuPDF Operations**
- Document creation and manipulation
- Text search and highlighting
- Annotation handling
- Image generation

✅ **Enhanced PDF Processor Features**
- Text extraction and positioning
- AI response quote detection
- Fuzzy text matching
- Context-aware highlighting
- Multi-language support

✅ **Unicode and International Text**
- Various quote styles
- German text processing
- Special character handling
- Normalization logic

✅ **Edge Cases**
- Empty inputs
- Special characters
- Large documents
- Resource cleanup

## Notes

- Tests create temporary PDF documents in memory
- No external files are required
- All tests clean up resources properly
- Tests are designed to be independent and can run in any order 