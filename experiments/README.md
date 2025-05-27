# Experiments and Development Tools

This folder contains development tools, debugging scripts, and experimental code that are useful for development but not part of the main application or test suite.

## Files

### `debug_context.py`
**Purpose**: Debug and analyze context window calculations for Ollama models.

**Usage**:
```bash
# Debug context detection for all available models
python experiments/debug_context.py

# Test context calculation with sample text
python experiments/debug_context.py test <model_name> <sample_text>
```

**Features**:
- Detects context lengths for available Ollama models
- Compares actual vs detected context lengths
- Tests document fitting within context windows
- Shows token usage statistics and warnings

### `simplified_extraction_demo.py`
**Purpose**: Demonstrates the streamlined PDF extraction approach using PyMuPDF4LLM.

**Usage**:
```bash
python experiments/simplified_extraction_demo.py
```

**Features**:
- Shows automatic structure detection
- Demonstrates section extraction
- Highlights benefits of the simplified approach
- Automatically finds PDF files in current directory

### `test_improved_extraction.py`
**Purpose**: Comprehensive testing tool for PDF extraction quality and methods.

**Usage**:
```bash
python experiments/test_improved_extraction.py <pdf_file>
```

**Features**:
- Tests multiple extraction methods
- Analyzes text quality and structure preservation
- Compares different approaches
- Saves extracted text for inspection
- Shows document statistics and metadata

## When to Use These Tools

- **During development**: When working on extraction or context handling features
- **For debugging**: When troubleshooting issues with specific models or documents
- **For analysis**: When evaluating extraction quality or performance
- **For demonstrations**: When showing capabilities to stakeholders

## Integration with Main Codebase

These tools import from the main `ragnarok` package and `app.py`, so they test the actual production code. They're kept separate to avoid cluttering the main application while remaining useful for development. 