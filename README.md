# Ragnarok - Enhanced PDF Processing

A powerful PDF processing system with high-quality text extraction and structure preservation, optimized for LLM/RAG applications.

## Key Features

- **High-Quality Text Extraction**: Uses PyMuPDF4LLM for superior structure preservation
- **Automatic Structure Detection**: Headers, tables, lists, and formatting automatically detected
- **LLM/RAG Optimized**: Specifically designed for AI applications
- **Local Processing**: All processing happens locally, no external service calls
- **Citation Highlighting**: Smart PDF highlighting for AI-generated citations

## PDF Extraction Capabilities

The system uses **PyMuPDF4LLM** as the primary extraction method because it:

- ✅ **Automatically detects document structure** (headers, tables, lists)
- ✅ **Preserves formatting** (bold, italic, etc.)
- ✅ **Optimized for LLM applications** 
- ✅ **Fast and reliable** (15s vs 2m30s compared to alternatives)
- ✅ **Local processing only**

### What You Get

- **Structured Markdown Output**: Headers marked with `#`, tables preserved, lists formatted
- **Section Extraction**: Automatic document section detection
- **Table of Contents**: Generated from document structure
- **Citation Highlighting**: AI responses can highlight source text in PDFs

## Installation

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Or use Conda**:
```bash
conda env create -f environment.yml
conda activate ragnarok
```

## Quick Start

### Basic Usage

```python
from ragnarok.enhanced_pdf_processor import EnhancedPDFProcessor

# Load PDF
with open('document.pdf', 'rb') as f:
    pdf_bytes = f.read()

# Create processor
processor = EnhancedPDFProcessor(pdf_bytes)

# Extract structured text
structured_text = processor.extract_full_text()
print(structured_text)  # Markdown with headers, tables, lists

# Get document sections
sections = processor.extract_sections()
for section_name, content in sections.items():
    print(f"## {section_name}")
    print(content[:200] + "...")
```

### Test the Extraction

Run the demo script to see the extraction in action:

```bash
python simplified_extraction_demo.py
```

This will:
- Find PDF files in the current directory
- Extract text with full structure preservation
- Show document sections and headers
- Display extraction statistics

## Dependencies

### Core Libraries
- **PyMuPDF4LLM** (>=0.0.5) - High-quality PDF to markdown conversion
- **PyMuPDF** (>=1.23.0) - PDF processing and highlighting
- **Streamlit** - Web interface
- **Loguru** - Logging

### Why PyMuPDF4LLM?

PyMuPDF4LLM was chosen as the primary extraction method because:

1. **Purpose-Built for LLM/RAG**: Specifically designed for AI applications
2. **Superior Structure Detection**: Automatically handles headers, tables, lists
3. **Performance**: Much faster than alternatives (15s vs 2m30s)
4. **Reliability**: Consistent, high-quality output
5. **Local Processing**: No external API calls required

## Example Output

**Before** (basic extraction):
```
Introduction This document describes the new system. Features The system has many features. Performance Tests show good performance.
```

**After** (PyMuPDF4LLM):
```markdown
# Introduction

This document describes the new system.

## Features

The system has many features:
- Feature 1
- Feature 2
- Feature 3

## Performance

Tests show good performance:

| Metric | Value |
|--------|-------|
| Speed  | Fast  |
| Memory | Low   |
```

## Architecture

The system is built around a single, reliable extraction method:

```
PDF Input → PyMuPDF4LLM → Structured Markdown → Sections/TOC
                ↓
         (fallback if needed)
                ↓
         Basic Text Extraction
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with various PDF types
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Testing

```bash
python -m pytest tests/ -v
```

## Configuration

The application automatically detects its environment:
- **Direct execution**: Uses `http://localhost:11434`
- **Docker**: Uses `http://host.docker.internal:11434`

## Troubleshooting

### Connection Issues
```bash
# Verify Ollama is running
curl http://localhost:11434/api/version

# For Docker: ensure Ollama accepts external connections
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

### Common Solutions
- **"No models found"**: Pull a model with `ollama pull olmo2:7b`
- **"Can't connect"**: Restart Ollama with correct host settings
- **Upload fails**: Use "🗑️ Clear Upload" button to reset file state