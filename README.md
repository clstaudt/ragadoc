# Ragadoc 🤖📄

**An AI document assistant that answers questions about your PDFs with citations and highlights them directly in the document.**

Ragadoc is a privacy-first Streamlit application that lets you chat with your documents using locally-run AI models. Ask questions, get grounded answers with citations, and see exactly where the information comes from with automatic PDF highlighting.

## ✨ Key Features

- 🤖 **AI Document Q&A** - Ask natural language questions about your PDFs
- 📍 **Citation Grounding** - Every answer includes specific citations from your document
- 🎯 **PDF Highlighting** - Citations are automatically highlighted in the original PDF
- 🔒 **Complete Privacy** - Uses only local AI models, your documents never leave your computer
- ⚡ **Fast Processing** - Optimized document parsing and retrieval system
- 🌐 **Easy Web Interface** - Simple Streamlit app, no technical knowledge required

## 🚀 Quick Start

### Model Selection Guide

Choose models based on your system capabilities:

| Model Type | Model Name | Size | RAM Required | Use Case |
|------------|------------|------|--------------|----------|
| **Embedding** | `nomic-embed-text` | ~274MB | 1GB | **Recommended** - General purpose |
| **Embedding** | `all-minilm` | ~23MB | 512MB | Lightweight alternative |
| **Chat** | `qwen3:14b` | ~8.5GB | 16GB | **Recommended**  |
| **Chat** | `llama3.1:8b` | ~4.7GB | 8GB | Balanced option |
| **Chat** | `mistral:latest` | ~4.1GB | 8GB | Quick responses |
| **Chat** | `phi3:mini` | ~2.3GB | 4GB | Low-resource systems |



### Prerequisites (Required for Both Installation Methods)

**1. Install Ollama** (for local AI models):
```bash
# macOS
brew install ollama

# Or download from https://ollama.com
```

**2. Start Ollama and install required models**:
```bash
ollama serve

# Install embedding model (required)
ollama pull nomic-embed-text

# Install a chat model (see recommendations above)
ollama pull qwen3:14b
```

### Installation Options

Choose your preferred installation method:

### Option 1: Direct Installation

**Additional Prerequisites:**
- Python 3.8+

**Installation Steps:**

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/ragadoc.git
   cd ragadoc
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
   Or with conda:
   ```bash
   conda env create -f environment.yml
   conda activate ragadoc
   ```

3. **Launch the application**:
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** to `http://localhost:8501`

### Option 2: Docker Installation

**Additional Prerequisites:**
- Docker and Docker Compose

**Installation Steps:**

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/ragadoc.git
   cd ragadoc
   ```

2. **Start with Docker Compose**:
   ```bash
   docker-compose up
   ```

3. **Open your browser** to `http://localhost:8501`

## 📖 How to Use

1. **Upload a PDF** - Drag and drop or browse for your document
2. **Select AI Model** - Choose from your locally installed Ollama models
3. **Start Chatting** - Ask questions about your document in natural language
4. **View Citations** - See highlighted text in the PDF that supports each answer
5. **Explore** - Continue the conversation to dive deeper into your document

### Example Questions

- "What are the main conclusions of this research paper?"
- "Summarize the financial results from Q3"
- "What methodology was used in the study?"
- "List all the recommendations mentioned"

## 🏗️ Architecture

Ragadoc uses a modern RAG (Retrieval-Augmented Generation) architecture:

```
PDF Upload → Text Extraction → Chunking → Vector Embeddings
                                               ↓
User Question → Semantic Search → Context Retrieval → AI Response
                                               ↓
                                    Citation Highlighting
```

**Tech Stack:**
- **Frontend**: Streamlit web interface
- **AI Models**: Ollama (local LLMs)
- **Vector DB**: ChromaDB for semantic search
- **PDF Processing**: PyMuPDF4LLM for structure-aware extraction
- **Embeddings**: nomic-embed-text model

## 🔧 Configuration

### Available Models

The app automatically detects your installed Ollama models. Popular choices:

- **qwen3:14b** - Well-balanced performance and accuracy (recommended)
- **llama3.1:8b** - Good alternative option
- **mistral:latest** - Fast and efficient
- **codellama:latest** - Good for technical documents

### Advanced Settings

Configure in the sidebar:
- **Chunk Size**: How much text to process at once (default: 512)
- **Chunk Overlap**: Text overlap between chunks (default: 50)
- **Top-K Results**: Number of relevant chunks to consider (default: 5)



## 🧪 Testing

Run the test suite:

```bash
pytest tests/ -v
```

Test the RAG system specifically:

```bash
python experiments/test_rag.py
```

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
pytest

# Run linting
flake8 ragadoc/
```

## 📋 Requirements

- Python 3.8+
- Ollama (for local AI models)
- 4GB+ RAM (depending on chosen model)
- PDF documents to analyze

## 🐛 Troubleshooting

### Common Issues

**Ollama Connection Error**
```bash
# Verify Ollama is running
curl http://localhost:11434/api/version

# If using Docker, ensure external access
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

**No Models Available**
```bash
# Install required models
ollama pull nomic-embed-text
ollama pull qwen3:14b
```

**Slow Performance**
- Try a smaller model like `mistral:latest`
- Reduce chunk size in settings
- Ensure sufficient RAM is available

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.com) for making local AI accessible
- [Streamlit](https://streamlit.io) for the amazing web framework
- [PyMuPDF](https://pymupdf.readthedocs.io/) for PDF processing
- [ChromaDB](https://www.trychroma.com/) for vector storage

---

**⭐ Star this repo if Ragadoc helps you work with your documents more effectively!**