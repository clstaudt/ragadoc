<div align="center">
  <img src="assets/logo.png" alt="Ragadoc Logo" width="120" height="120">
  
  # ragadoc
  
  **AI document assistant that answers questions about your PDFs with citations and highlights.**
  
  <p><em>Privacy-first Streamlit app for chatting with documents using local AI models.</em></p>
</div>

## ✨ Features

- 🤖 **AI Document Q&A** - Natural language questions about your PDFs
- 📍 **Citation Grounding** - Answers include specific citations from your document
- 🎯 **PDF Highlighting** - Citations automatically highlighted in the PDF
- 🔒 **Complete Privacy** - Local AI models only, documents never leave your computer

<div align="center">
  <img src="assets/screenshot_01.png" alt="Ragadoc Main Interface" width="80%">
</div>

<div align="center">
  <img src="assets/screenshot_02.png" alt="Ragadoc Document Analysis" width="80%">
</div>

> ⚠️ **Early Development** - This is a proof of concept. Expect incomplete features and potential breaking changes.

## 🚀 Quick Start

### Prerequisites

**1. Install and start Ollama**:
```bash
# macOS
brew install ollama

# Or download from https://ollama.com
```

**2. Pull models** (recommendations: `nomic-embed-text` for embeddings, `olmo3:7b` or `olmo3:32b` for chat):
```bash
ollama serve
ollama pull <embedding-model>
ollama pull <chat-model>
```

### Option 1: uv (Recommended)

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/clstaudt/ragadoc.git
cd ragadoc
uv sync
uv run streamlit run app.py
```

Open `http://localhost:8501`

### Option 2: Docker

```bash
git clone https://github.com/clstaudt/ragadoc.git
cd ragadoc
docker-compose up --build
```

Open `http://localhost:8501`

## ⚙️ Configuration

Copy `env.example` to `.env` to configure Ollama instances (local or remote).

## 📖 Usage

1. **Select Models** - Choose chat and embedding models in the sidebar
2. **Upload a PDF** - Drag and drop your document (extraction and indexing happens automatically)
3. **Chat** - Ask questions in natural language
4. **View Citations** - Answers include citations with highlighted PDF passages shown below

**Expert Mode** (optional): Toggle in sidebar to adjust RAG parameters like chunk size, similarity threshold, and retrieval count.

## 🏗️ Architecture

```
PDF Upload → Text Extraction → Chunking → Vector Embeddings
                                               ↓
User Question → Semantic Search → Context Retrieval → AI Response
                                               ↓
                                    Citation Highlighting
```

**Tech Stack:** Streamlit • Ollama • ChromaDB • PyMuPDF4LLM • uv

## 🐛 Troubleshooting

**Ollama Connection Error**
```bash
curl http://localhost:11434/api/version
# For Docker: OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

**Slow Performance** - Try a smaller model or reduce chunk size in RAG settings.

## 📄 License

GPL License - see [LICENSE](LICENSE).

---

**⭐ Star this repo if Ragadoc helps you work with your documents!**
