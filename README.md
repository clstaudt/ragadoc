# Ragnarok - PDF Chat with Local LLM

A Streamlit-based application for chatting with PDF documents using local Large Language Models via Ollama. Features intelligent citation highlighting and multi-document chat sessions.

## Quick Start

### Prerequisites
- **Python 3.8+**
- **Ollama**: Install from [https://ollama.ai](https://ollama.ai)

### Local Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start Ollama and pull a model**
   ```bash
   ollama serve
   ollama pull olmo2:7b  # or olmo2:13b for better performance
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open browser** to `http://localhost:8501`

### Docker Setup

1. **Start Ollama with Docker-compatible configuration**
   ```bash
   OLLAMA_HOST=0.0.0.0:11434 ollama serve
   ```

2. **Run with Docker Compose**
   ```bash
   docker-compose up -d --build
   ```

## Usage

1. **Upload a PDF** using the file uploader
2. **Ask questions** about the document
3. **View citations** highlighted directly in the PDF viewer
4. **Manage multiple chats** via the sidebar

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