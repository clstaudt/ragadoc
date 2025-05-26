# Ragnarok - PDF Chat with Local LLM

A Streamlit-based application for chatting with PDF documents using local Large Language Models via Ollama. Features intelligent citation highlighting and multi-document chat sessions.

## Features

- **Local LLM Integration**: Chat with documents using Ollama models (no cloud dependency)
- **Smart Citations**: Automatically highlights referenced text in PDFs with contextual evidence
- **Multi-Chat Support**: Manage multiple chat sessions with different documents
- **PDF Processing**: Robust text extraction from PDF documents
- **Interactive PDF Viewer**: View documents with embedded highlights
- **Docker Support**: Easy deployment with Docker Compose

## Prerequisites

- **Python 3.8+**
- **Ollama**: Install and configure Ollama locally
  ```bash
  # Install Ollama (macOS/Linux)
  curl -fsSL https://ollama.ai/install.sh | sh
  
  # Pull a model (required)
  ollama pull llama3.2
  ```

## Installation

### Option 1: Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ragnarok
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install as a package (optional)**
   ```bash
   # For development
   pip install -e .
   
   # Or for production
   pip install .
   ```

4. **Start Ollama**
   ```bash
   ollama serve
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Access the application**
   - Open your browser to `http://localhost:8501`

### Option 2: Docker Setup

Quick Docker Compose setup for running with external Ollama.

1. **Start Ollama with Docker-compatible configuration**
   ```bash
   # Important: Ollama must accept connections from Docker containers
   OLLAMA_HOST=0.0.0.0:11434 ollama serve
   ```

2. **Build and run the application**
   ```bash
   # Build and start the container
   docker-compose up -d --build
   
   # View logs (optional)
   docker-compose logs -f app
   ```

3. **Access the application**
   - **URL:** http://localhost:8501
   - The app will automatically connect to your local Ollama instance

## Usage

1. **Upload a PDF document**
   - Click the file uploader in the main interface
   - Select a PDF file from your computer
   - Wait for text extraction to complete

2. **Start chatting**
   - Type your questions about the document
   - The AI will respond with citations from the document
   - Citations are automatically highlighted in the PDF viewer

3. **Smart Citations (Optional)**
   - Toggle "Smart Citations" in the sidebar
   - View evidence snippets directly below AI responses
   - See highlighted text with page thumbnails

4. **Manage chat sessions**
   - Create new chats for different documents
   - Switch between chat sessions in the sidebar
   - Delete old chats when no longer needed

## Docker Management

```bash
# Stop the application
docker-compose down

# Restart with rebuild
docker-compose down && docker-compose up -d --build

# View real-time logs
docker-compose logs -f app

# Check container status
docker-compose ps
```

## Configuration

- **Memory**: Container uses minimal resources (~500MB)
- **Models**: Download models before using: `ollama pull <model-name>`
- **Port**: Change port in `docker-compose.yml` if 8501 is in use

## Troubleshooting

### Connection Issues
```bash
# 1. Verify Ollama is accessible
curl http://localhost:11434/api/version

# 2. Check Ollama is configured correctly
# Must run with: OLLAMA_HOST=0.0.0.0:11434 ollama serve

# 3. Check app logs
docker-compose logs app
```

### Common Solutions
- **"No models found"**: Start Ollama with `OLLAMA_HOST=0.0.0.0:11434`
- **"Can't connect"**: Restart both Ollama and the container  
- **Port 8501 in use**: Change port in docker-compose.yml: `"8502:8501"`

## Dependencies

- **streamlit**: Web application framework
- **ollama**: Local LLM integration
- **pdfplumber**: PDF text extraction
- **PyMuPDF**: Advanced PDF processing and highlighting
- **streamlit-pdf-viewer**: Interactive PDF display

## Development

### Project Structure
```
ragnarok/
├── app.py                  # Main Streamlit application
├── ragnarok/              # Core package
│   ├── __init__.py
│   ├── core.py
│   └── enhanced_pdf_processor.py
├── tests/                 # Test suite
├── examples/              # Example files
├── requirements.txt       # Python dependencies
├── docker-compose.yml     # Docker setup
└── Dockerfile            # Container definition
```

### Running Tests
```bash
pytest tests/
```

### Environment Setup
```bash
# Create conda environment (optional)
conda env create -f environment.yml
conda activate ragnarok
```

## Notes

- **Data Storage**: Chat history is stored in memory (lost on restart)
- **Performance**: First model load may take 30+ seconds
- **Compatibility**: Tested with Ollama 0.7.x and Python 3.8+
- **PDF Support**: Handles most standard PDF formats

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details 