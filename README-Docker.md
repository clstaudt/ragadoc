# Docker Setup for Ragnarok PDF Chat

Simple Docker Compose setup for running the Ragnarok PDF Chat application with external Ollama.

## Prerequisites

- Docker & Docker Compose
- Ollama installed locally

## Quick Start

### 1. Start Ollama with Docker-compatible configuration:
```bash
# Important: Ollama must accept connections from Docker containers
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

### 2. Build and run the application:
```bash
# Build and start the container
docker-compose up -d --build

# View logs (optional)
docker-compose logs -f app
```

### 3. Access the application:
- **URL:** http://localhost:8501
- The app will automatically connect to your local Ollama instance
- Upload a PDF and start chatting!

## Management Commands

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

## Requirements

- **Ollama Models:** Download models before using: `ollama pull llama3.2`
- **PDF Files:** The app processes PDF documents for Q&A
- **Memory:** Container uses minimal resources (~500MB)

## Troubleshooting

**Connection Issues:**
```bash
# 1. Verify Ollama is accessible
curl http://localhost:11434/api/version

# 2. Check Ollama is configured correctly
# Must run with: OLLAMA_HOST=0.0.0.0:11434 ollama serve

# 3. Check app logs
docker-compose logs app
```

**Common Solutions:**
- **"No models found"**: Start Ollama with `OLLAMA_HOST=0.0.0.0:11434`
- **"Can't connect"**: Restart both Ollama and the container
- **Port 8501 in use**: Change port in docker-compose.yml: `"8502:8501"`

## Notes

- **Data:** Chat history is stored in memory (lost on restart)
- **Performance:** First model load may take 30+ seconds
- **Compatibility:** Tested with Ollama 0.7.x and Docker Desktop 