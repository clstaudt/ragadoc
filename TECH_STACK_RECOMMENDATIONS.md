# Ragnarok - Technology Stack Recommendations

## Overview
This document outlines technology choices for building Ragnarok, a document-based conversational AI application with local LLM integration.

## Core Requirements Recap
- Document ingestion from files, folders, databases
- Background worker processes for vector database ingestion
- Local LLM hosting via Ollama
- Web interface for chat and document inspection
- Citation and reference system

## Updated Focus: Simple Prototype First
For rapid prototyping and validation, we'll prioritize:
- **Streamlit** for quick UI development
- **Ollama embeddings** for consistency with LLM hosting
- **Python-native** solutions throughout the stack

## Technology Stack Options

### Backend Framework

#### Option 1: FastAPI (Python) ⭐ **RECOMMENDED**
**Pros:**
- Excellent async support for LLM and vector operations
- Built-in OpenAPI documentation
- Native Pydantic integration for data validation
- Excellent WebSocket support for real-time chat
- Rich ecosystem for AI/ML libraries
- Fast development and easy testing

**Cons:**
- Python performance limitations for CPU-intensive tasks
- Memory usage can be higher than compiled languages

**Use Case Fit:** Perfect for RAG applications with AI integrations

#### Option 2: Node.js (Express/Fastify)
**Pros:**
- Single language across frontend/backend
- Excellent async I/O performance
- Large ecosystem of packages
- Good WebSocket support

**Cons:**
- Limited AI/ML library ecosystem compared to Python
- Not ideal for compute-heavy document processing
- Weaker typing compared to Python+Pydantic

**Use Case Fit:** Good for rapid prototyping, less ideal for AI-heavy workloads

#### Option 3: Go (Gin/Fiber)
**Pros:**
- Excellent performance and low memory usage
- Great concurrency model
- Fast compilation and deployment
- Good for microservices

**Cons:**
- Limited AI/ML ecosystem
- Would need to integrate with Python for embeddings
- Steeper learning curve for AI developers

**Use Case Fit:** Better for high-performance microservices, overkill for this use case

### Frontend Framework

#### Option 1: Streamlit ⭐ **RECOMMENDED FOR PROTOTYPE**
**Pros:**
- Extremely rapid development (hours vs days)
- Pure Python - no separate frontend codebase
- Built-in components for chat interfaces
- Great for data visualization and document display
- Perfect for technical demos and prototypes
- File upload components built-in

**Cons:**
- Limited customization compared to React/Vue
- Not suitable for production web apps
- Performance limitations with many users
- Less polished UI compared to modern web frameworks

**Use Case Fit:** Perfect for prototyping and internal tools

#### Option 2: Next.js 14 (React + TypeScript)
**Pros:**
- Production-ready
- Highly customizable
- Better performance and UX
- Large component ecosystem

**Cons:**
- Much slower development for prototypes
- Requires separate frontend/backend coordination
- Overkill for simple prototypes

**Use Case Fit:** Better for production applications

#### Option 3: Gradio
**Pros:**
- Even simpler than Streamlit for ML demos
- Great for model showcasing
- Built-in sharing capabilities

**Cons:**
- Less flexible than Streamlit
- Limited for complex document management UIs

**Use Case Fit:** Good for simple model demos, less suitable for document-heavy apps

### Vector Database

#### Option 1: ChromaDB ⭐ **RECOMMENDED**
**Pros:**
- Simple Python API
- Built-in embedding functions
- Good performance for small-medium datasets
- Easy local development
- Great documentation

**Cons:**
- Limited scalability for very large datasets
- Newer product with evolving API

**Use Case Fit:** Excellent for getting started and medium-scale deployments

#### Option 2: Qdrant
**Pros:**
- High performance Rust implementation
- Excellent scalability
- Rich filtering capabilities
- Good Python client
- Self-hosted or cloud options

**Cons:**
- More complex setup
- Steeper learning curve
- Requires more resources

**Use Case Fit:** Better for production scale and complex filtering needs

#### Option 3: Pinecone
**Pros:**
- Managed service (no ops overhead)
- Excellent performance and scale
- Good developer experience
- Built-in analytics

**Cons:**
- Cloud-only (not local)
- Recurring costs
- Vendor lock-in
- May have latency for local development

**Use Case Fit:** Great for production, but doesn't fit local-first requirement

#### Option 4: Weaviate
**Pros:**
- Open source with managed options
- GraphQL API
- Built-in ML capabilities
- Good scalability

**Cons:**
- More complex architecture
- Heavier resource requirements
- Steeper learning curve

**Use Case Fit:** Overkill for most RAG applications

### Database (Metadata & Sessions)

#### Option 1: PostgreSQL ⭐ **RECOMMENDED**
**Pros:**
- Mature and reliable
- Excellent JSON support (JSONB)
- Rich indexing capabilities
- Great Python ecosystem (asyncpg, SQLAlchemy)
- Vector extensions available (pgvector)
- ACID compliance

**Cons:**
- Can be overkill for simple use cases
- Requires more setup than SQLite

**Use Case Fit:** Perfect for production applications with complex data relationships

#### Option 2: SQLite
**Pros:**
- Zero configuration
- Excellent for development
- Built-in to Python
- Small footprint

**Cons:**
- Limited concurrent write performance
- Not ideal for production web apps
- No built-in vector capabilities

**Use Case Fit:** Great for development and single-user deployments

#### Option 3: MongoDB
**Pros:**
- Flexible schema
- Good JSON/document storage
- Built-in vector search (Atlas)
- Horizontal scaling

**Cons:**
- NoSQL complexity for relational data
- Less mature Python async support
- Overkill for structured metadata

**Use Case Fit:** Good if you need flexible schemas, but PostgreSQL is better for this use case

### Background Job Processing

#### Option 1: Celery + Redis ⭐ **RECOMMENDED**
**Pros:**
- Mature and battle-tested
- Great Python integration
- Rich feature set (scheduling, routing, monitoring)
- Excellent monitoring tools (Flower)
- Supports both queues and scheduled tasks

**Cons:**
- Can be complex to configure
- Redis dependency
- Memory usage for large queues

**Use Case Fit:** Perfect for document processing workflows

#### Option 2: RQ (Redis Queue)
**Pros:**
- Simpler than Celery
- Python-native
- Good for basic job queues
- Easy debugging

**Cons:**
- Less feature-rich than Celery
- No built-in scheduling
- Limited monitoring tools

**Use Case Fit:** Good for simpler job processing needs

#### Option 3: Dramatiq
**Pros:**
- Modern design
- Type-safe
- Good performance
- Multiple broker support

**Cons:**
- Smaller community
- Less mature than Celery
- Fewer monitoring tools

**Use Case Fit:** Good alternative to Celery, but less mature ecosystem

### Caching & Session Store

#### Option 1: Redis ⭐ **RECOMMENDED**
**Pros:**
- Excellent performance
- Rich data structures
- Great Python support
- Can serve multiple roles (cache, sessions, job queue)
- Persistent storage options

**Cons:**
- Memory-based (cost consideration)
- Single point of failure without clustering

**Use Case Fit:** Perfect for caching, sessions, and job queues

#### Option 2: Memcached
**Pros:**
- Simple and fast
- Lower memory usage
- Mature and stable

**Cons:**
- Limited data structures
- No persistence
- Cannot be used for job queues

**Use Case Fit:** Good for pure caching, but Redis is more versatile

### Document Processing

#### Option 1: Python Ecosystem ⭐ **RECOMMENDED**
**Libraries:**
- **PDFs**: PyPDF2, pdfplumber, pymupdf
- **Office**: python-docx, python-pptx, openpyxl
- **Text**: beautifulsoup4, markdown, chardet
- **Images**: pytesseract (OCR), Pillow

**Pros:**
- Rich ecosystem of libraries
- Good integration with AI libraries
- Extensive format support

**Cons:**
- Some libraries have limitations
- Performance varies by library

**Use Case Fit:** Excellent for comprehensive document processing

#### Option 2: Apache Tika (Java)
**Pros:**
- Supports 1000+ file formats
- Very robust parsing
- Mature and battle-tested

**Cons:**
- Java dependency
- More complex integration
- Higher resource usage

**Use Case Fit:** Overkill unless you need exotic format support

### Text Embedding Models

#### Option 1: sentence-transformers ⭐ **RECOMMENDED**
**Models:**
- `all-MiniLM-L6-v2` (fast, good quality)
- `all-mpnet-base-v2` (slower, better quality)
- `multi-qa-mpnet-base-dot-v1` (optimized for Q&A)

**Pros:**
- Easy to use Python library
- Many pre-trained models
- Good performance/quality trade-offs
- Local execution

**Cons:**
- Limited to available models
- Some models are large

**Use Case Fit:** Perfect for RAG applications

#### Option 2: OpenAI Embeddings
**Pros:**
- High quality embeddings
- Simple API
- Regular updates

**Cons:**
- Requires API calls (cost and latency)
- Not local
- Vendor dependency

**Use Case Fit:** Good quality but doesn't fit local requirement

#### Option 3: Ollama Embeddings
**Pros:**
- Fully local
- Consistent with LLM hosting
- Easy integration

**Cons:**
- Limited model selection
- Newer, less proven

**Use Case Fit:** Good for consistency with local LLM setup

### Development Tools

#### Option 1: Docker + Docker Compose ⭐ **RECOMMENDED**
**Pros:**
- Consistent development environment
- Easy service orchestration
- Production parity
- Simple dependency management

**Cons:**
- Learning curve for Docker newcomers
- Resource overhead

#### Option 2: Virtual Environments (venv/conda)
**Pros:**
- Lightweight
- Native Python tooling
- No Docker dependency

**Cons:**
- Doesn't handle non-Python services
- Environment inconsistencies
- Complex service management

#### Option 3: Dev Containers (VS Code)
**Pros:**
- Integrated development experience
- Consistent environments
- Good for team development

**Cons:**
- VS Code specific
- Still requires Docker knowledge

## Recommended Tech Stack

### Primary Recommendation
```
Backend:        FastAPI (Python 3.11+)
Frontend:       Next.js 14 + TypeScript + Tailwind CSS + shadcn/ui
Vector DB:      ChromaDB
Database:       PostgreSQL 15+
Cache/Jobs:     Redis + Celery
Embeddings:     sentence-transformers (all-MiniLM-L6-v2)
LLM:           Ollama (llama2, mistral, or similar)
Development:    Docker Compose
```

### Why This Stack?

1. **Python-First Ecosystem**: Leverages the rich AI/ML ecosystem
2. **Modern Web Standards**: Next.js provides excellent developer experience
3. **Local-First**: All components can run locally
4. **Scalable**: Can grow from development to production
5. **Battle-Tested**: Each component is proven in production
6. **Great Developer Experience**: Good tooling and documentation
7. **Cost-Effective**: Open source with optional managed services

### Alternative Stack (Lighter)
For simpler requirements or resource constraints:
```
Backend:        FastAPI (Python)
Frontend:       React + Vite + TypeScript
Vector DB:      ChromaDB
Database:       SQLite (dev) → PostgreSQL (prod)
Cache:          Redis (minimal setup)
Jobs:           RQ (simpler than Celery)
Embeddings:     sentence-transformers
LLM:           Ollama
Development:    Virtual environments + simple scripts
```

### Enterprise Stack (Maximum Scale)
For large-scale production deployments:
```
Backend:        FastAPI + Kubernetes
Frontend:       Next.js + CDN
Vector DB:      Qdrant cluster
Database:       PostgreSQL cluster
Cache:          Redis cluster
Jobs:           Celery + Redis cluster
Embeddings:     Custom fine-tuned models
LLM:           Multiple Ollama instances + load balancer
Monitoring:     Prometheus + Grafana + ELK stack
```

## Next Steps

1. **Start Simple**: Begin with the primary recommendation
2. **Prototype Core Features**: Document ingestion + basic chat
3. **Validate Architecture**: Test with real documents and queries
4. **Scale Incrementally**: Add complexity as needed
5. **Monitor Performance**: Use proper observability from the start

Would you like me to dive deeper into any specific technology choice or discuss the rationale behind any recommendation? 

## Python RAG Framework

#### Option 1: LlamaIndex ⭐ **RECOMMENDED FOR COMPREHENSIVE RAG**
**Pros:**
- Complete RAG framework with batteries included
- Excellent document loaders for many formats
- Built-in chunking strategies
- Vector store abstractions (supports ChromaDB, Qdrant, etc.)
- Query engines with citation support
- Async support for better performance
- Great Ollama integration
- Comprehensive documentation and examples

**Cons:**
- Can be overkill for simple use cases
- Abstractions might hide important details
- Larger dependency footprint

**Use Case Fit:** Perfect for full-featured RAG applications

**Key LlamaIndex Components:**
```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
```

#### Option 2: LangChain
**Pros:**
- Very mature ecosystem
- Extensive integrations
- Good community support
- Flexible architecture

**Cons:**
- Can be overly complex
- API changes frequently
- Performance can be inconsistent
- Less optimized for RAG specifically

**Use Case Fit:** Good for complex workflows, but LlamaIndex is better for RAG

#### Option 3: Custom with ollama-python + ChromaDB ⭐ **RECOMMENDED FOR LEARNING/CONTROL**
**Pros:**
- Full control over the pipeline
- Minimal dependencies
- Easy to understand and debug
- Direct integration with services
- Learn RAG concepts deeply

**Cons:**
- More code to write
- Need to implement chunking, citations, etc.
- No built-in optimizations

**Use Case Fit:** Great for learning and when you need full control

**Key Libraries:**
```python
import ollama  # ollama-python client
import chromadb  # vector database
import streamlit as st  # frontend
```

## Recommended Prototype Stack

### Minimal Viable Prototype
```
Frontend:       Streamlit
Backend Logic:  Pure Python functions
RAG Framework:  ollama-python + ChromaDB (direct)
Vector DB:      ChromaDB (embedded)
Database:       SQLite
Embeddings:     Ollama (nomic-embed-text)
LLM:           Ollama (llama2/mistral)
Development:    Python virtual environment
```

### Enhanced Prototype
```
Frontend:       Streamlit + streamlit-chat
Backend Logic:  LlamaIndex framework
Vector DB:      ChromaDB (server mode)
Database:       PostgreSQL
Embeddings:     Ollama (nomic-embed-text)
LLM:           Ollama (llama2/mistral)
Background:     Simple async processing
Development:    Python virtual environment
```

## Implementation Approaches

### Approach 1: LlamaIndex (Recommended for Features)
```python
# Quick setup with LlamaIndex
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# Configure Ollama
Settings.llm = Ollama(model="llama2", request_timeout=60.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# Load and index documents
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)

# Query with citations
query_engine = index.as_query_engine(response_mode="compact")
response = query_engine.query("Your question here")
```

### Approach 2: Direct Integration (Recommended for Learning)
```python
import ollama
import chromadb
import streamlit as st

# Initialize clients
ollama_client = ollama.Client()
chroma_client = chromadb.Client()

# Create embeddings
def create_embedding(text):
    response = ollama_client.embeddings(
        model="nomic-embed-text",
        prompt=text
    )
    return response['embedding']

# Query LLM
def query_llm(prompt, context):
    response = ollama_client.chat(
        model="llama2",
        messages=[{
            'role': 'user',
            'content': f"Context: {context}\n\nQuestion: {prompt}"
        }]
    )
    return response['message']['content']
```

## Essential Python Dependencies for Prototype

```python
# requirements.txt
streamlit==1.28.0
ollama==0.1.0
chromadb==0.4.15

# Document processing
PyPDF2==3.0.1
python-docx==1.1.0
beautifulsoup4==4.12.2

# Data handling
pandas==2.1.3
numpy==1.24.3

# Optional: For enhanced RAG
llama-index==0.9.0
llama-index-embeddings-ollama==0.1.0
llama-index-llms-ollama==0.1.0

# Optional: For persistence
sqlalchemy==2.0.23
```

## Development Approach

### Phase 1: Basic Prototype (1-2 days)
1. Streamlit app with file upload
2. Simple text extraction from PDFs
3. Direct Ollama integration for embeddings and LLM
4. Basic ChromaDB storage
5. Simple chat interface

### Phase 2: Enhanced Features (1 week)
1. Multiple document format support
2. Better chunking strategies
3. Citation tracking and display
4. Conversation history
5. Document browser in Streamlit

### Phase 3: Production Considerations
1. Migrate to Next.js frontend
2. Add proper database with migrations
3. Implement background job processing
4. Add user authentication
5. Containerize with Docker

## Quick Start Guide

```bash
# 1. Install Ollama and pull models
ollama pull llama2
ollama pull nomic-embed-text

# 2. Create virtual environment
python -m venv ragnarok-env
source ragnarok-env/bin/activate  # On Windows: ragnarok-env\Scripts\activate

# 3. Install dependencies
pip install streamlit ollama chromadb PyPDF2 python-docx

# 4. Create simple app
# See implementation examples above

# 5. Run prototype
streamlit run app.py
```

This approach gives you a working prototype in hours rather than days, while still providing a solid foundation for future enhancement! 