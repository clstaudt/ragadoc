# RAG System Implementation

This document describes the Retrieval-Augmented Generation (RAG) system implemented in Ragnarok to solve the large document context window issue.

## Problem Solved

**Issue**: Large documents exceed the model's context window, causing the AI to "forget" document content and provide responses without using the document information.

**Solution**: RAG system that chunks documents, stores them in a vector database, and retrieves only relevant chunks for each query.

## Architecture

### Components

1. **Document Chunking**: Uses LlamaIndex's `SentenceSplitter` to break documents into overlapping chunks
2. **Vector Embeddings**: Uses Ollama's embedding models (default: `nomic-embed-text`)
3. **Vector Storage**: ChromaDB for persistent vector storage
4. **Semantic Retrieval**: Retrieves most relevant chunks based on query similarity
5. **Response Generation**: Uses retrieved chunks as context for the LLM

### Flow

```
Document Upload → Chunking → Embeddings → Vector DB → Query → Retrieval → Response
```

## Features

### Automatic Chunking
- Configurable chunk size (default: 512 tokens)
- Configurable overlap (default: 50 tokens)
- Preserves context across chunk boundaries

### Semantic Search
- Uses vector similarity for chunk retrieval
- Configurable similarity threshold (default: 0.7)
- Configurable number of retrieved chunks (default: 5)

### Fallback Support
- Graceful fallback to traditional full-document processing
- Error handling and recovery
- User notification of processing method

### Configuration Options
- Chunk size and overlap
- Similarity threshold
- Number of retrieved chunks
- Embedding model selection
- Enable/disable RAG processing

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Required Ollama Models

```bash
# Embedding model (required)
ollama pull nomic-embed-text

# Alternative embedding models
ollama pull mxbai-embed-large
ollama pull all-minilm

# LLM models (if not already installed)
ollama pull llama3.1:8b
ollama pull mistral:latest
```

### 3. Start Ollama

```bash
ollama serve
```

## Usage

### Basic Usage

1. **Enable RAG**: Check "Enable RAG (Semantic Search)" in the sidebar
2. **Upload Document**: Upload a PDF document as usual
3. **Wait for Processing**: The system will automatically chunk and process the document
4. **Ask Questions**: Questions will use semantic search to find relevant chunks

### Configuration

Access RAG settings in the sidebar under "🔍 RAG Settings":

- **Chunk Size**: Size of text chunks (256-1024 tokens)
- **Chunk Overlap**: Overlap between chunks (0-200 tokens)
- **Similarity Threshold**: Minimum similarity for retrieval (0.0-1.0)
- **Max Retrieved Chunks**: Number of chunks to retrieve (1-10)
- **Embedding Model**: Model for generating embeddings

### Visual Feedback

The system provides clear feedback about processing method:

- ✅ **RAG Processing**: "Response generated using RAG (semantic search)"
- 📄 **Traditional Processing**: "Response generated using full document"
- ⚠️ **Fallback**: "RAG system failed, using traditional processing"

### Retrieved Chunks Display

When using RAG, you can view retrieved chunks:
- Expandable section showing relevant chunks
- Similarity scores for each chunk
- Chunk content preview

## Testing

### Quick Test

```bash
python experiments/test_rag.py
```

This will:
1. Check dependencies
2. Verify Ollama is running
3. Test document processing
4. Test query retrieval
5. Verify responses

### Manual Testing

1. Upload a large document (>10,000 words)
2. Enable RAG in settings
3. Ask specific questions about different parts of the document
4. Verify responses use relevant information
5. Check retrieved chunks for relevance

## Performance Benefits

### Memory Efficiency
- Only relevant chunks loaded into context
- Supports documents of any size
- Consistent memory usage regardless of document size

### Response Quality
- More focused responses using relevant content
- Better handling of multi-topic documents
- Reduced hallucination from irrelevant context

### Scalability
- Persistent vector storage
- Fast similarity search
- Supports multiple documents (future enhancement)

## Configuration Examples

### For Large Documents (>50 pages)
```python
{
    "chunk_size": 1024,
    "chunk_overlap": 100,
    "similarity_threshold": 0.6,
    "top_k": 7
}
```

### For Precise Retrieval
```python
{
    "chunk_size": 256,
    "chunk_overlap": 25,
    "similarity_threshold": 0.8,
    "top_k": 3
}
```

### For Comprehensive Coverage
```python
{
    "chunk_size": 512,
    "chunk_overlap": 75,
    "similarity_threshold": 0.5,
    "top_k": 10
}
```

## Troubleshooting

### Common Issues

**RAG System Not Available**
- Check Ollama is running: `ollama serve`
- Verify embedding model: `ollama pull nomic-embed-text`
- Check dependencies: `pip install -r requirements.txt`

**Poor Retrieval Quality**
- Lower similarity threshold (0.5-0.6)
- Increase number of retrieved chunks
- Try different embedding model
- Adjust chunk size for your document type

**Slow Processing**
- Reduce chunk overlap
- Use smaller embedding model
- Increase chunk size (fewer chunks)

**Memory Issues**
- Reduce number of retrieved chunks
- Use smaller chunk size
- Clear old documents from vector DB

### Debug Mode

Enable debug logging to see detailed RAG operations:

```python
from loguru import logger
logger.add("rag_debug.log", level="DEBUG")
```

## Future Enhancements

### Planned Features
- Multi-document support
- Hybrid search (keyword + semantic)
- Document summarization
- Chunk re-ranking
- Custom embedding fine-tuning

### Advanced Configuration
- Custom chunking strategies
- Multiple vector stores
- Query expansion
- Response fusion

## API Reference

### RAGSystem Class

```python
from ragnarok import RAGSystem, create_rag_system

# Create system
rag = create_rag_system(
    ollama_base_url="http://localhost:11434",
    embedding_model="nomic-embed-text",
    chunk_size=512,
    chunk_overlap=50,
    similarity_threshold=0.7,
    top_k=5
)

# Process document
stats = rag.process_document(text, document_id)

# Query document
result = rag.query_document(question)

# Get retrieval info
chunks = rag.get_retrieval_info(question)

# Cleanup
rag.cleanup()
```

### Key Methods

- `process_document(text, doc_id)`: Process and store document
- `query_document(question)`: Get AI response with retrieval
- `get_retrieval_info(question)`: Get chunk information only
- `clear_document(doc_id)`: Remove document from storage
- `get_system_info()`: Get configuration details
- `cleanup()`: Clean up resources

## Contributing

When contributing to the RAG system:

1. **Test thoroughly** with various document types and sizes
2. **Maintain backward compatibility** with traditional processing
3. **Add appropriate error handling** and user feedback
4. **Update documentation** for new features
5. **Consider performance impact** of changes

## License

Same as the main Ragnarok project. 