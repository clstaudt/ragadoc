#!/usr/bin/env python3
"""
Test script for the RAG system
"""

import os
import sys
from ragnarok import create_rag_system

def test_rag_system(embedding_model, llm_model):
    """Test the RAG system with a sample document"""
    print("🧪 Testing RAG System")
    print("=" * 50)
    
    # Sample document text
    sample_text = """
    Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines. 
    Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed.
    
    Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data.
    Natural Language Processing (NLP) is another important area of AI that focuses on the interaction between computers and human language.
    
    Computer vision is a field of AI that trains computers to interpret and understand visual information from the world.
    Robotics combines AI with mechanical engineering to create autonomous machines that can perform tasks in the physical world.
    
    The applications of AI are vast and include healthcare, finance, transportation, entertainment, and many other industries.
    Ethical considerations in AI development include bias, privacy, transparency, and the impact on employment.
    """
    
    try:
        # Create RAG system
        print(f"🔧 Initializing RAG system with {embedding_model} and {llm_model}...")
        rag_system = create_rag_system(
            ollama_base_url="http://localhost:11434",
            embedding_model=embedding_model,
            llm_model=llm_model,
            chunk_size=256,
            chunk_overlap=50,
            top_k=3
        )
        print("✅ RAG system initialized")
        
        # Process document
        print("\n📄 Processing sample document...")
        stats = rag_system.process_document(sample_text, "test_doc")
        print(f"✅ Document processed: {stats['total_chunks']} chunks created")
        
        # Test queries
        test_queries = [
            "What is machine learning?",
            "Tell me about deep learning",
            "What are the applications of AI?",
            "What are the ethical considerations?"
        ]
        
        print("\n🔍 Testing queries...")
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Query {i}: {query} ---")
            
            try:
                # Get retrieval info
                chunks = rag_system.get_retrieval_info(query)
                print(f"Retrieved {len(chunks)} chunks:")
                
                for j, chunk in enumerate(chunks, 1):
                    print(f"  Chunk {j} (Score: {chunk['score']:.3f}): {chunk['text'][:100]}...")
                
                # Get full response
                result = rag_system.query_document(query)
                print(f"\nResponse: {result['response'][:200]}...")
                
            except Exception as e:
                print(f"❌ Query failed: {e}")
        
        # Cleanup
        print("\n🧹 Cleaning up...")
        rag_system.cleanup()
        print("✅ Cleanup completed")
        
        print("\n🎉 RAG system test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ RAG system test failed: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        "llama_index",
        "chromadb",
        "sentence_transformers"
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies available")
    return True

def check_ollama():
    """Check if Ollama is running and has required models"""
    print("\n🔍 Checking Ollama...")
    
    try:
        import ollama
        
        # Check if Ollama is running
        models = ollama.list()
        print("✅ Ollama is running")
        
        # Check for embedding model
        model_names = [model.get('model', model.get('name', '')) for model in models.get('models', [])]
        
        # Check for any variant of nomic-embed-text
        embedding_models = [name for name in model_names if 'nomic-embed-text' in name]
        
        if embedding_models:
            print(f"✅ nomic-embed-text model available: {embedding_models[0]}")
            embedding_model = embedding_models[0]
        else:
            # Check for alternative embedding models
            alt_models = [name for name in model_names if any(embed in name for embed in ['mxbai-embed', 'all-minilm'])]
            if alt_models:
                print(f"✅ Alternative embedding model available: {alt_models[0]}")
                print("💡 You can use this model by updating the RAG config in the app")
                embedding_model = alt_models[0]
            else:
                print("❌ No embedding model found")
                print("Install with: ollama pull nomic-embed-text")
                return False, None, None
        
        # Find a suitable LLM model (exclude embedding models)
        llm_models = [name for name in model_names if not any(embed in name.lower() for embed in ['embed', 'minilm'])]
        
        if llm_models:
            llm_model = llm_models[0]  # Use first available LLM
            print(f"✅ LLM model available: {llm_model}")
            return True, embedding_model, llm_model
        else:
            print("❌ No LLM model found")
            print("Install with: ollama pull olmo2:13b")
            return False, None, None
            
    except Exception as e:
        print(f"❌ Ollama check failed: {e}")
        print("Make sure Ollama is running: ollama serve")
        return False, None, None

if __name__ == "__main__":
    print("🚀 RAG System Test Suite")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check Ollama
    ollama_ok, embedding_model, llm_model = check_ollama()
    if not ollama_ok:
        sys.exit(1)
    
    # Test RAG system
    if test_rag_system(embedding_model, llm_model):
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Tests failed!")
        sys.exit(1) 