"""
RAG (Retrieval-Augmented Generation) System for Document Q&A

This module implements a complete RAG pipeline using llama_index:
- Document chunking with overlap
- Vector embeddings using Ollama
- ChromaDB for vector storage
- Semantic retrieval for relevant chunks
- Context-aware response generation
"""

import os
import tempfile
import shutil
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import streamlit as st
from loguru import logger

# LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex, 
    Document, 
    Settings,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

# ChromaDB imports
import chromadb
from chromadb.config import Settings as ChromaSettings


class RAGSystem:
    """
    Complete RAG system for document processing and querying
    """
    
    def __init__(
        self, 
        ollama_base_url: str = "http://localhost:11434",
        embedding_model: str = "nomic-embed-text",
        llm_model: str = "llama3.1:8b",
        chunk_size: int = 128,
        chunk_overlap: int = 25,
        similarity_threshold: float = 0.7,
        top_k: int = 10
    ):
        """
        Initialize the RAG system
        
        Args:
            ollama_base_url: Ollama server URL
            embedding_model: Model for generating embeddings
            llm_model: Model for text generation
            chunk_size: Size of text chunks in tokens
            chunk_overlap: Overlap between chunks in tokens
            similarity_threshold: Minimum similarity for retrieved chunks
            top_k: Number of chunks to retrieve
        """
        self.ollama_base_url = ollama_base_url
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        
        # Storage paths
        self.storage_dir = Path(tempfile.gettempdir()) / "ragnarok_rag"
        self.chroma_dir = self.storage_dir / "chroma_db"
        
        # Initialize components
        self._setup_llama_index()
        self._setup_chroma_client()
        
        # Index and query engine (will be set when document is processed)
        self.index: Optional[VectorStoreIndex] = None
        self.query_engine: Optional[RetrieverQueryEngine] = None
        self.current_document_id: Optional[str] = None
        
    def _setup_llama_index(self):
        """Configure LlamaIndex settings"""
        try:
            # Setup embedding model
            self.embed_model = OllamaEmbedding(
                model_name=self.embedding_model,
                base_url=self.ollama_base_url,
                ollama_additional_kwargs={"mirostat": 0}
            )
            
            # Test embedding model immediately
            logger.info(f"Testing embedding model: {self.embedding_model}")
            test_embedding = self.embed_model.get_text_embedding("test")
            logger.info(f"Embedding model working: dimension={len(test_embedding)}, sample={test_embedding[:3]}")
            
            # Setup LLM
            self.llm = Ollama(
                model=self.llm_model,
                base_url=self.ollama_base_url,
                request_timeout=120.0
            )
            
            # Configure global settings
            Settings.embed_model = self.embed_model
            Settings.llm = self.llm
            Settings.chunk_size = self.chunk_size
            Settings.chunk_overlap = self.chunk_overlap
            
            logger.info(f"LlamaIndex configured with embedding model: {self.embedding_model}")
            logger.info(f"Global Settings.embed_model: {Settings.embed_model}")
            
        except Exception as e:
            logger.error(f"Failed to setup LlamaIndex: {e}")
            raise
    
    def _setup_chroma_client(self):
        """Setup ChromaDB client"""
        try:
            # Ensure storage directory exists
            self.chroma_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.chroma_dir),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            logger.info(f"ChromaDB initialized at: {self.chroma_dir}")
            
        except Exception as e:
            logger.error(f"Failed to setup ChromaDB: {e}")
            raise
    
    def process_document(self, document_text: str, document_id: str) -> Dict[str, Any]:
        """
        Process a document: chunk it, create embeddings, and store in vector DB
        
        Args:
            document_text: The full text of the document
            document_id: Unique identifier for the document
            
        Returns:
            Dictionary with processing statistics
        """
        try:
            logger.info(f"Processing document: {document_id}")
            logger.info(f"Document text length: {len(document_text)} characters")
            
            # Test embedding generation before processing
            logger.info("Testing embedding generation...")
            test_embedding = self.embed_model.get_text_embedding("test text")
            logger.info(f"Test embedding dimension: {len(test_embedding)}")
            logger.info(f"Test embedding sample: {test_embedding[:3]}")
            
            # Create document object
            document = Document(
                text=document_text,
                metadata={
                    "document_id": document_id,
                    "source": "pdf_upload"
                }
            )
            logger.info(f"Created document object with {len(document_text)} characters")
            
            # Setup text splitter
            text_splitter = SentenceSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            
            # Test chunking
            logger.info("Testing text chunking...")
            test_nodes = text_splitter.get_nodes_from_documents([document])
            logger.info(f"Created {len(test_nodes)} chunks")
            if test_nodes:
                logger.info(f"First chunk length: {len(test_nodes[0].text)} characters")
                logger.info(f"First chunk preview: {test_nodes[0].text[:100]}...")
            
            # Create collection for this document
            collection_name = f"doc_{document_id}"
            
            collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={
                    "document_id": document_id,
                    "hnsw:space": "cosine"  # Explicitly set cosine distance
                }
            )
            logger.info(f"Created new collection: {collection_name}")
            
            # Create vector store
            vector_store = ChromaVectorStore(chroma_collection=collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Create index with explicit embedding model
            logger.info(f"Creating vector index with chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}")
            logger.info(f"Using embedding model: {self.embed_model}")
            
            # Ensure the embedding model is properly set
            Settings.embed_model = self.embed_model
            
            self.index = VectorStoreIndex.from_documents(
                [document],
                storage_context=storage_context,
                transformations=[text_splitter],
                embed_model=self.embed_model,  # Explicitly pass embedding model
                show_progress=True
            )
            logger.info("Vector index created successfully")
            
            # Test the index immediately
            logger.info("Testing index retrieval...")
            from llama_index.core.retrievers import VectorIndexRetriever
            test_retriever = VectorIndexRetriever(index=self.index, similarity_top_k=3)
            test_nodes = test_retriever.retrieve("test query")
            logger.info(f"Test retrieval returned {len(test_nodes)} nodes")
            for i, node in enumerate(test_nodes):
                score = getattr(node, 'score', 'NO_SCORE')
                logger.info(f"Test node {i+1}: score={score}, text_length={len(node.text)}")
            
            # Setup query engine
            self._setup_query_engine()
            
            # Store current document ID
            self.current_document_id = document_id
            
            # Get statistics - count actual nodes (chunks)
            # Try multiple ways to get the node count
            try:
                # Method 1: Try to get from vector store
                if hasattr(self.index.vector_store, '_collection'):
                    chunk_count = self.index.vector_store._collection.count()
                    logger.info(f"Chunk count from vector store: {chunk_count}")
                else:
                    # Method 2: Get from docstore
                    all_nodes = list(self.index.docstore.docs.values())
                    chunk_count = len(all_nodes)
                    logger.info(f"Chunk count from docstore: {chunk_count}")
            except Exception as e:
                logger.warning(f"Could not get accurate chunk count: {e}")
                chunk_count = 1  # Fallback
            
            stats = {
                "document_id": document_id,
                "total_chunks": chunk_count,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "embedding_model": self.embedding_model,
                "collection_name": collection_name
            }
            
            logger.info(f"Document processed successfully: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to process document: {e}")
            raise
    
    def _setup_query_engine(self):
        """Setup the query engine with retriever and post-processor"""
        if not self.index:
            raise ValueError("No index available. Process a document first.")
        
        # Create retriever
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.top_k
        )
        
        # Create post-processor for similarity filtering
        postprocessor = SimilarityPostprocessor(
            similarity_cutoff=self.similarity_threshold
        )
        
        # Create query engine
        self.query_engine = RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=[postprocessor]
        )
        
        logger.info("Query engine configured")
    
    def query_document(self, question: str) -> Dict[str, Any]:
        """
        Query the document using RAG
        
        Args:
            question: User's question
            
        Returns:
            Dictionary containing response and metadata
        """
        if not self.query_engine:
            raise ValueError("No query engine available. Process a document first.")
        
        try:
            logger.info(f"Querying document: {question}")
            logger.info(f"Using similarity_threshold={self.similarity_threshold}, top_k={self.top_k}")
            
            # Get response from query engine
            response = self.query_engine.query(question)
            logger.info(f"Query engine returned response: {len(str(response))} characters")
            
            # Extract source nodes (retrieved chunks)
            source_nodes = response.source_nodes if hasattr(response, 'source_nodes') else []
            
            # Prepare retrieved chunks info
            retrieved_chunks = []
            for i, node in enumerate(source_nodes):
                chunk_info = {
                    "chunk_id": i + 1,
                    "text": node.text,
                    "score": getattr(node, 'score', 0.0),
                    "metadata": node.metadata if hasattr(node, 'metadata') else {}
                }
                retrieved_chunks.append(chunk_info)
            
            result = {
                "response": str(response),
                "question": question,
                "retrieved_chunks": retrieved_chunks,
                "num_chunks_retrieved": len(retrieved_chunks),
                "document_id": self.current_document_id
            }
            
            logger.info(f"Query completed. Retrieved {len(retrieved_chunks)} chunks")
            return result
            
        except Exception as e:
            logger.error(f"Failed to query document: {e}")
            raise
    
    def get_retrieval_info(self, question: str) -> List[Dict[str, Any]]:
        """
        Get detailed information about retrieved chunks without generating response
        
        Args:
            question: User's question
            
        Returns:
            List of retrieved chunk information
        """
        if not self.index:
            raise ValueError("No index available. Process a document first.")
        
        try:
            logger.info(f"=== RETRIEVAL DEBUG START ===")
            logger.info(f"Question: {question}")
            
            # Test query embedding generation
            logger.info("Generating query embedding...")
            query_embedding = self.embed_model.get_text_embedding(question)
            logger.info(f"Query embedding dimension: {len(query_embedding)}")
            logger.info(f"Query embedding sample: {query_embedding[:3]}")
            
            # Create retriever
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=self.top_k
            )
            logger.info(f"Created retriever with top_k={self.top_k}")
            
            # Check vector store status
            if hasattr(self.index.vector_store, '_collection'):
                collection = self.index.vector_store._collection
                stored_count = collection.count()
                logger.info(f"Vector store contains {stored_count} embeddings")
                
                # Test direct ChromaDB query
                logger.info("Testing direct ChromaDB query...")
                try:
                    chroma_results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=min(self.top_k, stored_count)
                    )
                    logger.info(f"Direct ChromaDB query returned {len(chroma_results['ids'][0])} results")
                    if chroma_results['distances'][0]:
                        logger.info(f"Direct ChromaDB distances: {chroma_results['distances'][0]}")
                        # Convert distances to similarities (ChromaDB uses cosine distance)
                        similarities = [1 - dist for dist in chroma_results['distances'][0]]
                        logger.info(f"Converted to similarities: {similarities}")
                except Exception as e:
                    logger.error(f"Direct ChromaDB query failed: {e}")
            
            # Retrieve nodes
            logger.info("Retrieving nodes via LlamaIndex...")
            nodes = retriever.retrieve(question)
            logger.info(f"Retrieved {len(nodes)} nodes before filtering")
            
            # Log all scores for debugging
            for i, node in enumerate(nodes):
                score = getattr(node, 'score', 0.0)
                logger.info(f"Node {i+1} score: {score:.6f} (threshold: {self.similarity_threshold})")
                logger.info(f"Node {i+1} text preview: {node.text[:100]}...")
                
                # Check if node has embedding
                if hasattr(node, 'embedding') and node.embedding:
                    logger.info(f"Node {i+1} has embedding: dimension={len(node.embedding)}")
                else:
                    logger.info(f"Node {i+1} has no embedding stored")
            
            # Filter by similarity threshold
            filtered_nodes = [
                node for node in nodes 
                if getattr(node, 'score', 0.0) >= self.similarity_threshold
            ]
            
            logger.info(f"After filtering: {len(filtered_nodes)} nodes remain")
            logger.info(f"=== RETRIEVAL DEBUG END ===")
            
            # Prepare chunk information
            chunks_info = []
            for i, node in enumerate(filtered_nodes):
                chunk_info = {
                    "chunk_id": i + 1,
                    "text": node.text,
                    "score": getattr(node, 'score', 0.0),
                    "metadata": node.metadata if hasattr(node, 'metadata') else {},
                    "length": len(node.text)
                }
                chunks_info.append(chunk_info)
            
            return chunks_info
            
        except Exception as e:
            logger.error(f"Failed to get retrieval info: {e}")
            raise
    
    def clear_document(self, document_id: str = None):
        """
        Clear the current document from memory and storage
        
        Args:
            document_id: Specific document ID to clear (optional)
        """
        try:
            target_id = document_id or self.current_document_id
            
            if target_id:
                collection_name = f"doc_{target_id}"
                try:
                    self.chroma_client.delete_collection(collection_name)
                    logger.info(f"Cleared document collection: {collection_name}")
                except Exception as e:
                    logger.warning(f"Could not delete collection {collection_name}: {e}")
            
            # Reset instance variables
            self.index = None
            self.query_engine = None
            self.current_document_id = None
            
            logger.info("Document cleared from RAG system")
            
        except Exception as e:
            logger.error(f"Failed to clear document: {e}")
    
    def clear_all_documents(self):
        """
        Clear all documents and collections from the vector store
        """
        try:
            # Delete ALL existing collections
            existing_collections = self.chroma_client.list_collections()
            for existing_collection in existing_collections:
                collection_obj = existing_collection if hasattr(existing_collection, 'name') else existing_collection
                existing_name = collection_obj.name if hasattr(collection_obj, 'name') else str(collection_obj)
                try:
                    self.chroma_client.delete_collection(existing_name)
                    logger.info(f"Deleted collection: {existing_name}")
                except Exception as e:
                    logger.warning(f"Could not delete collection {existing_name}: {e}")
            
            # Reset instance variables
            self.index = None
            self.query_engine = None
            self.current_document_id = None
            
            logger.info("All documents cleared from RAG system")
            
        except Exception as e:
            logger.error(f"Failed to clear all documents: {e}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the RAG system configuration"""
        return {
            "ollama_base_url": self.ollama_base_url,
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "similarity_threshold": self.similarity_threshold,
            "top_k": self.top_k,
            "storage_dir": str(self.storage_dir),
            "current_document_id": self.current_document_id,
            "has_active_document": self.index is not None
        }
    
    def cleanup(self):
        """Cleanup resources and temporary files"""
        try:
            if self.storage_dir.exists():
                shutil.rmtree(self.storage_dir)
                logger.info("RAG system cleanup completed")
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")


def create_rag_system(
    ollama_base_url: str = "http://localhost:11434",
    embedding_model: str = "nomic-embed-text",
    **kwargs
) -> RAGSystem:
    """
    Factory function to create a RAG system instance
    
    Args:
        ollama_base_url: Ollama server URL
        embedding_model: Embedding model name
        **kwargs: Additional configuration options
        
    Returns:
        Configured RAG system instance
    """
    return RAGSystem(
        ollama_base_url=ollama_base_url,
        embedding_model=embedding_model,
        **kwargs
    ) 