"""
Refactored Streamlit App for Document Q&A

This is the frontend UI layer that uses the ragnarok backend modules.
Significantly reduced from the original monolithic app.py file.
"""

import streamlit as st
import ollama
import os
import io
from datetime import datetime
from streamlit_pdf_viewer import pdf_viewer
import pdfplumber

# Import ragnarok backend modules
from ragnarok import (
    EnhancedPDFProcessor, 
    RAGSystem, 
    create_rag_system,
    ModelManager,
    ChatManager,
    LLMInterface,
    PromptBuilder,
    ReasoningParser
)
from loguru import logger

# Configuration
st.set_page_config(
    page_title="Document Q&A", 
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("Document Q&A")

# Environment detection
def is_running_in_docker():
    """Check if we're running inside a Docker container"""
    return (
        os.path.exists('/.dockerenv') or 
        os.environ.get('STREAMLIT_SERVER_ADDRESS') == '0.0.0.0'
    )

# Initialize environment
in_docker = is_running_in_docker()
if in_docker:
    ollama_base_url = os.environ.get('OLLAMA_BASE_URL', 'http://host.docker.internal:11434')
else:
    ollama_base_url = "http://localhost:11434"


def init_session_state():
    """Initialize Streamlit session state with backend managers"""
    
    # Initialize backend managers
    if "model_manager" not in st.session_state:
        st.session_state.model_manager = ModelManager(ollama_base_url, in_docker)
    
    if "chat_manager" not in st.session_state:
        st.session_state.chat_manager = ChatManager()
        # Create first chat
        st.session_state.chat_manager.create_new_chat()
    
    if "llm_interface" not in st.session_state:
        st.session_state.llm_interface = LLMInterface(ollama_base_url, in_docker)
    
    # UI state
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None
    
    if "generating" not in st.session_state:
        st.session_state.generating = False
    
    if "stop_generation" not in st.session_state:
        st.session_state.stop_generation = False
    
    # RAG configuration
    if "rag_config" not in st.session_state:
        st.session_state.rag_config = {
            "chunk_size": 256,
            "chunk_overlap": 25,
            "similarity_threshold": 0.7,
            "top_k": 10,
            "embedding_model": "nomic-embed-text",
            "llm_model": None
        }
    
    # Initialize RAG system
    if "rag_system" not in st.session_state:
        init_rag_system()


def init_rag_system():
    """Initialize the RAG system"""
    try:
        # Get available models for embedding model check
        available_models = st.session_state.model_manager.get_available_models()
        embedding_model = st.session_state.rag_config["embedding_model"]
        
        # Check if we need to add :latest suffix
        if embedding_model not in available_models:
            embedding_model_with_suffix = f"{embedding_model}:latest"
            if embedding_model_with_suffix in available_models:
                embedding_model = embedding_model_with_suffix
                logger.info(f"Using embedding model: {embedding_model}")
        
        # Create RAG system with current configuration
        rag_config = st.session_state.rag_config.copy()
        rag_config["embedding_model"] = embedding_model
        
        # Use the selected model from the UI
        if st.session_state.selected_model:
            rag_config["llm_model"] = st.session_state.selected_model
        else:
            # Try to get the first available model
            if available_models:
                llm_models = [m for m in available_models 
                             if not any(embed in m.lower() for embed in ['embed', 'minilm'])]
                if llm_models:
                    rag_config["llm_model"] = llm_models[0]
        
        st.session_state.rag_system = create_rag_system(
            ollama_base_url=ollama_base_url,
            **rag_config
        )
        logger.info("RAG system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        st.session_state.rag_system = None


def show_citations(response, chat, user_question=""):
    """Show citation-based references"""
    if not chat or not chat.document_content:
        return
        
    try:
        pdf_processor = EnhancedPDFProcessor(chat.document_content)
        pdf_processor.display_citation_based_references(
            response, chat.document_text, user_question
        )
    except Exception as e:
        st.warning(f"Could not show citations: {e}")


def render_sidebar():
    """Render the sidebar with chat history and RAG configuration"""
    with st.sidebar:
        st.header("Settings")
        
        # Global Model Selection
        try:
            available_models = st.session_state.model_manager.get_available_models()
            if available_models:
                previous_model = st.session_state.selected_model
                st.session_state.selected_model = st.selectbox(
                    "🤖 Chat Model (Global):",
                    available_models,
                    index=0 if not st.session_state.selected_model else 
                          (available_models.index(st.session_state.selected_model) 
                           if st.session_state.selected_model in available_models else 0),
                    key="global_model_selector",
                    help="This model will be used for all chats"
                )
                
                # Reinitialize RAG system if model changed
                if previous_model != st.session_state.selected_model and previous_model is not None:
                    logger.info(f"Model changed from {previous_model} to {st.session_state.selected_model}")
                    if "rag_system" in st.session_state:
                        del st.session_state["rag_system"]
                        init_rag_system()
                    st.rerun()
            else:
                st.error("❌ No Ollama models found")
                st.caption("Please ensure Ollama is running")
                return
        except Exception as e:
            st.error(f"❌ Error connecting to Ollama: {e}")
            return
        
        # Global Embedding Model Selection
        previous_embedding_model = st.session_state.rag_config["embedding_model"]
        embedding_model = st.selectbox(
            "🔍 Embedding Model (Global):", 
            ["nomic-embed-text", "mxbai-embed-large", "all-minilm"], 
            index=0 if st.session_state.rag_config["embedding_model"] == "nomic-embed-text" else 
                  (1 if st.session_state.rag_config["embedding_model"] == "mxbai-embed-large" else 2),
            key="global_embedding_selector",
            help="This model will be used for document embedding and semantic search"
        )
        
        # Update embedding model if changed
        if embedding_model != previous_embedding_model:
            st.session_state.rag_config["embedding_model"] = embedding_model
            logger.info(f"Embedding model changed from {previous_embedding_model} to {embedding_model}")
            if "rag_system" in st.session_state:
                del st.session_state["rag_system"]
                init_rag_system()
            st.info("⚠️ Embedding model changed. Upload a new document to apply changes.")
            st.rerun()
        
        st.divider()
        
        st.header("Chat History")
        
        # Connection info
        if in_docker:
            st.caption(f"🐳 Docker → {ollama_base_url}")
        else:
            st.caption(f"💻 Direct → localhost:11434")
        
        # New chat button
        if st.button("New Chat", use_container_width=True, type="primary"):
            # Stop any ongoing generation before creating new chat
            if st.session_state.get('generating', False):
                st.session_state.stop_generation = True
                st.session_state.generating = False
                logger.info("Stopped ongoing generation due to new chat creation")
            
            st.session_state.chat_manager.create_new_chat()
            st.rerun()
        
        st.divider()
        
        # RAG Configuration
        with st.expander("🔍 RAG Settings", expanded=False):
            st.info("🔍 **Smart Retrieval**: The system first finds ALL chunks above the similarity threshold, then limits to the max number.")
            
            # RAG parameters (excluding embedding model which is now global)
            chunk_size = st.slider("Chunk Size (tokens)", 256, 1024, st.session_state.rag_config["chunk_size"], 64)
            chunk_overlap = st.slider("Chunk Overlap (tokens)", 0, 200, st.session_state.rag_config["chunk_overlap"], 10)
            similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, st.session_state.rag_config["similarity_threshold"], 0.05)
            top_k = st.slider("Max Retrieved Chunks", 1, 20, st.session_state.rag_config["top_k"], 1)
            
            # Update configuration if changed (excluding embedding model)
            new_config = {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap, 
                "similarity_threshold": similarity_threshold,
                "top_k": top_k,
                "embedding_model": st.session_state.rag_config["embedding_model"]  # Keep current embedding model
            }
            
            if new_config != st.session_state.rag_config:
                st.session_state.rag_config = new_config
                st.info("⚠️ RAG settings changed. Upload a new document to apply changes.")
            
            # RAG system status
            if st.session_state.rag_system:
                st.success("✅ RAG System Ready")
                
                # Show available documents
                available_docs = st.session_state.rag_system.get_available_documents()
                if available_docs:
                    st.info(f"📊 {len(available_docs)} document(s) in system")
                    
                    # Show current document for current chat
                    current_chat = st.session_state.chat_manager.get_current_chat()
                    if current_chat and current_chat.rag_processed:
                        stats = current_chat.rag_stats or {}
                        st.info(f"📄 Current: {stats.get('total_chunks', 0)} chunks")
                    else:
                        st.warning("📄 No document in current chat")
                else:
                    st.warning("📄 No documents processed yet")
            else:
                st.error("❌ RAG System Not Available")
                if st.button("🔄 Retry RAG Initialization"):
                    if "rag_system" in st.session_state:
                        del st.session_state["rag_system"]
                    init_rag_system()
                    st.rerun()
        
        st.divider()
        
        # Chat history
        sorted_chats = st.session_state.chat_manager.get_sorted_chats()
        if sorted_chats:
            for chat_id, chat_session in sorted_chats:
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    is_current = chat_id == st.session_state.chat_manager.current_chat_id
                    button_type = "primary" if is_current else "secondary"
                    
                    if st.button(chat_session.title, key=f"chat-{chat_id}", 
                               use_container_width=True, type=button_type):
                        # Stop any ongoing generation before switching
                        if st.session_state.get('generating', False):
                            st.session_state.stop_generation = True
                            st.session_state.generating = False
                            logger.info("Stopped ongoing generation due to chat switch")
                        
                        # Switch to the chat
                        st.session_state.chat_manager.switch_to_chat(chat_id)
                        
                        # Load the appropriate document for this chat if it has one
                        if chat_session.document_id and st.session_state.rag_system:
                            try:
                                success = st.session_state.rag_system.load_document(chat_session.document_id)
                                if success:
                                    logger.info(f"Loaded document {chat_session.document_id} for chat {chat_id}")
                                else:
                                    logger.warning(f"Could not load document {chat_session.document_id} for chat {chat_id}")
                            except Exception as e:
                                logger.error(f"Error loading document for chat {chat_id}: {e}")
                        
                        st.rerun()
                
                with col2:
                    if st.button("×", key=f"del-{chat_id}", help="Delete"):
                        # Stop any ongoing generation before deleting chat
                        if st.session_state.get('generating', False):
                            st.session_state.stop_generation = True
                            st.session_state.generating = False
                            logger.info("Stopped ongoing generation due to chat deletion")
                        
                        st.session_state.chat_manager.delete_chat(chat_id)
                        st.rerun()


def render_document_upload():
    """Render document upload interface"""
    st.header("Upload Document")
    st.info("Upload a PDF document to start chatting")
    
    # Use unique key per chat to avoid file state conflicts
    current_chat = st.session_state.chat_manager.get_current_chat()
    uploader_key = f"uploader_{current_chat.id if current_chat else 'default'}"
    
    uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'], key=uploader_key)
    
    if uploaded_file is not None:
        try:
            with st.spinner("Processing PDF..."):
                # Extract text using EnhancedPDFProcessor
                pdf_bytes = uploaded_file.getvalue()
                processor = EnhancedPDFProcessor(pdf_bytes)
                extracted_text = processor.extract_full_text()
            
            if extracted_text and extracted_text.strip():
                # IMPORTANT: Make sure we update the CURRENT chat, not create a new one
                # Get the current chat again to ensure we have the right reference
                current_chat = st.session_state.chat_manager.get_current_chat()
                if not current_chat:
                    # If no current chat exists, create one
                    chat_id = st.session_state.chat_manager.create_new_chat()
                    current_chat = st.session_state.chat_manager.get_current_chat()
                
                # Update chat with document using backend
                success = st.session_state.chat_manager.update_document(
                    uploaded_file.name,
                    pdf_bytes,
                    extracted_text,
                    current_chat.id  # Explicitly specify the chat ID
                )
                
                if success:
                    st.success(f"Document '{uploaded_file.name}' processed successfully!")
                    st.info(f"Extracted {len(extracted_text.split()):,} words")
                    
                    # Process with RAG system - now required
                    if st.session_state.rag_system and getattr(st.session_state.rag_system, 'index', None) is not None:
                        # RAG system is ready, process document
                        try:
                            with st.spinner("Processing document with RAG system..."):
                                import uuid
                                document_id = str(uuid.uuid4()).replace('-', '')[:16]
                                
                                rag_stats = st.session_state.rag_system.process_document(
                                    extracted_text, document_id
                                )
                                
                                # Update chat with RAG processing info
                                st.session_state.chat_manager.update_rag_processing(
                                    rag_stats, current_chat.id  # Explicitly specify the chat ID
                                )
                                
                                st.success("✅ Document processed with RAG system!")
                                st.info(f"Created {rag_stats['total_chunks']} chunks for semantic search")
                                
                        except Exception as e:
                            logger.error(f"RAG processing failed: {e}")
                            st.error(f"❌ **RAG processing failed**: {str(e)}")
                            st.error("⚠️ Please try again or check your RAG system configuration.")
                            return
                    elif st.session_state.rag_system:
                        # RAG system exists but no index - reinitialize and try again
                        try:
                            with st.spinner("Reinitializing RAG system and processing document..."):
                                # Reinitialize RAG system
                                init_rag_system()
                                
                                import uuid
                                document_id = str(uuid.uuid4()).replace('-', '')[:16]
                                
                                rag_stats = st.session_state.rag_system.process_document(
                                    extracted_text, document_id
                                )
                                
                                # Update chat with RAG processing info
                                st.session_state.chat_manager.update_rag_processing(
                                    rag_stats, current_chat.id  # Explicitly specify the chat ID
                                )
                                
                                st.success("✅ Document processed with RAG system!")
                                st.info(f"Created {rag_stats['total_chunks']} chunks for semantic search")
                                
                        except Exception as e:
                            logger.error(f"RAG processing failed after reinit: {e}")
                            st.error(f"❌ **RAG processing failed**: {str(e)}")
                            st.error("⚠️ Please try again or check your RAG system configuration.")
                            return
                    else:
                        st.error("❌ **RAG system not available**")
                        st.error("Please check the RAG Settings in the sidebar and retry initialization.")
                        return
                    
                    # Force a rerun to show chat interface
                    st.rerun()
                else:
                    st.error("Failed to update document in current chat")
            else:
                st.error("❌ **Document Processing Failed**")
                st.error("Could not extract readable text from this PDF.")
                st.markdown("""
                **This could be due to:**
                - The PDF contains only images or scanned content
                - The PDF is corrupted or password-protected
                - The PDF format is not supported
                """)
        except Exception as e:
            st.error("❌ **Document Processing Failed**")
            st.error(f"Error processing file: {e}")
            st.info("💡 Please try a different PDF file.")


def render_chat_interface():
    """Render the main chat interface"""
    current_chat = st.session_state.chat_manager.get_current_chat()
    if not current_chat:
        return
    
    # Show current document info
    if current_chat.document_name:
        with st.expander("📄 Current Document", expanded=False):
            st.write(f"**Document:** {current_chat.document_name}")
            
            # Show RAG processing status
            if current_chat.rag_processed:
                rag_stats = current_chat.rag_stats or {}
                st.success("✅ Processed with RAG system")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Chunks Created", rag_stats.get("total_chunks", 0))
                with col2:
                    st.metric("Chunk Size", f"{rag_stats.get('chunk_size', 0)} tokens")
            else:
                st.warning("⚠️ Document not processed with RAG system")
            
            # Show PDF and extracted text side by side
            if current_chat.document_content and current_chat.document_text:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("📄 PDF Document")
                    pdf_viewer(
                        input=current_chat.document_content,
                        width="100%",
                        height=600,
                        render_text=True,
                        key=f"pdf_viewer_{current_chat.id}"
                    )
                
                with col2:
                    st.subheader("📝 Extracted Text")
                    # Show extracted text in a scrollable container
                    st.text_area(
                        "Document content:",
                        value=current_chat.document_text,
                        height=600,
                        disabled=True,
                        label_visibility="collapsed"
                    )
    
    # Display chat messages
    for message in current_chat.messages:
        with st.chat_message(message.role):
            st.markdown(message.content)
    
    # Chat input - only show if document text is valid and not currently generating
    if current_chat.document_text and current_chat.document_text.strip():
        if st.session_state.generating:
            st.info("🤖 Generating response... Use the stop button above to interrupt.")
            st.chat_input("Generating response...", disabled=True)
        else:
            if prompt := st.chat_input("Ask about your document..."):
                if not st.session_state.selected_model:
                    st.warning("Please select a model first")
                    return
                
                # Add user message
                with st.chat_message("user"):
                    st.markdown(prompt)
                current_chat.add_message("user", prompt)
                
                # Generate AI response
                with st.chat_message("assistant"):
                    response = generate_response_with_ui(prompt, current_chat)
                    current_chat.add_message("assistant", response)
                    show_citations(response, current_chat, prompt)
    else:
        st.warning("⚠️ **Chat Disabled**: No valid document content available. Please upload a PDF document with readable text to start chatting.")


def generate_response_with_ui(prompt, current_chat):
    """Generate AI response with UI components (streaming, reasoning, stop button) using RAG"""
    # Set generation state
    st.session_state.generating = True
    st.session_state.stop_generation = False
    
    try:
        # Check RAG system requirements and load appropriate document
        if not (st.session_state.rag_system and 
                current_chat.rag_processed and
                current_chat.document_id):
            st.error("❌ **RAG system not ready**")
            st.error("Please ensure the document is processed with RAG system.")
            return "Error: RAG system not ready for query processing."
        
        # Ensure the correct document is loaded for this chat
        if (st.session_state.rag_system.current_document_id != current_chat.document_id):
            try:
                success = st.session_state.rag_system.load_document(current_chat.document_id)
                if not success:
                    st.error(f"❌ **Could not load document** for this chat")
                    st.error("The document may have been deleted from the RAG system.")
                    return "Error: Could not load document for this chat."
            except Exception as e:
                st.error(f"❌ **Error loading document**: {str(e)}")
                return f"Error: Could not load document - {str(e)}"
        
        logger.info("Using RAG system for response generation")
        
        # Get retrieved chunks for RAG
        try:
            retrieval_info = st.session_state.rag_system.get_retrieval_info(prompt)
            
            # Get all retrieved chunks (before filtering) for fallback
            try:
                from llama_index.core.retrievers import VectorIndexRetriever
                retriever = VectorIndexRetriever(
                    index=st.session_state.rag_system.index,
                    similarity_top_k=st.session_state.rag_system.top_k
                )
                all_nodes = retriever.retrieve(prompt)
            except Exception as e:
                logger.warning(f"Could not get unfiltered chunks: {e}")
                all_nodes = []
            
            # Display retrieved chunks information IMMEDIATELY (before generation)
            if retrieval_info:
                # Show chunks that passed the similarity threshold
                with st.expander(f"📚 Retrieved {len(retrieval_info)} relevant chunks", expanded=False):
                    for chunk in retrieval_info:
                        st.markdown(f"**Chunk {chunk['chunk_id']}** (Score: {chunk['score']:.3f})")
                        st.text_area(
                            f"Content {chunk['chunk_id']} ({len(chunk['text'])} characters):",
                            value=chunk['text'],
                            height=200,
                            disabled=True,
                            key=f"chunk_{chunk['chunk_id']}_{hash(prompt)}"
                        )
                        st.markdown("---")
                context_text = "\n\n".join([chunk['text'] for chunk in retrieval_info])
            elif all_nodes:
                # Show chunks that were retrieved but didn't pass threshold
                st.warning(f"⚠️ **Similarity threshold too high**: Using top {min(3, len(all_nodes))} chunks with lower scores")
                with st.expander(f"📚 Using {min(3, len(all_nodes))} chunks (below threshold)", expanded=False):
                    for i, node in enumerate(all_nodes[:3]):
                        score = getattr(node, 'score', 0.0)
                        st.markdown(f"**Chunk {i+1}** (Score: {score:.3f}) - Below threshold ({st.session_state.rag_config['similarity_threshold']})")
                        st.text_area(
                            f"Content {i+1} ({len(node.text)} characters):",
                            value=node.text,
                            height=200,
                            disabled=True,
                            key=f"fallback_chunk_{i}_{hash(prompt)}"
                        )
                        st.markdown("---")
                context_text = "\n\n".join([node.text for node in all_nodes[:3]])
            else:
                st.error("❌ No chunks retrieved for this query")
                st.error("Try adjusting your question or lowering the similarity threshold in RAG Settings.")
                return "Error: No relevant content found for this query."
                
        except Exception as rag_error:
            logger.error(f"RAG retrieval failed: {rag_error}")
            st.error(f"❌ **RAG retrieval failed**: {str(rag_error)}")
            return f"Error: RAG retrieval failed - {str(rag_error)}"
        
        # Create containers for dynamic updates (after chunks are displayed)
        reasoning_placeholder = st.empty()
        answer_placeholder = st.empty()
        stop_container = st.container()
        
        # Show stop button
        with stop_container:
            stop_button_placeholder = st.empty()
            with stop_button_placeholder.container():
                col1, col2 = st.columns([10, 1])
                with col2:
                    if st.button("⏹", key=f"stop_gen_{hash(prompt)}", help="Stop generation"):
                        st.session_state.stop_generation = True
        
        # Generate response with streaming
        full_response = ""
        reasoning_content = ""
        answer_content = ""
        reasoning_started = False
        in_reasoning = False
        generation_stopped = False
        
        # Use direct ollama streaming with RAG context
        system_prompt = PromptBuilder.create_system_prompt(context_text, is_rag=True)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # Determine Ollama client based on environment
        if in_docker:
            client = ollama.Client(host=ollama_base_url)
            chat_stream = client.chat(
                model=st.session_state.selected_model,
                messages=messages,
                stream=True
            )
        else:
            chat_stream = ollama.chat(
                model=st.session_state.selected_model,
                messages=messages,
                stream=True
            )
        
        # Handle streaming response with reasoning support
        chunk_count = 0
        for chunk in chat_stream:
            chunk_count += 1
            
            # Check if user wants to stop on every chunk for responsiveness
            if st.session_state.get('stop_generation', False):
                generation_stopped = True
                logger.info(f"Generation stopped at chunk {chunk_count}")
                break
                
            if chunk['message']['content']:
                chunk_content = chunk['message']['content']
                full_response += chunk_content
                
                # Check for reasoning tags
                think_start = full_response.find('<think>')
                think_end = full_response.find('</think>')
                
                if think_start != -1:
                    reasoning_started = True
                    
                    if think_end != -1:
                        # Reasoning is complete, extract both parts
                        reasoning_content = full_response[think_start + 7:think_end].strip()
                        answer_content = full_response[think_end + 8:].strip()
                        in_reasoning = False
                        
                        # Show completed reasoning
                        with reasoning_placeholder.container():
                            with st.expander("🤔 Reasoning", expanded=False):
                                st.markdown(reasoning_content)
                        
                        # Show the actual answer
                        if answer_content:
                            answer_placeholder.markdown(answer_content)
                    else:
                        # Still in reasoning phase
                        in_reasoning = True
                        current_reasoning = full_response[think_start + 7:].strip()
                        
                        # Show reasoning with spinner or content
                        with reasoning_placeholder.container():
                            with st.expander("🤔 Reasoning", expanded=False):
                                if current_reasoning:
                                    st.markdown(current_reasoning)
                                else:
                                    with st.spinner("Thinking..."):
                                        st.empty()
                else:
                    # No reasoning tags detected, stream normally
                    answer_content = full_response
                    answer_placeholder.markdown(answer_content)
        
        # Handle stopped generation
        if generation_stopped or st.session_state.get('stop_generation', False):
            final_answer = answer_content if reasoning_started else full_response
            if final_answer.strip():
                final_answer += "\n\n*[Generation stopped by user]*"
            else:
                final_answer = "*[Generation stopped by user before any content was generated]*"
            
            # Update display with stopped message
            if reasoning_started and answer_content:
                answer_placeholder.markdown(final_answer)
            elif not reasoning_started:
                answer_placeholder.markdown(final_answer)
            
            # Show stopped message and remove stop button
            with stop_container:
                stop_button_placeholder.empty()
                st.info("🛑 Generation stopped by user")
        else:
            # Clear stop button when generation completes normally
            with stop_container:
                stop_button_placeholder.empty()
        
        # Show method information
        st.info("🔍 Response generated using RAG (semantic search)")
        
        # Return the final answer (without reasoning tags) for storage
        final_answer = answer_content if reasoning_started else full_response
        
        # Add stopped indicator if generation was interrupted
        if generation_stopped or st.session_state.get('stop_generation', False):
            if final_answer.strip():
                final_answer += "\n\n*[Generation stopped by user]*"
            else:
                final_answer = "*[Generation stopped by user before any content was generated]*"
        
        return final_answer
        
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return ""
    finally:
        # Reset generation state
        st.session_state.generating = False
        st.session_state.stop_generation = False


def main():
    """Main application function"""
    # Initialize session state
    init_session_state()
    
    # Render sidebar (which now includes model selection)
    render_sidebar()
    
    # Check if model is selected before proceeding
    if not st.session_state.selected_model:
        st.warning("⚠️ Please select a chat model in the sidebar to get started.")
        return
    
    # Main content - check if we have a valid document
    current_chat = st.session_state.chat_manager.get_current_chat()
    
    if not current_chat or not st.session_state.chat_manager.has_valid_document():
        render_document_upload()
    else:
        render_chat_interface()


if __name__ == "__main__":
    main() 