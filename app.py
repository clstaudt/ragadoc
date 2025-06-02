import streamlit as st
import ollama
import uuid
import os
from datetime import datetime
from streamlit_pdf_viewer import pdf_viewer
import pdfplumber
import io
from ragnarok import EnhancedPDFProcessor, RAGSystem, create_rag_system
from loguru import logger

# Configure loguru logging - simple console logging with defaults
# loguru automatically logs to console by default, no configuration needed

# Check if we're running in Docker
def is_running_in_docker():
    """Check if we're running inside a Docker container"""
    return (
        os.path.exists('/.dockerenv') or 
        os.environ.get('STREAMLIT_SERVER_ADDRESS') == '0.0.0.0'
    )

# Initialize environment
in_docker = is_running_in_docker()

if in_docker:
    # Docker environment - use configured endpoint
    ollama_base_url = os.environ.get('OLLAMA_BASE_URL', 'http://host.docker.internal:11434')
else:
    # Direct execution - use default ollama behavior (don't touch env vars at all)
    ollama_base_url = "http://localhost:11434"  # Just for display

# Configuration
st.set_page_config(
    page_title="Document Q&A", 
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("Document Q&A")

class ChatManager:
    """Enhanced chat management with RAG system integration"""
    
    def __init__(self):
        self.init_session_state()
        self.init_rag_system()
    
    def init_session_state(self):
        """Initialize all session state variables"""
        defaults = {
            "chats": {},
            "current_chat_id": None,
            "selected_model": None,
            "use_rag": True,  # Enable RAG by default
            "generating": False,  # Track if currently generating
            "stop_generation": False,  # Flag to stop generation
            "rag_config": {
                "chunk_size": 256,
                "chunk_overlap": 25,
                "similarity_threshold": 0.7,
                "top_k": 10,
                "embedding_model": "nomic-embed-text",
                "llm_model": None  # Will be set to selected model
            }
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
        
        # Create first chat if none exist
        if not st.session_state.chats:
            self.create_new_chat(clear_rag=False)  # Don't clear RAG on initialization
    
    def init_rag_system(self):
        """Initialize the RAG system"""
        if "rag_system" not in st.session_state:
            try:
                # Determine Ollama URL based on environment
                if is_running_in_docker():
                    ollama_url = os.environ.get('OLLAMA_BASE_URL', 'http://host.docker.internal:11434')
                else:
                    ollama_url = "http://localhost:11434"
                
                # Get available models to find the correct embedding model name
                available_models = ModelManager.get_available_models()
                embedding_model = st.session_state.rag_config["embedding_model"]
                
                # Check if we need to add :latest suffix
                if embedding_model not in available_models:
                    # Try with :latest suffix
                    embedding_model_with_suffix = f"{embedding_model}:latest"
                    if embedding_model_with_suffix in available_models:
                        embedding_model = embedding_model_with_suffix
                        logger.info(f"Using embedding model: {embedding_model}")
                
                # Create RAG system with current configuration
                rag_config = st.session_state.rag_config.copy()
                rag_config["embedding_model"] = embedding_model
                
                # Use the selected model from the UI, not the hardcoded default
                if st.session_state.selected_model:
                    rag_config["llm_model"] = st.session_state.selected_model
                    logger.info(f"Using selected LLM model: {st.session_state.selected_model}")
                else:
                    # If no model selected yet, try to get the first available model
                    if available_models:
                        # Use first non-embedding model
                        llm_models = [m for m in available_models if not any(embed in m.lower() for embed in ['embed', 'minilm'])]
                        if llm_models:
                            rag_config["llm_model"] = llm_models[0]
                            logger.info(f"No model selected, using first available: {llm_models[0]}")
                
                st.session_state.rag_system = create_rag_system(
                    ollama_base_url=ollama_url,
                    **rag_config
                )
                logger.info("RAG system initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize RAG system: {e}")
                st.session_state.rag_system = None
    
    def create_new_chat(self, clear_rag=False):
        """Create a new chat session with optional vector store clearing"""
        chat_id = str(uuid.uuid4())
        st.session_state.chats[chat_id] = {
            "messages": [],
            "created_at": datetime.now(),
            "title": "New Chat",
            "document_name": None,
            "document_content": None,
            "document_text": "",
            "rag_processed": False,
            "rag_stats": None
        }
        st.session_state.current_chat_id = chat_id
        
        # Only clear the RAG system if explicitly requested
        if clear_rag and st.session_state.get("rag_system"):
            try:
                st.session_state.rag_system.clear_all_documents()
                logger.info(f"Cleared RAG system for new chat: {chat_id}")
            except Exception as e:
                logger.warning(f"Could not clear RAG system: {e}")
        
        return chat_id
    
    def get_current_chat(self):
        """Get current chat data"""
        if st.session_state.current_chat_id:
            return st.session_state.chats.get(st.session_state.current_chat_id, {})
        return {}
    
    def add_message(self, role, content):
        """Add message to current chat"""
        chat = self.get_current_chat()
        if chat:
            chat["messages"].append({"role": role, "content": content})
            
            # Update title from first user message
            if role == "user" and chat["title"] == "New Chat":
                chat["title"] = content[:50] + ("..." if len(content) > 50 else "")
    
    def delete_chat(self, chat_id):
        """Delete a chat"""
        if chat_id in st.session_state.chats:
            del st.session_state.chats[chat_id]
            if st.session_state.current_chat_id == chat_id:
                # Switch to another chat or create new one
                if st.session_state.chats:
                    st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]
                else:
                    self.create_new_chat(clear_rag=False)  # Don't clear RAG when auto-creating after deletion

class ModelManager:
    """Simplified model management"""
    
    @staticmethod
    def get_available_models():
        """Get available Ollama models"""
        try:
            if in_docker:
                # Docker - use explicit client configuration
                client = ollama.Client(host=ollama_base_url)
                models_info = client.list()
            else:
                # Direct execution - use default ollama client
                models_info = ollama.list()
                
            # Handle both dict and ListResponse object
            if hasattr(models_info, 'models'):
                models_list = models_info.models
                return [model.model if hasattr(model, 'model') else model.get('model', model.get('name', '')) 
                       for model in models_list]
            elif isinstance(models_info, dict) and 'models' in models_info:
                return [model.get('model', model.get('name', '')) 
                       for model in models_info['models']]
            return []
        except Exception as e:
            # Provide helpful error message based on likely causes
            error_msg = f"Error connecting to Ollama: {e}"
            if "Connection refused" in str(e) or "No connection" in str(e):
                error_msg += "\n\nPlease ensure Ollama is running:\n"
                if in_docker:
                    error_msg += "- For Docker access: `OLLAMA_HOST=0.0.0.0:11434 ollama serve`"
                else:
                    error_msg += "- Direct execution: `ollama serve`"
            st.error(error_msg)
            return []
    
    @staticmethod
    def get_model_info(model_name):
        """Get detailed model information including context length"""
        try:
            if in_docker:
                # Docker - use explicit client configuration
                client = ollama.Client(host=ollama_base_url)
                model_info = client.show(model_name)
            else:
                # Direct execution - use default ollama client
                model_info = ollama.show(model_name)
            
            return model_info
        except Exception as e:
            logger.warning(f"Could not get model info for {model_name}: {e}")
            return None
    
    @staticmethod
    def get_context_length(model_name):
        """Get the context length for a specific model - returns None if cannot be determined"""
        model_info = ModelManager.get_model_info(model_name)
        
        # Try to get context length from model parameters first
        if model_info:
            try:
                # Check in parameters first (handle both dict and object)
                parameters = None
                if hasattr(model_info, 'parameters'):
                    parameters = model_info.parameters
                elif isinstance(model_info, dict) and 'parameters' in model_info:
                    parameters = model_info['parameters']
                
                if parameters:
                    num_ctx = None
                    if hasattr(parameters, 'get') and parameters.get('num_ctx'):
                        num_ctx = parameters['num_ctx']
                    elif hasattr(parameters, 'num_ctx'):
                        num_ctx = parameters.num_ctx
                    elif isinstance(parameters, dict) and 'num_ctx' in parameters:
                        num_ctx = parameters['num_ctx']
                    
                    if num_ctx:
                        return int(num_ctx)
                
                # Check in modelfile for PARAMETER num_ctx
                modelfile = None
                if hasattr(model_info, 'modelfile'):
                    modelfile = model_info.modelfile
                elif isinstance(model_info, dict) and 'modelfile' in model_info:
                    modelfile = model_info['modelfile']
                
                if modelfile:
                    import re
                    ctx_match = re.search(r'PARAMETER\s+num_ctx\s+(\d+)', modelfile, re.IGNORECASE)
                    if ctx_match:
                        return int(ctx_match.group(1))
                
                # Get model family for fallback detection
                family = None
                if hasattr(model_info, 'details') and hasattr(model_info.details, 'family'):
                    family = model_info.details.family
                elif isinstance(model_info, dict) and 'details' in model_info and 'family' in model_info['details']:
                    family = model_info['details']['family']
                        
            except Exception as e:
                logger.warning(f"Error parsing model info for {model_name}: {e}")
        
        # Return None if we truly cannot determine it
        logger.warning(f"Could not determine context length for {model_name} - no explicit parameter found and unknown model family")
        return None

class ContextChecker:
    """Utility class for checking context window compatibility"""
    
    @staticmethod
    def estimate_token_count(text):
        """Estimate token count using multiple methods for better accuracy"""
        if not text:
            return 0
        
        # Try to use tiktoken for more accurate counting (if available)
        try:
            import tiktoken
            # Use cl100k_base encoding (used by GPT-4, similar to most modern LLMs)
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except ImportError:
            pass
        
        # Fallback: Multiple estimation methods
        char_count = len(text)
        word_count = len(text.split())
        
        # Different estimation approaches
        char_based = char_count // 4  # ~4 chars per token (conservative)
        word_based = int(word_count * 1.3)  # ~1.3 tokens per word (average)
        
        # Use the higher estimate to be conservative
        return max(char_based, word_based)
    
    @staticmethod
    def check_document_fits_context(document_text, model_name, user_prompt=""):
        """Check if document + system prompt fits in model's context window"""
        if not document_text or not model_name:
            return True, None, None
        
        context_length = ModelManager.get_context_length(model_name)
        if context_length is None:
            return True, None, f"Cannot determine context length for model '{model_name}'. Context checking skipped."
        
        # Generate the actual system prompt to measure its real size
        system_prompt = f"""You are a document analysis assistant. Answer questions ONLY using information from these relevant document excerpts:

RELEVANT DOCUMENT EXCERPTS:
{document_text}

INITIAL CHECK:
First, verify you have received document excerpts above. If the excerpts are empty or missing, respond: "Error: No relevant document content found for this question."

RESPONSE RULES:
Choose ONE approach based on whether the excerpts contain relevant information:

1. **IF ANSWERABLE**: Provide a complete answer with citations
- Every factual claim must have a citation [1], [2], etc.
- List citations at the end using this exact format:
    [1] "exact quote from document"
    [2] "another exact quote"

2. **IF NOT ANSWERABLE**: Decline to answer
- State: "I cannot answer this based on the available document excerpts"
- Do NOT include any citations when declining
- Do not attempt to answer the question with your own knowledge.

CITATION GUIDELINES:
- Use verbatim quotes in their original language (never translate)
- Quote meaningful phrases (5-15 words) that provide sufficient context
- Include descriptive context around numbers/measurements (e.g., "increased by 50% compared to" not just "50%")
- Avoid very short snippets like single numbers, dates, or isolated words
- Each citation should be substantial enough to be meaningful on its own
- Each citation on its own line

LANGUAGE RULES:
- Respond in the user's language
- Keep citations in the document's original language
"""
        
        # Measure actual token counts
        system_tokens = ContextChecker.estimate_token_count(system_prompt)
        user_tokens = ContextChecker.estimate_token_count(user_prompt)
        
        # Reserve space for response (conservative estimate)
        response_reserve = 1000
        
        total_tokens = system_tokens + user_tokens + response_reserve
        
        fits = total_tokens <= context_length
        usage_percent = (total_tokens / context_length) * 100
        
        return fits, {
            'context_length': context_length,
            'system_tokens': system_tokens,
            'user_tokens': user_tokens,
            'response_reserve': response_reserve,
            'total_estimated_tokens': total_tokens,
            'usage_percent': usage_percent,
            'available_tokens': context_length - total_tokens
        }, None
    
    @staticmethod
    def display_context_warning(context_info, model_name):
        """Display context window usage information and warnings"""
        if not context_info:
            return
        
        usage_percent = context_info['usage_percent']
        
        if usage_percent > 100:
            st.error("⚠️ **Document Too Large for Context Window**")
            st.error(f"""
            **Model:** {model_name}  
            **Context Limit:** {context_info['context_length']:,} tokens  
            **Document Size:** ~{context_info['document_tokens']:,} tokens  
            **Usage:** {usage_percent:.1f}% (exceeds limit by {context_info['total_estimated_tokens'] - context_info['context_length']:,} tokens)
            
            **The document is too large for this model's context window.**
            """)
            
            with st.expander("💡 Solutions for Large Documents", expanded=True):
                st.markdown(f"""
                **Option 1: Use a model with larger context window**
                - Switch to a model like `llama3.1:8b` (128k context) or `mistral:latest` (32k context)
                
                **Option 2: Create a custom model with larger context**
                ```bash
                # Create a Modelfile
                echo "FROM {model_name}
                PARAMETER num_ctx 32768" > Modelfile
                
                # Create custom model
                ollama create {model_name.split(':')[0]}-large -f Modelfile
                ```
                
                **Option 3: Document chunking (future feature)**
                - Break document into smaller chunks
                - Process each chunk separately
                """)
                
        elif usage_percent > 80:
            st.warning("⚠️ **High Context Usage**")
            st.warning(f"""
            **Model:** {model_name}  
            **Context Limit:** {context_info['context_length']:,} tokens  
            **Document Size:** ~{context_info['document_tokens']:,} tokens  
            **Usage:** {usage_percent:.1f}% of context window  
            **Available:** ~{context_info['available_tokens']:,} tokens for conversation
            
            **The document uses most of the context window. Long conversations may be truncated.**
            """)
            
        elif usage_percent > 50:
            st.info("ℹ️ **Moderate Context Usage**")
            st.info(f"""
            **Model:** {model_name}  
            **Context Usage:** {usage_percent:.1f}% ({context_info['document_tokens']:,} / {context_info['context_length']:,} tokens)  
            **Available:** ~{context_info['available_tokens']:,} tokens for conversation
            """)
            
        else:
            st.success("✅ **Document fits comfortably in context window**")
            st.success(f"""
            **Model:** {model_name}  
            **Context Usage:** {usage_percent:.1f}% ({context_info['document_tokens']:,} / {context_info['context_length']:,} tokens)  
            **Available:** ~{context_info['available_tokens']:,} tokens for conversation
            """)

def render_sidebar(chat_manager):
    """Render the sidebar with chat history and RAG configuration"""
    with st.sidebar:
        st.header("Chat History")
        
        # New chat button
        if st.button("New Chat", use_container_width=True, type="primary"):
            chat_manager.create_new_chat(clear_rag=True)  # Clear RAG for explicit new chat
            st.rerun()
        
        st.divider()
        
        # RAG Configuration
        with st.expander("🔍 RAG Settings", expanded=False):
            # RAG toggle
            use_rag = st.checkbox(
                "Enable RAG (Semantic Search)",
                value=st.session_state.use_rag,
                help="Use Retrieval-Augmented Generation for better handling of large documents"
            )
            
            if use_rag != st.session_state.use_rag:
                st.session_state.use_rag = use_rag
                st.rerun()
            
            if use_rag:
                st.info("🔍 **Smart Retrieval**: The system first finds ALL chunks above the similarity threshold, then limits to the max number. This ensures no high-quality chunks are missed.")
                
                # RAG parameters
                chunk_size = st.slider(
                    "Chunk Size (tokens)",
                    min_value=256,
                    max_value=1024,
                    value=st.session_state.rag_config["chunk_size"],
                    step=64,
                    help="Size of text chunks for processing"
                )
                
                chunk_overlap = st.slider(
                    "Chunk Overlap (tokens)",
                    min_value=0,
                    max_value=200,
                    value=st.session_state.rag_config["chunk_overlap"],
                    step=10,
                    help="Overlap between consecutive chunks"
                )
                
                similarity_threshold = st.slider(
                    "Similarity Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.rag_config["similarity_threshold"],
                    step=0.05,
                    help="Minimum similarity score for retrieved chunks"
                )
                
                top_k = st.slider(
                    "Max Retrieved Chunks",
                    min_value=1,
                    max_value=20,  # Increased max since we now get all high-similarity chunks first
                    value=st.session_state.rag_config["top_k"],
                    step=1,
                    help="Maximum chunks to include in final answer (after filtering by similarity threshold). All chunks above similarity threshold are found first, then limited to this number."
                )
                
                embedding_model = st.selectbox(
                    "Embedding Model",
                    options=["nomic-embed-text", "mxbai-embed-large", "all-minilm"],
                    index=0 if st.session_state.rag_config["embedding_model"] == "nomic-embed-text" else 1,
                    help="Model used for generating embeddings"
                )
                
                # Update configuration if changed
                new_config = {
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "similarity_threshold": similarity_threshold,
                    "top_k": top_k,
                    "embedding_model": embedding_model
                }
                
                if new_config != st.session_state.rag_config:
                    st.session_state.rag_config = new_config
                    st.info("⚠️ RAG settings changed. Upload a new document to apply changes.")
            
            # RAG system status
            if st.session_state.get("rag_system"):
                st.success("✅ RAG System Ready")
                
                # Show current document status
                current_chat = chat_manager.get_current_chat()
                if current_chat.get("rag_processed"):
                    stats = current_chat.get("rag_stats", {})
                    st.info(f"📊 Document: {stats.get('total_chunks', 0)} chunks")
                else:
                    st.warning("📄 No RAG-processed document")
                    
                    # Show last RAG error if any
                    if hasattr(st.session_state, 'rag_error'):
                        with st.expander("⚠️ Last RAG Error", expanded=False):
                            st.error(st.session_state.rag_error)
                            if st.button("Clear Error", key="clear_rag_error"):
                                del st.session_state.rag_error
                                st.rerun()
            else:
                st.error("❌ RAG System Not Available")
                if st.button("🔄 Retry RAG Initialization", help="Try to reinitialize the RAG system"):
                    # Clear the failed RAG system and try again
                    if "rag_system" in st.session_state:
                        del st.session_state["rag_system"]
                    chat_manager.init_rag_system()
                    st.rerun()
        
        st.divider()
        
        # Chat history
        if st.session_state.chats:
            sorted_chats = sorted(
                st.session_state.chats.items(),
                key=lambda x: x[1]["created_at"],
                reverse=True
            )
            
            for chat_id, chat_data in sorted_chats:
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    is_current = chat_id == st.session_state.current_chat_id
                    button_type = "primary" if is_current else "secondary"
                    
                    if st.button(
                        chat_data["title"],
                        key=f"chat-{chat_id}",
                        use_container_width=True,
                        type=button_type
                    ):
                        st.session_state.current_chat_id = chat_id
                        st.rerun()
                
                with col2:
                    if st.button("×", key=f"del-{chat_id}", help="Delete"):
                        chat_manager.delete_chat(chat_id)
                        st.rerun()

def render_document_upload(chat_manager):
    """Render document upload interface"""
    st.header("Upload Document")
    st.info("Upload a PDF document to start chatting")
    
    # Clear upload button for problematic files
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🗑️ Clear Upload", help="Clear file upload state if stuck"):
            # Force clear the uploader by creating a new chat
            chat_manager.create_new_chat(clear_rag=True)  # Clear RAG when clearing upload issues
            st.rerun()
    
    # Use a unique key per chat to avoid file state conflicts
    uploader_key = f"uploader_{st.session_state.current_chat_id}"
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        key=uploader_key
    )
    
    if uploaded_file is not None:
        try:
            with st.spinner("Processing PDF..."):
                # Extract text using EnhancedPDFProcessor directly
                try:
                    pdf_bytes = uploaded_file.getvalue()
                    processor = EnhancedPDFProcessor(pdf_bytes)
                    extracted_text = processor.extract_full_text()
                except Exception as e:
                    st.error(f"Error extracting text from PDF: {e}")
                    # Fallback to basic extraction if enhanced fails
                    try:
                        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                            extracted_text = ""
                            for page in pdf.pages:
                                page_text = page.extract_text()
                                if page_text:
                                    extracted_text += page_text + "\n"
                    except Exception as fallback_e:
                        st.error(f"Fallback extraction also failed: {fallback_e}")
                        extracted_text = ""
            
            if extracted_text and extracted_text.strip():
                # Update current chat with document
                chat = chat_manager.get_current_chat()
                chat.update({
                    "document_name": uploaded_file.name,
                    "document_content": uploaded_file.getvalue(),
                    "document_text": extracted_text,
                    "title": f"Doc: {uploaded_file.name}",
                    "rag_processed": False,
                    "rag_stats": None
                })
                
                st.success(f"Document '{uploaded_file.name}' processed successfully!")
                st.info(f"Extracted {len(extracted_text.split()):,} words")
                
                # Process with RAG system if enabled
                if st.session_state.use_rag and st.session_state.rag_system:
                    try:
                        with st.spinner("Processing document with RAG system..."):
                            # Generate a simple, safe document ID using UUID
                            import uuid
                            document_id = str(uuid.uuid4()).replace('-', '')[:16]  # 16 chars, alphanumeric only
                            
                            logger.info(f"Starting RAG processing for document: {document_id}")
                            logger.info(f"Original filename: {uploaded_file.name}")
                            logger.info(f"Document text length: {len(extracted_text)} characters")
                            
                            rag_stats = st.session_state.rag_system.process_document(
                                extracted_text, document_id
                            )
                            logger.info(f"RAG processing completed: {rag_stats}")
                            
                            # Update chat with RAG processing info
                            chat["rag_processed"] = True
                            chat["rag_stats"] = rag_stats
                            
                            st.success("✅ Document processed with RAG system!")
                            st.info(f"Created {rag_stats['total_chunks']} chunks for semantic search")
                            
                    except Exception as e:
                        logger.error(f"RAG processing failed: {e}")
                        # Store error in session state so it persists
                        st.session_state.rag_error = str(e)
                        st.error(f"❌ **RAG processing failed**: {str(e)}")
                        st.warning("⚠️ Falling back to traditional full-document processing")
                        st.info("💡 Try restarting the app or check that your embedding model is working")
                        
                        # Show detailed error in expander
                        with st.expander("🔍 Error Details", expanded=False):
                            st.code(str(e))
                            st.caption("Check the terminal/logs for more details")
                elif st.session_state.use_rag and not st.session_state.rag_system:
                    st.error("❌ **RAG System Not Available**")
                    st.warning("⚠️ RAG is enabled but system failed to initialize")
                    st.info("💡 Check logs for initialization errors")
                
                # Check context window compatibility
                if st.session_state.selected_model:
                    fits, context_info, error = ContextChecker.check_document_fits_context(
                        extracted_text, st.session_state.selected_model
                    )
                    
                    if error:
                        st.warning(f"Could not check context compatibility: {error}")
                    elif context_info:
                        usage_percent = context_info['usage_percent']
                        
                        # Always show progress bar and basic info after upload
                        st.markdown("---")
                        st.markdown("**📊 Context Check:**")
                        
                        # Show progress bar for context usage
                        progress_value = min(usage_percent / 100, 1.0)  # Cap at 100% for display
                        st.progress(progress_value, text=f"Context Usage: {usage_percent:.1f}%")
                        
                        # Show status with appropriate color and clear messaging
                        if usage_percent > 100:
                            st.error(f"⚠️ **Document too large** - Uses {usage_percent:.0f}% of context window")
                            excess_tokens = context_info['total_estimated_tokens'] - context_info['context_length']
                            st.caption(f"Document exceeds limit by ~{excess_tokens:,} tokens")
                        elif usage_percent > 80:
                            st.warning(f"⚠️ **High context usage** - {usage_percent:.0f}% of {context_info['context_length']:,} tokens")
                            st.caption(f"~{context_info['available_tokens']:,} tokens remaining for conversation")
                        elif usage_percent > 50:
                            st.info(f"ℹ️ **Moderate context usage** - {usage_percent:.0f}% of {context_info['context_length']:,} tokens")
                            st.caption(f"~{context_info['available_tokens']:,} tokens remaining for conversation")
                        else:
                            st.success(f"✅ **Good fit** - Uses {usage_percent:.0f}% of {context_info['context_length']:,} tokens")
                            st.caption(f"~{context_info['available_tokens']:,} tokens remaining for conversation")
                        
                        # Show breakdown in expander
                        with st.expander("📊 Token Breakdown", expanded=False):
                            st.write(f"**System prompt:** ~{context_info['system_tokens']:,} tokens")
                            st.write(f"**Response reserve:** ~{context_info['response_reserve']:,} tokens")
                            st.write(f"**Total estimated:** ~{context_info['total_estimated_tokens']:,} tokens")
                            st.write(f"**Context limit:** {context_info['context_length']:,} tokens")
                else:
                    st.info("💡 Select a model to check context window compatibility")
                
                # Only rerun if RAG processing was successful, otherwise show errors
                if not st.session_state.use_rag or chat.get("rag_processed"):
                    st.rerun()
            else:
                st.error("❌ **Document Processing Failed**")
                st.error("Could not extract readable text from this PDF. This could be due to:")
                st.markdown("""
                - The PDF contains only images or scanned content
                - The PDF is corrupted or password-protected
                - The PDF format is not supported
                
                **Please try:**
                - A different PDF document with selectable text
                - Converting scanned PDFs to text-searchable format first
                - Ensuring the PDF is not password-protected
                """)
                st.info("💡 The chat interface will remain disabled until a valid document is uploaded.")
        except Exception as e:
            st.error("❌ **Document Processing Failed**")
            st.error(f"Error processing file: {e}")
            st.markdown("""
            **This error occurred while trying to process your PDF. Common causes:**
            - File corruption or invalid PDF format
            - Insufficient memory for large files
            - Network issues during upload
            
            **Please try:**
            - Uploading a different PDF file
            - Clicking '🗑️ Clear Upload' and trying again
            - Ensuring the file is a valid PDF document
            """)
            st.info("💡 The chat interface will remain disabled until a valid document is uploaded.")

def render_chat_interface(chat_manager):
    """Render the main chat interface"""
    chat = chat_manager.get_current_chat()
    
    # Show current document info
    if chat.get("document_name"):
        with st.expander("📄 Current Document", expanded=False):
            st.write(f"**Document:** {chat['document_name']}")
            
            # Show RAG processing status
            if chat.get("rag_processed"):
                rag_stats = chat.get("rag_stats", {})
                st.success("✅ Processed with RAG system")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Chunks Created", rag_stats.get("total_chunks", 0))
                with col2:
                    st.metric("Chunk Size", f"{rag_stats.get('chunk_size', 0)} tokens")
            elif st.session_state.use_rag:
                st.warning("⚠️ RAG processing not completed")
            else:
                st.info("📄 Using traditional full-document processing")
            
            # Show context compatibility info with progress bar (for traditional processing)
            if not chat.get("rag_processed") and st.session_state.selected_model and chat.get("document_text"):
                fits, context_info, error = ContextChecker.check_document_fits_context(
                    chat["document_text"], st.session_state.selected_model
                )
                
                if context_info:
                    usage_percent = context_info['usage_percent']
                    
                    # Show progress bar for context usage
                    progress_value = min(usage_percent / 100, 1.0)  # Cap at 100% for display
                    st.progress(progress_value, text=f"Context Usage: {usage_percent:.1f}%")
                    
                    # Show brief summary
                    st.caption(f"~{context_info['system_tokens']:,} tokens / {context_info['context_length']:,} limit")
                    
                    # Recommend RAG for large documents
                    if usage_percent > 80:
                        st.info("💡 Consider enabling RAG for better handling of this large document")
            
            # Show PDF and extracted text side by side
            if chat.get("document_content") and chat.get("document_text"):
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("📄 PDF Document")
                    pdf_viewer(
                        input=chat["document_content"],
                        width="100%",
                        height=600,
                        render_text=True,
                        key=f"pdf_viewer_{st.session_state.current_chat_id}"
                    )
                
                with col2:
                    st.subheader("📝 Extracted Text")
                    # Show extracted text in a scrollable container
                    st.text_area(
                        "Document content:",
                        value=chat["document_text"],
                        height=600,
                        disabled=True,
                        label_visibility="collapsed"
                    )
    
    # Display chat messages
    for message in chat.get("messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input - only show if document text is valid and not currently generating
    document_text = chat.get("document_text", "")
    if document_text and document_text.strip():
        # Show generation status or chat input
        if st.session_state.generating:
            st.info("🤖 Generating response... Use the stop button above to interrupt.")
            # Disable input during generation
            st.chat_input("Generating response...", disabled=True)
        else:
            if prompt := st.chat_input("Ask about your document..."):
                if not st.session_state.selected_model:
                    st.warning("Please select a model first")
                    return
                
                # Add user message
                with st.chat_message("user"):
                    st.markdown(prompt)
                chat_manager.add_message("user", prompt)
                
                # Generate AI response
                with st.chat_message("assistant"):
                    try:
                        # Debug: Log the RAG decision logic
                        logger.info(f"RAG Decision Debug:")
                        logger.info(f"  use_rag: {st.session_state.use_rag}")
                        logger.info(f"  rag_system exists: {st.session_state.rag_system is not None}")
                        logger.info(f"  rag_processed: {chat.get('rag_processed')}")
                        logger.info(f"  rag_system.index exists: {getattr(st.session_state.rag_system, 'index', None) is not None if st.session_state.rag_system else False}")
                        
                        # Try RAG system first if available and document is processed
                        if (st.session_state.use_rag and 
                            st.session_state.rag_system and 
                            chat.get("rag_processed") and
                            getattr(st.session_state.rag_system, 'index', None) is not None):
                            try:
                                logger.info("Using RAG system for response generation")
                                
                                # Get retrieved chunks for RAG
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
                                
                                # Display retrieved chunks information
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
                                    st.error("❌ No chunks retrieved - this shouldn't happen with a processed document")
                                    context_text = ""
                                
                                # Generate response using retrieved context
                                response = generate_ai_response_unified(prompt, context_text, is_rag=True, chat_data=chat)
                                chat_manager.add_message("assistant", response)
                                
                                # Show citations for RAG method too
                                show_citations(response, chat, prompt)
                                
                                # Show RAG-specific information
                                st.info("🔍 Response generated using RAG (semantic search)")
                                
                            except Exception as rag_error:
                                logger.warning(f"RAG response failed, falling back to traditional method: {rag_error}")
                                st.warning("⚠️ RAG system failed, using traditional processing")
                                
                                # Fallback to traditional method
                                response = generate_ai_response_unified(prompt, chat["document_text"], is_rag=False)
                                chat_manager.add_message("assistant", response)
                                
                                # Show citations for traditional method
                                show_citations(response, chat, prompt)
                        else:
                            # Use traditional method
                            logger.info("Using traditional method for response generation")
                            if not st.session_state.use_rag:
                                logger.info("  Reason: RAG disabled")
                            elif not st.session_state.rag_system:
                                logger.info("  Reason: RAG system not available")
                            elif not chat.get("rag_processed"):
                                logger.info("  Reason: Document not RAG processed")
                            elif getattr(st.session_state.rag_system, 'index', None) is None:
                                logger.info("  Reason: RAG system has no index")
                                
                            response = generate_ai_response_unified(prompt, chat["document_text"], is_rag=False)
                            # Note: Display is handled within generate_ai_response for reasoning support
                            chat_manager.add_message("assistant", response)
                            
                            # Show citations
                            show_citations(response, chat, prompt)
                            
                            # Show info about processing method
                            if not st.session_state.use_rag:
                                st.info("📄 Response generated using full document (RAG disabled)")
                            elif not chat.get("rag_processed"):
                                st.info("📄 Response generated using full document (RAG not processed)")
                            elif getattr(st.session_state.rag_system, 'index', None) is None:
                                st.info("📄 Response generated using full document (RAG system has no index)")
                            
                    except Exception as e:
                        st.error(f"Error generating response: {e}")
                        # Reset generation state on error
                        st.session_state.generating = False
                        st.session_state.stop_generation = False
    else:
        # Show message when document processing failed
        st.warning("⚠️ **Chat Disabled**: No valid document content available. Please upload a PDF document with readable text to start chatting.")
        st.info("The document may have failed to process, or the extracted text may be empty. Try uploading a different PDF file.")

def create_system_prompt(document_content, is_rag=False):
    """
    Create a unified system prompt by assembling reusable components
    
    Args:
        document_content: Either full document text or retrieved chunks text
        is_rag: Boolean indicating if this is for RAG (chunks) or traditional (full doc)
    
    Returns:
        Formatted system prompt string
    """
    # Base role and instruction (shared)
    base_role = "You are a document analysis assistant. Answer questions ONLY using information from this"
    
    # Content section (varies by method)
    if is_rag:
        content_section = f"""relevant document excerpts:

RELEVANT DOCUMENT EXCERPTS:
{document_content}"""
        content_type = "relevant document excerpts"
        content_plural = "excerpts are"
        error_msg = "Error: No relevant document content found for this question."
        decline_msg = "I cannot answer this based on the available document excerpts"
    else:
        content_section = f"""document:

DOCUMENT CONTENT:
{document_content}"""
        content_type = "document"
        content_plural = "document is"
        error_msg = "Error: No document content received, cannot proceed."
        decline_msg = "I cannot answer this based on the document"
    
    # Initial check (varies by method)
    initial_check = f"""INITIAL CHECK:
First, verify you have received {content_type} above. If the {content_plural} empty or missing, respond: "{error_msg}\""""
    
    # Response rules (mostly shared, slight variation in decline message)
    response_rules = f"""RESPONSE RULES:
Choose ONE approach based on whether the {content_type} contain{'s' if not is_rag else ''} relevant information:

1. **IF ANSWERABLE**: Provide a complete answer with citations
- Every factual claim must have a citation [1], [2], etc.
- List citations at the end using this exact format:
    [1] "exact quote from document"
    [2] "another exact quote"

2. **IF NOT ANSWERABLE**: Decline to answer
- State: "{decline_msg}"
- Do NOT include any citations when declining
- Do not attempt to answer the question with your own knowledge."""
    
    # Citation guidelines (shared)
    citation_guidelines = """CITATION GUIDELINES:
- Use verbatim quotes in their original language (never translate)
- Quote meaningful phrases (5-15 words) that provide sufficient context
- Include descriptive context around numbers/measurements (e.g., "increased by 50% compared to" not just "50%")
- Avoid very short snippets like single numbers, dates, or isolated words
- Each citation should be substantial enough to be meaningful on its own
- Each citation on its own line"""
    
    # Language rules (shared)
    language_rules = """LANGUAGE RULES:
- Respond in the user's language
- Keep citations in the document's original language"""
    
    # Assemble the complete prompt
    return f"""{base_role} {content_section}

{initial_check}

{response_rules}

{citation_guidelines}

{language_rules}
"""

def generate_ai_response_unified(prompt, document_content, is_rag=False, chat_data=None):
    """
    Unified AI response generation for both RAG and traditional methods
    
    Args:
        prompt: User's question
        document_content: Either full document text or retrieved chunks text
        is_rag: Boolean indicating if this is RAG method
        chat_data: Chat data (only needed for RAG to show chunk info)
    
    Returns:
        Generated response text
    """
    # Check if document content is empty
    if not document_content or not document_content.strip():
        return "I apologize, but I cannot answer your question because the document could not be processed or contains no readable text. Please try uploading a different PDF document."
    
    # Set generation state
    st.session_state.generating = True
    st.session_state.stop_generation = False
    
    # Create unified system prompt
    system_prompt = create_system_prompt(document_content, is_rag)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    # State tracking for parsing reasoning
    full_response = ""
    reasoning_content = ""
    answer_content = ""
    in_reasoning = False
    reasoning_started = False
    generation_stopped = False
    
    # Create containers for dynamic updates
    reasoning_placeholder = st.empty()
    answer_placeholder = st.empty()
    
    # Create a stop button container that updates during generation
    stop_container = st.container()
    
    try:
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
        
        # Show initial stop button
        with stop_container:
            stop_button_placeholder = st.empty()
            with stop_button_placeholder.container():
                # Simple, clean stop button aligned to the right
                col1, col2 = st.columns([10, 1])
                with col2:
                    if st.button("⏹", key=f"stop_gen_{hash(prompt)}", help="Stop generation", type="secondary"):
                        st.session_state.stop_generation = True
                        generation_stopped = True
        
        # Handle streaming response with reasoning support
        chunk_count = 0
        for chunk in chat_stream:
            chunk_count += 1
            
            # Periodically check if user wants to stop (every few chunks)
            if chunk_count % 5 == 0 and st.session_state.get('stop_generation', False):
                generation_stopped = True
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
        st.error(f"Error during streaming: {e}")
        return ""
    finally:
        # Reset generation state
        st.session_state.generating = False
        st.session_state.stop_generation = False

def show_citations(response, chat, user_question=""):
    """Show citation-based references"""
    if chat.get("document_content"):
        try:
            pdf_processor = EnhancedPDFProcessor(chat["document_content"])
            pdf_processor.display_citation_based_references(
                response, chat["document_text"], user_question
            )
        except Exception as e:
            st.warning(f"Could not show citations: {e}")

def main():
    """Main application"""
    chat_manager = ChatManager()
    
    # Show connection info
    with st.sidebar:
        if in_docker:
            st.caption(f"🐳 Docker → {ollama_base_url}")
        else:
            st.caption(f"💻 Direct → localhost:11434")
    
    # Model selection
    available_models = ModelManager.get_available_models()
    if available_models:
        previous_model = st.session_state.selected_model
        st.session_state.selected_model = st.selectbox(
            "Choose an Ollama model:",
            available_models,
            index=0 if not st.session_state.selected_model else 
                  (available_models.index(st.session_state.selected_model) 
                   if st.session_state.selected_model in available_models else 0),
            key="model_selector"
        )
        
        # Force rerun if model changed to ensure context check updates immediately
        if previous_model != st.session_state.selected_model and previous_model is not None:
            # Reinitialize RAG system with new model
            if st.session_state.use_rag and "rag_system" in st.session_state:
                logger.info(f"Model changed from {previous_model} to {st.session_state.selected_model}, reinitializing RAG system")
                del st.session_state["rag_system"]
                chat_manager.init_rag_system()
            st.rerun()
        
        # Only show context warnings in main area for serious issues (>80% usage)
        current_chat = chat_manager.get_current_chat()
        document_text = current_chat.get("document_text", "")
        
        if st.session_state.selected_model and document_text and document_text.strip():
            fits, context_info, error = ContextChecker.check_document_fits_context(
                document_text, st.session_state.selected_model
            )
            
            if error:
                st.markdown("---")
                st.info(f"ℹ️ {error}")
                st.caption("Context checking requires model configuration that includes context window size.")
            elif context_info:
                usage_percent = context_info['usage_percent']
                
                # Only show warnings for serious issues (>80% usage)
                if usage_percent > 100:
                    st.markdown("---")
                    st.error(f"⚠️ **Document too large** - Uses {usage_percent:.0f}% of context window")
                    excess_tokens = context_info['total_estimated_tokens'] - context_info['context_length']
                    st.caption(f"Document exceeds limit by ~{excess_tokens:,} tokens")
                elif usage_percent > 80:
                    st.markdown("---")
                    st.warning(f"⚠️ **High context usage** - {usage_percent:.0f}% of {context_info['context_length']:,} tokens")
                    st.caption(f"~{context_info['available_tokens']:,} tokens remaining for conversation")
    else:
        st.error("No Ollama models found. Please ensure Ollama is running.")
        return
    
    # Render sidebar
    render_sidebar(chat_manager)
    
    # Main content
    chat = chat_manager.get_current_chat()
    document_text = chat.get("document_text", "")
    
    if not document_text or not document_text.strip():
        render_document_upload(chat_manager)
    else:
        render_chat_interface(chat_manager)

if __name__ == "__main__":
    main() 