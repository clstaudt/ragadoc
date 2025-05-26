import streamlit as st
import ollama
import uuid
import os
from datetime import datetime
from streamlit_pdf_viewer import pdf_viewer
import pdfplumber
import io
from ragnarok import EnhancedPDFProcessor
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
    page_title="Ollama PDF Chat", 
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("Ollama PDF Chat")

class ChatManager:
    """Simplified chat management"""
    
    def __init__(self):
        self.init_session_state()
    
    def init_session_state(self):
        """Initialize all session state variables"""
        defaults = {
            "chats": {},
            "current_chat_id": None,
            "selected_model": None
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
        
        # Create first chat if none exist
        if not st.session_state.chats:
            self.create_new_chat()
    
    def create_new_chat(self):
        """Create a new chat session"""
        chat_id = str(uuid.uuid4())
        st.session_state.chats[chat_id] = {
            "messages": [],
            "created_at": datetime.now(),
            "title": "New Chat",
            "document_name": None,
            "document_content": None,
            "document_text": ""
        }
        st.session_state.current_chat_id = chat_id
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
                    self.create_new_chat()

class PDFProcessor:
    """Simplified PDF processing"""
    
    @staticmethod
    def extract_text(pdf_file) -> str:
        """Extract text from PDF using pdfplumber only"""
        try:
            pdf_bytes = pdf_file.getvalue()
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text
        except Exception as e:
            st.error(f"Error extracting text from PDF: {e}")
            return ""

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
                
            if 'models' in models_info:
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

def render_sidebar(chat_manager):
    """Render the sidebar with chat history"""
    with st.sidebar:
        st.header("Chat History")
        
        # New chat button
        if st.button("New Chat", use_container_width=True, type="primary"):
            chat_manager.create_new_chat()
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
            chat_manager.create_new_chat()
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
                extracted_text = PDFProcessor.extract_text(uploaded_file)
            
            if extracted_text:
                # Update current chat with document
                chat = chat_manager.get_current_chat()
                chat.update({
                    "document_name": uploaded_file.name,
                    "document_content": uploaded_file.getvalue(),
                    "document_text": extracted_text,
                    "title": f"Doc: {uploaded_file.name}"
                })
                
                st.success(f"Document '{uploaded_file.name}' processed successfully!")
                st.info(f"Extracted {len(extracted_text.split()):,} words")
                st.rerun()
            else:
                st.error("Could not extract text from PDF")
        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.info("💡 If this file was uploaded before, try clicking '🗑️ Clear Upload' and upload again")

def render_chat_interface(chat_manager):
    """Render the main chat interface"""
    chat = chat_manager.get_current_chat()
    
    # Show current document info
    if chat.get("document_name"):
        with st.expander("📄 Current Document", expanded=False):
            st.write(f"**Document:** {chat['document_name']}")
            
            # Show PDF viewer
            if chat.get("document_content"):
                pdf_viewer(
                    input=chat["document_content"],
                    width="100%",
                    height=600,
                    render_text=True,
                    key=f"pdf_viewer_{st.session_state.current_chat_id}"
                )
    
    # Display chat messages
    for message in chat.get("messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if chat.get("document_text"):
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
                    response = generate_ai_response(prompt, chat["document_text"])
                    # Note: Display is handled within generate_ai_response for reasoning support
                    chat_manager.add_message("assistant", response)
                    
                    # Show citations
                    show_citations(response, chat)
                        
                except Exception as e:
                    st.error(f"Error generating response: {e}")

def generate_ai_response(prompt, document_text):
    """Generate AI response using Ollama with reasoning support"""
    system_prompt = f"""Answer questions based on this document:

{document_text}

CRITICAL CITATION FORMAT:
You MUST use citations in this EXACT format for text highlighting to work:

1. Write your answer normally with citation numbers like [1], [2]
2. At the end, list each citation on a new line starting with the number in brackets, followed by a space and the exact quote in double quotes

REQUIRED FORMAT:
[1] "exact quote from document"
[2] "another exact quote from document"

EXAMPLE:
Question: Does he have experience in the medical field?
Answer: Yes, Christian Staudt has experience in the medical field. [1] [2]

[1] "project development for AI applications: medical data mining & AI, AI for renewable energy control"
[2] "developing a prototype for data-driven measurement of global marketing campaign performance across channels"

RULES:
- Citations MUST start at the beginning of a line
- Citations MUST use the format [number] "quote" 
- Use exact quotes from the document, not paraphrases
- Each citation on its own line
- Do NOT use colons, "Exact quote:", or other text before the quote"""
    
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
    
    # Create containers for dynamic updates
    reasoning_placeholder = st.empty()
    answer_placeholder = st.empty()
    
    try:
        if in_docker:
            # Docker - use explicit client configuration
            client = ollama.Client(host=ollama_base_url)
            chat_stream = client.chat(
                model=st.session_state.selected_model,
                messages=messages,
                stream=True
            )
        else:
            # Direct execution - use default ollama client
            chat_stream = ollama.chat(
                model=st.session_state.selected_model,
                messages=messages,
                stream=True
            )
            
        for chunk in chat_stream:
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
                        
                        # Show completed reasoning in expandable container
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
        
        # Return the final answer (without reasoning tags) for storage
        if reasoning_started:
            return answer_content
        else:
            return full_response
            
    except Exception as e:
        st.error(f"Error during streaming: {e}")
        return ""

def show_citations(response, chat):
    """Show citation-based references"""
    if chat.get("document_content"):
        try:
            pdf_processor = EnhancedPDFProcessor(chat["document_content"])
            pdf_processor.display_citation_based_references(
                response, chat["document_text"]
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
        st.session_state.selected_model = st.selectbox(
            "Choose an Ollama model:",
            available_models,
            index=0 if not st.session_state.selected_model else 
                  (available_models.index(st.session_state.selected_model) 
                   if st.session_state.selected_model in available_models else 0)
        )
    else:
        st.error("No Ollama models found. Please ensure Ollama is running.")
        return
    
    # Render sidebar
    render_sidebar(chat_manager)
    
    # Main content
    chat = chat_manager.get_current_chat()
    if not chat.get("document_text"):
        render_document_upload(chat_manager)
    else:
        render_chat_interface(chat_manager)

if __name__ == "__main__":
    main() 