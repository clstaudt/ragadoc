import streamlit as st
import ollama
import uuid
from datetime import datetime
from streamlit_pdf_viewer import pdf_viewer
import PyPDF2
import pdfplumber
import io

st.set_page_config(page_title="Ollama Chatbot", layout="wide")
st.title("Ollama Chatbot")

# PDF Processing functions
def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from uploaded PDF file"""
    try:
        # Try with pdfplumber first (usually better for text extraction)
        pdf_bytes = pdf_file.getvalue()
        
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        if text.strip():
            return text
        
        # Fallback to PyPDF2 if pdfplumber doesn't work
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

# Function to get available Ollama models
def get_ollama_models():
    try:
        models_info = ollama.list()
        model_names = []
        if 'models' in models_info and isinstance(models_info['models'], list):
            for model_details in models_info['models']:
                # Handle both dictionary and object formats
                if hasattr(model_details, 'model'):
                    # Object with model attribute
                    model_names.append(model_details.model)
                elif isinstance(model_details, dict) and 'model' in model_details:
                    # Dictionary with model key
                    model_names.append(model_details['model'])
                else:
                    st.warning(f"Found an entry without a 'model' key/attribute: {model_details}")
        else:
            st.warning("Ollama list response did not contain a 'models' list or it was malformed.")
        return model_names
    except ollama.ResponseError as e:
        st.error(f"Ollama API Error: {e.error} (Status code: {e.status_code}). Make sure Ollama is running and accessible.")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching Ollama models: {e}. Make sure Ollama is running.")
        return []

# Initialize chat system
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

# Helper functions for chat management
def create_new_chat():
    """Create a new chat session"""
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {
        "messages": [],
        "created_at": datetime.now(),
        "title": "New Document Chat",
        "document_uploaded": False,
        "document_name": None,
        "document_content": None,
        "document_text": ""
    }
    st.session_state.current_chat_id = chat_id
    return chat_id

def get_current_messages():
    """Get messages for the current chat"""
    if st.session_state.current_chat_id and st.session_state.current_chat_id in st.session_state.chats:
        return st.session_state.chats[st.session_state.current_chat_id]["messages"]
    return []

def add_message_to_current_chat(role, content):
    """Add a message to the current chat"""
    if st.session_state.current_chat_id and st.session_state.current_chat_id in st.session_state.chats:
        st.session_state.chats[st.session_state.current_chat_id]["messages"].append({
            "role": role, 
            "content": content
        })
        # Update chat title based on first user message
        if role == "user" and st.session_state.chats[st.session_state.current_chat_id]["title"] == "New Document Chat":
            title = content[:50] + "..." if len(content) > 50 else content
            st.session_state.chats[st.session_state.current_chat_id]["title"] = title

def delete_chat(chat_id):
    """Delete a chat session"""
    if chat_id in st.session_state.chats:
        del st.session_state.chats[chat_id]
        if st.session_state.current_chat_id == chat_id:
            # Switch to another chat or create new one
            if st.session_state.chats:
                st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]
            else:
                st.session_state.current_chat_id = None

def get_chat_preview(chat_data):
    """Get a preview of the last message in the chat"""
    messages = chat_data.get("messages", [])
    document_name = chat_data.get("document_name", None)
    
    if not messages:
        if document_name:
            return f"📄 {document_name}"
        return "No document uploaded yet"
    
    last_message = messages[-1]
    content = last_message["content"]
    
    # Truncate long messages
    if len(content) > 60:
        content = content[:60] + "..."
    
    # Add role prefix
    if last_message["role"] == "user":
        return f"You: {content}"
    else:
        return content

def format_chat_time(created_at):
    """Format the chat creation time"""
    now = datetime.now()
    diff = now - created_at
    
    if diff.days > 0:
        return created_at.strftime("%b %d")
    elif diff.seconds > 3600:
        return created_at.strftime("%H:%M")
    else:
        minutes = diff.seconds // 60
        if minutes == 0:
            return "now"
        return f"{minutes}m ago"

# Initialize click tracking
if "chat_clicked" not in st.session_state:
    st.session_state.chat_clicked = None
if "delete_clicked" not in st.session_state:
    st.session_state.delete_clicked = None

# Create first chat if none exist
if not st.session_state.chats:
    create_new_chat()

# Sidebar for chat management
with st.sidebar:
    st.header("💬 Chat History")
    
    # New Document Chat button
    if st.button("📄 New Document Chat", use_container_width=True, type="primary"):
        create_new_chat()
        st.rerun()
    
    st.markdown("---")
    
    # Display chat history
    if st.session_state.chats:
        # Sort chats by creation time (newest first)
        sorted_chats = sorted(
            st.session_state.chats.items(), 
            key=lambda x: x[1]["created_at"], 
            reverse=True
        )
        
        for chat_id, chat_data in sorted_chats:
            is_current = chat_id == st.session_state.current_chat_id
            preview = get_chat_preview(chat_data)
            time_str = format_chat_time(chat_data["created_at"])
            
            # Create a clean container for each chat
            with st.container():
                col1, col2 = st.columns([5, 1])
                
                with col1:
                    # Simple, clean button with all info
                    button_type = "primary" if is_current else "secondary"
                    
                    # Create a clean button label
                    button_label = f"💬 **{chat_data['title']}**\n{preview}\n*{time_str}*"
                    
                    if st.button(
                        button_label,
                        key=f"chat-{chat_id}",
                        use_container_width=True,
                        type=button_type
                    ):
                        st.session_state.current_chat_id = chat_id
                        st.rerun()
                
                with col2:
                    if st.button("🗑️", key=f"del-{chat_id}", help="Delete", type="secondary"):
                        delete_chat(chat_id)
                        st.rerun()
                
                # Add subtle spacing
                st.write("")
    else:
        st.info("💡 No chats yet. Click 'New Document Chat' to start your first conversation!")

# Main content area
# Get available models
available_models = get_ollama_models()

if not available_models:
    st.warning("No usable Ollama models found. Please ensure Ollama is running, models are installed, and they are correctly configured.")
    # We still allow the app to run so the user can see error messages.
    # Consider st.stop() if you want to halt execution completely.

# Model selection dropdown
# Only show dropdown if models are available
if available_models:
    st.session_state.selected_model = st.selectbox(
        "Choose an Ollama model:",
        available_models,
        index=available_models.index(st.session_state.selected_model) if st.session_state.selected_model and st.session_state.selected_model in available_models else 0
    )
else:
    st.session_state.selected_model = None # Ensure no model is selected if none are available

# Check if current chat has a document uploaded
current_chat = st.session_state.chats.get(st.session_state.current_chat_id, {})
document_uploaded = current_chat.get("document_uploaded", False)

if not document_uploaded and st.session_state.current_chat_id:
    # Show document upload interface
    st.markdown("## 📄 Upload Your Document")
    st.markdown("To start chatting, please upload a PDF document first.")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Drag and drop a PDF file here or click to browse",
        key=f"uploader_{st.session_state.current_chat_id}"
    )
    
    if uploaded_file is not None:
        # Extract text from the PDF
        with st.spinner("Processing PDF and extracting text..."):
            extracted_text = extract_text_from_pdf(uploaded_file)
        
        if extracted_text:
            # Update chat with document info
            st.session_state.chats[st.session_state.current_chat_id]["document_uploaded"] = True
            st.session_state.chats[st.session_state.current_chat_id]["document_name"] = uploaded_file.name
            st.session_state.chats[st.session_state.current_chat_id]["document_content"] = uploaded_file.getvalue()
            st.session_state.chats[st.session_state.current_chat_id]["document_text"] = extracted_text
            st.session_state.chats[st.session_state.current_chat_id]["title"] = f"📄 {uploaded_file.name}"
            
            st.success(f"✅ Document '{uploaded_file.name}' uploaded and processed successfully!")
            st.info("💬 You can now start asking questions about your document below.")
            
            # Show extracted text info
            word_count = len(extracted_text.split())
            st.info(f"📊 Extracted {word_count:,} words from the document")
            st.rerun()
        else:
            st.error("❌ Could not extract text from the PDF. Please ensure it's a text-based PDF document.")

else:
    # Show current document info if uploaded
    if document_uploaded:
        with st.expander("📄 Current Document", expanded=False):
            # Shorten long file names for display
            full_name = current_chat.get('document_name', 'Unknown')
            display_name = full_name
            if len(full_name) > 40:
                display_name = full_name[:37] + "..."
            
            st.write(f"**Document:** {display_name}")
            
            # Display PDF viewer using streamlit-pdf-viewer with maximum size
            document_content = current_chat.get('document_content')
            if document_content:
                pdf_viewer(
                    input=document_content,
                    width="100%",
                    height=800,
                    render_text=True,
                    key=f"pdf_viewer_{st.session_state.current_chat_id}"
                )
    
    # Display chat messages from history on app rerun
    current_messages = get_current_messages()
    for message in current_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# React to user input - only show if document is uploaded
if document_uploaded:
    if prompt := st.chat_input("Ask a question about your document..."):
            if not st.session_state.selected_model:
                st.warning("Please select a model from the dropdown above. If no models are listed, check Ollama.")
            else:
                # Display user message in chat message container
                with st.chat_message("user"):
                    st.markdown(prompt)
                # Add user message to chat history
                add_message_to_current_chat("user", prompt)

                try:
                    # Display assistant response with streaming
                    with st.chat_message("assistant"):
                        # Get document context
                        current_chat = st.session_state.chats.get(st.session_state.current_chat_id, {})
                        document_text = current_chat.get('document_text', '')
                        
                        # Prepare messages for Ollama API (list of dicts)
                        current_conversation = []
                        
                        # Add document context as system message if available
                        if document_text:
                            system_prompt = f"""You are a helpful assistant that answers questions about documents. You have been provided with the following document content:

--- DOCUMENT CONTENT ---
{document_text}
--- END DOCUMENT CONTENT ---

Please answer questions based on this document content. If a question cannot be answered from the document, please say so clearly."""
                            
                            current_conversation.append({
                                'role': 'system', 
                                'content': system_prompt
                            })
                        
                        # Add conversation history
                        for msg in get_current_messages():
                            current_conversation.append({'role': msg['role'], 'content': msg['content']})
                        
                        def generate_response():
                            """Generator function for streaming response"""
                            for chunk in ollama.chat(
                                model=st.session_state.selected_model,
                                messages=current_conversation,
                                stream=True
                            ):
                                if chunk['message']['content']:
                                    yield chunk['message']['content']
                        
                        # Stream the response
                        assistant_response = st.write_stream(generate_response())
                    
                    # Add assistant response to chat history
                    add_message_to_current_chat("assistant", assistant_response)
                except ollama.ResponseError as e:
                    st.error(f"Ollama API Error during chat: {e.error} (Status code: {e.status_code})")
                except Exception as e:
                    st.error(f"Error communicating with Ollama during chat: {e}")
else:
    # Show a message when no document is uploaded
    if st.session_state.current_chat_id:
        st.info("📄 Please upload a PDF document above to start chatting.") 