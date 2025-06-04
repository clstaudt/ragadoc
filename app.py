"""
Refactored Streamlit App for Document Q&A

This is now a much simpler main entry point that uses the UI modules
from the ragnarok package.
"""

import streamlit as st
from ragnarok import (
    setup_streamlit_config,
    init_session_state,
    render_sidebar,
    render_document_upload,
    render_chat_interface
)


def main():
    """Main application function"""
    # Setup Streamlit configuration
    setup_streamlit_config()
    
    # Initialize session state
    init_session_state()
    
    # Render sidebar (which includes model selection and chat history)
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