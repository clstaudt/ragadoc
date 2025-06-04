"""
UI Configuration and Environment Detection

Handles environment detection and configuration setup for the Streamlit UI.
"""

import os
import streamlit as st


def is_running_in_docker():
    """Check if we're running inside a Docker container"""
    return (
        os.path.exists('/.dockerenv') or 
        os.environ.get('STREAMLIT_SERVER_ADDRESS') == '0.0.0.0'
    )


def get_ollama_base_url():
    """Get the appropriate Ollama base URL based on environment"""
    in_docker = is_running_in_docker()
    if in_docker:
        return os.environ.get('OLLAMA_BASE_URL', 'http://host.docker.internal:11434')
    else:
        return "http://localhost:11434"


def setup_streamlit_config():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="ragadoc - AI-assisted Document Q&A", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("ragadoc - AI-assisted Document Q&A") 