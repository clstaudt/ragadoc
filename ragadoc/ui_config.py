"""
UI Configuration and Environment Detection

Handles environment detection and configuration setup for the Streamlit UI.
"""

import base64
import os
import streamlit as st
from pathlib import Path


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


def load_darkstreaming_theme():
    """Load the darkstreaming CSS theme"""
    css_path = Path(__file__).parent / "themes" / "darkstreaming.css"
    if css_path.exists():
        with open(css_path) as f:
            return f.read()
    return ""


def apply_darkstreaming_theme():
    """Apply the darkstreaming theme with fallback support"""
    css_content = load_darkstreaming_theme()
    
    if css_content:
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    else:
        # Fallback to basic dark theme if file not found
        apply_basic_dark_theme()


def apply_basic_dark_theme():
    """Minimal fallback dark theme"""
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(180deg, #121212 0%, #0a0a0a 100%);
        color: #ffffff;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #FFB000, #FFC947);
        color: #000;
        border: none;
        border-radius: 25px;
        font-weight: 600;
    }
    
    .stSelectbox > div > div {
        background: #242424;
        border: 1px solid #2a2a2a;
        color: #ffffff;
    }
    
    h1 {
        background: linear-gradient(45deg, #FFB000, #FFC947);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    </style>
    """, unsafe_allow_html=True)


def add_logo_and_title():
    """Add logo and title to the app header"""
    logo_path = Path(__file__).parent.parent / "assets" / "logo.png"
    
    if logo_path.exists():
        # Compact header with logo and title using proper CSS classes for styling
        st.markdown("""
        <div class="logo-header" style="margin-bottom: 1rem;">
            <img src="data:image/png;base64,{}" alt="ragadoc logo" style="width: 70px; height: auto;">
            <h1 style="margin: 0.5rem 0 0 0; font-size: 2rem;">ragadoc - AI Document Assistant</h1>
            <p style="margin: 0.3rem 0 0 0; font-style: italic; color: #b3b3b3; font-size: 0.9rem;">
                Ask questions about your documents - get grounded answers with citations and highlights.
            </p>
        </div>
        """.format(get_base64_logo(logo_path)), unsafe_allow_html=True)
    else:
        # Fallback to text-only title with proper gradient styling
        st.markdown("<h1 style='font-size: 2rem; margin: 0 0 0.5rem 0; text-align: center;'>ragadoc - AI Document Assistant</h1>", unsafe_allow_html=True)
        st.markdown("<p style='margin: 0 0 1rem 0; font-style: italic; color: #b3b3b3; font-size: 0.9rem; text-align: center;'>Ask questions about your documents - get grounded answers with citations and highlights.</p>", unsafe_allow_html=True)


def get_base64_logo(logo_path):
    """Convert logo to base64 for embedding in HTML"""
    try:
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""


def setup_streamlit_config():
    """Configure Streamlit page settings with darkstreaming theme"""
    st.set_page_config(
        page_title="ragadoc - AI-powered Document Assistant", 
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply the darkstreaming theme
    apply_darkstreaming_theme()
    
    # Add logo and title
    add_logo_and_title() 