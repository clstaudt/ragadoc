#!/usr/bin/env python3
"""
Demo script showing PyMuPDF highlighting capabilities
Run this to test highlighting functionality before integrating with the main app
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import fitz  # PyMuPDF
from ragnarok import EnhancedPDFProcessor
import io

st.set_page_config(page_title="PyMuPDF Highlighting Demo", layout="wide")
st.title("🎯 PyMuPDF Highlighting Demo")

st.markdown("""
This demo shows the PyMuPDF highlighting capabilities described in your document.
Upload a PDF and try different highlighting scenarios.

**Note**: PDF processors are created fresh each time to avoid serialization issues with Streamlit caching.
""")

# Demo modes
demo_mode = st.radio(
    "Choose Demo Mode:",
    ["Manual Text Search", "Simulated AI Response", "Custom Highlight Terms"],
    horizontal=True
)

uploaded_file = st.file_uploader("Upload a PDF to test highlighting", type=['pdf'])

if uploaded_file:
    # Create processor
    pdf_bytes = uploaded_file.getvalue()
    processor = EnhancedPDFProcessor(pdf_bytes)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📄 Original Document")
        # Show original PDF
        processor.display_highlighted_pdf_in_streamlit()
    
    with col2:
        st.subheader("🎯 With Highlights")
        
        if demo_mode == "Manual Text Search":
            # Manual search terms
            search_terms = st.text_area(
                "Enter terms to highlight (one per line):",
                placeholder="artificial intelligence\nmachine learning\nneural networks"
            ).strip().split('\n') if st.text_area(
                "Enter terms to highlight (one per line):",
                placeholder="artificial intelligence\nmachine learning\nneural networks"
            ).strip() else []
            
            if search_terms:
                processor.display_highlighted_pdf_in_streamlit(search_terms)
            else:
                st.info("Enter some search terms to see highlighting in action!")
                
        elif demo_mode == "Simulated AI Response":
            # Simulate AI response with quotes
            sample_responses = {
                "Technical Paper": '''Based on the document, "machine learning algorithms show significant improvement" when applied to this problem. The paper specifically mentions that "neural networks achieved 95% accuracy" in the experimental results.''',
                
                "Research Article": '''The study concludes that "climate change has accelerated" over the past decade. Furthermore, "global temperatures have risen by 1.5 degrees" according to the data presented.''',
                
                "Business Report": '''The quarterly results indicate "revenue increased by 23%" compared to last year. The report also states that "customer satisfaction improved significantly" due to new initiatives.'''
            }
            
            response_type = st.selectbox("Choose simulated AI response:", list(sample_responses.keys()))
            simulated_response = sample_responses[response_type]
            
            st.text_area("Simulated AI Response:", simulated_response, height=100, disabled=True)
            
            # Extract and highlight
            document_text = processor.extract_full_text()
            highlight_terms = processor.create_ai_response_highlights(simulated_response, document_text)
            
            if highlight_terms:
                st.success(f"Found {len(highlight_terms)} quotations to highlight!")
                processor.display_highlighted_pdf_in_streamlit(highlight_terms)
                
                # Show highlighted terms without nested expander
                st.markdown("### 📝 Highlighted Terms")
                for i, term in enumerate(highlight_terms):
                    st.markdown(f"**{i+1}.** \"{term}\"")
            else:
                st.warning("No matching quoted text found in the document.")
                processor.display_highlighted_pdf_in_streamlit()
                
        elif demo_mode == "Custom Highlight Terms":
            # Custom terms with color options
            custom_terms = st.multiselect(
                "Select terms to highlight:",
                ["the", "and", "of", "to", "in", "for", "is", "on", "that", "by"],
                default=["the", "and"]
            )
            
            color_option = st.selectbox(
                "Highlight Color:",
                ["Yellow", "Green", "Blue", "Red", "Orange"],
                index=0
            )
            
            color_map = {
                "Yellow": (1, 1, 0),
                "Green": (0, 1, 0),
                "Blue": (0, 0, 1),
                "Red": (1, 0, 0),
                "Orange": (1, 0.5, 0)
            }
            
            if custom_terms:
                # Create custom highlighted PDF
                highlighted_pdf_bytes = processor.search_and_highlight_text(
                    custom_terms, 
                    highlight_color=color_map[color_option]
                )
                
                from streamlit_pdf_viewer import pdf_viewer
                pdf_viewer(
                    input=highlighted_pdf_bytes,
                    width="100%",
                    height=600,
                    render_text=True,
                    key="custom_highlighted_pdf"
                )
                
                st.info(f"Highlighted {len(custom_terms)} terms in {color_option.lower()}")
            else:
                processor.display_highlighted_pdf_in_streamlit()

    # Show document stats
    with st.expander("📊 Document Statistics"):
        full_text = processor.extract_full_text()
        word_count = len(full_text.split())
        char_count = len(full_text)
        page_count = processor.doc.page_count
        
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Pages", page_count)
        col_b.metric("Words", f"{word_count:,}")
        col_c.metric("Characters", f"{char_count:,}")

else:
    st.info("👆 Upload a PDF file to start the highlighting demo!")
    
    # Show features
    st.markdown("""
    ## 🌟 Features Demonstrated
    
    1. **Manual Text Search** - Search for specific terms and highlight them
    2. **AI Response Simulation** - Show how quoted text from AI responses gets highlighted
    3. **Custom Highlighting** - Choose colors and terms for highlighting
    
    ## 🔧 Technical Capabilities
    
    - **Coordinate-based highlighting** using PyMuPDF quads
    - **Multiple highlight colors** for categorization
    - **Text extraction with position data** for precise highlighting
    - **PDF viewer integration** with Streamlit components
    - **Performance optimization** through caching
    
    ## 📚 Integration with Your Chat App
    
    This highlighting system can be integrated into your existing Ollama chatbot to:
    - Automatically highlight text that the AI references
    - Show evidence for AI responses
    - Improve user understanding of document-based answers
    - Create visual connections between chat and document content
    """) 