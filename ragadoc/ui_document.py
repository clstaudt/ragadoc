"""
Document Upload UI Components

Handles the document upload interface including PDF processing and RAG integration.
"""

import streamlit as st
import uuid
from loguru import logger

from .enhanced_pdf_processor import EnhancedPDFProcessor
from .ui_session import init_rag_system


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
                    
                    # Process with RAG system - now required
                    if st.session_state.rag_system and getattr(st.session_state.rag_system, 'index', None) is not None:
                        # RAG system is ready, process document
                        try:
                            with st.spinner("Processing document with RAG system..."):
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