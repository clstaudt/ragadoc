"""
Document Upload UI Components

Handles the document upload interface including PDF processing and RAG integration.
"""

import os
import tempfile
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

                    backend_type = st.session_state.get("rag_backend_type", "vector")

                    # Record which backend was used for this chat
                    current_chat.rag_backend = backend_type
                    
                    if backend_type == "vector":
                        # ── Vector RAG processing ─────────────────────
                        # Check if RAG system needs re-initialization due to changed settings
                        needs_reinit = False
                        if st.session_state.rag_system:
                            current_config = st.session_state.rag_config
                            rag_sys = st.session_state.rag_system
                            
                            # Check chunk settings (only for vector backend)
                            chunk_size_changed = getattr(rag_sys, 'chunk_size', None) != current_config.get("chunk_size")
                            chunk_overlap_changed = getattr(rag_sys, 'chunk_overlap', None) != current_config.get("chunk_overlap")
                            
                            # Check embedding model (account for :latest suffix)
                            config_embed = current_config.get("embedding_model", "")
                            rag_embed = getattr(rag_sys, 'embedding_model', "")
                            embedding_changed = (rag_embed != config_embed and 
                                                rag_embed != f"{config_embed}:latest")
                            
                            if chunk_size_changed or chunk_overlap_changed or embedding_changed:
                                needs_reinit = True
                                logger.info(f"RAG system needs reinit due to changed settings")
                        
                        # Reinitialize RAG system if settings changed or no system exists
                        if needs_reinit or not st.session_state.rag_system:
                            with st.spinner("Applying RAG settings..."):
                                init_rag_system()
                                logger.info("RAG system reinitialized with current settings")
                        
                        # Process with vector RAG system
                        if st.session_state.rag_system:
                            try:
                                with st.spinner("Processing document with RAG system..."):
                                    document_id = str(uuid.uuid4()).replace('-', '')[:16]
                                    
                                    rag_stats = st.session_state.rag_system.process_document(
                                        extracted_text, document_id
                                    )
                                    
                                    st.session_state.chat_manager.update_rag_processing(
                                        rag_stats, current_chat.id
                                    )
                                    
                                    st.success("✅ Document indexed with vector RAG!")
                                    st.info(f"Created {rag_stats['total_chunks']} chunks for semantic search")
                                    
                            except Exception as e:
                                logger.error(f"RAG processing failed: {e}")
                                st.error(f"❌ **RAG processing failed**: {str(e)}")
                                st.error("⚠️ Please try again or check your RAG system configuration.")
                                return
                        else:
                            st.error("❌ **RAG system not available**")
                            st.error("Please check the RAG Settings in the sidebar and retry initialization.")
                            return

                    elif backend_type == "pageindex":
                        # ── PageIndex RAG processing ──────────────────
                        if not st.session_state.rag_system:
                            with st.spinner("Initializing PageIndex RAG system..."):
                                init_rag_system()

                        if st.session_state.rag_system:
                            tmp_path = None
                            try:
                                # PageIndex needs the PDF file path
                                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                                    tmp.write(pdf_bytes)
                                    tmp_path = tmp.name

                                with st.spinner("Building document tree with PageIndex (this may take several minutes)..."):
                                    document_id = str(uuid.uuid4()).replace('-', '')[:16]
                                    
                                    rag_stats = st.session_state.rag_system.process_document(
                                        extracted_text, document_id, pdf_path=tmp_path
                                    )
                                    
                                    st.session_state.chat_manager.update_rag_processing(
                                        rag_stats, current_chat.id
                                    )
                                    
                                    st.success("✅ Document tree built with PageIndex!")
                                    st.info(f"Generated tree with {rag_stats['total_nodes']} nodes for reasoning-based retrieval")

                            except Exception as e:
                                logger.error(f"PageIndex processing failed: {e}")
                                st.error(f"❌ **PageIndex processing failed**: {str(e)}")
                                st.error("⚠️ Please try again. Ensure your chat model supports structured JSON output.")
                                return
                            finally:
                                if tmp_path and os.path.exists(tmp_path):
                                    os.unlink(tmp_path)
                        else:
                            st.error("❌ **PageIndex RAG system not available**")
                            st.error("Ensure the `pageindex` package is installed: `pip install pageindex`")
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