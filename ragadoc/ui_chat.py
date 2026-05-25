"""
Chat Interface UI Components

Handles the main chat interface including message display, response generation,
and citation display.
"""

import streamlit as st
import ollama
from loguru import logger
from streamlit_pdf_viewer import pdf_viewer

from .enhanced_pdf_processor import EnhancedPDFProcessor
from .llm_interface import PromptBuilder
from .ui_session import get_current_ollama_url


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


def _retrieve_context_vector(prompt):
    """Retrieve context using vector RAG (chunks + similarity)."""
    current_threshold = st.session_state.rag_config.get("similarity_threshold", 0.7)
    current_top_k = st.session_state.rag_config.get("top_k", 10)

    retrieval_info = st.session_state.rag_system.get_retrieval_info(
        prompt,
        similarity_threshold=current_threshold,
        top_k=current_top_k
    )

    # Get all retrieved chunks (before filtering) for fallback
    all_nodes = []
    try:
        from llama_index.core.retrievers import VectorIndexRetriever
        retriever = VectorIndexRetriever(
            index=st.session_state.rag_system.index,
            similarity_top_k=current_top_k
        )
        all_nodes = retriever.retrieve(prompt)
    except Exception as e:
        logger.warning(f"Could not get unfiltered chunks: {e}")

    # Display retrieved chunks information IMMEDIATELY (before generation)
    if retrieval_info:
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
        return "\n\n".join([chunk['text'] for chunk in retrieval_info])
    elif all_nodes:
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
        return "\n\n".join([node.text for node in all_nodes[:3]])
    else:
        return None


def _retrieve_context_pageindex(prompt):
    """Retrieve context using PageIndex tree search (reasoning-based)."""
    with st.spinner("Searching document tree..."):
        retrieval_info = st.session_state.rag_system.get_retrieval_info(prompt)

    if not retrieval_info:
        return None

    # Show LLM reasoning about which sections are relevant
    reasoning = retrieval_info[0].get("metadata", {}).get("reasoning", "")
    if reasoning:
        with st.expander("🤔 Retrieval Reasoning", expanded=False):
            st.markdown(reasoning)

    # Show retrieved document sections (not "chunks", no similarity scores)
    with st.expander(f"🌳 Retrieved {len(retrieval_info)} document sections", expanded=False):
        for section in retrieval_info:
            meta = section.get("metadata", {})
            title = meta.get("title", "Untitled")
            start = meta.get("start_page", "?")
            end = meta.get("end_page", "?")
            st.markdown(f"**{title}** (Pages {start}–{end})")
            st.text_area(
                f"{title} ({section['length']} characters):",
                value=section["text"],
                height=200,
                disabled=True,
                key=f"section_{section['chunk_id']}_{hash(prompt)}"
            )
            st.markdown("---")

    return "\n\n".join([s["text"] for s in retrieval_info])


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
        
        backend_type = getattr(st.session_state.rag_system, 'backend_type', 'vector')
        logger.info(f"Using {backend_type} RAG system for response generation")
        
        # ── Retrieval (backend-specific) ─────────────────────────
        try:
            if backend_type == "pageindex":
                context_text = _retrieve_context_pageindex(prompt)
            else:
                context_text = _retrieve_context_vector(prompt)

            if not context_text:
                if backend_type == "pageindex":
                    st.warning("No relevant sections found. Try rephrasing your question.")
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
        loading_placeholder = st.empty()
        
        # Prepare for aligned loading message and stop button
        stop_button_placeholder = st.empty()
        
        # Generate response with streaming
        full_response = ""
        reasoning_content = ""
        answer_content = ""
        reasoning_started = False
        in_reasoning = False
        generation_stopped = False
        first_chunk_received = False
        
        # Use direct ollama streaming with RAG context
        system_prompt = PromptBuilder.create_system_prompt(context_text, is_rag=True)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # Create chat stream using selected Ollama instance
        client = ollama.Client(host=get_current_ollama_url())
        chat_stream = client.chat(
            model=st.session_state.selected_model,
            messages=messages,
            stream=True
        )
        
        # Show loading message and stop button aligned
        with loading_placeholder.container():
            col1, col2 = st.columns([10, 1])
            with col1:
                st.write("⏳ Processing...")
            with col2:
                if st.button("⏹", key=f"stop_gen_{hash(prompt)}", help="Stop generation"):
                    st.session_state.stop_generation = True
        
        # Handle streaming response with reasoning support
        chunk_count = 0
        has_content = False
        for chunk in chat_stream:
            chunk_count += 1
            
            # Check if this chunk has actual content
            if chunk['message']['content']:
                has_content = True
            
            # Clear loading spinner on first chunk with actual content
            if has_content and not first_chunk_received:
                first_chunk_received = True
                loading_placeholder.empty()  # Clear the spinner
            
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
            # Clear loading spinner if still showing when stopped
            if not first_chunk_received:
                loading_placeholder.empty()
                
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
            
            # Show stopped message
            with stop_container:
                st.info("🛑 Generation stopped by user")
        # Clear the entire loading container (which includes the stop button)
        # This happens for both stopped and completed generation
        
        # Method information removed - not needed for end users
        
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
        # Clear loading spinner if still showing in case of error
        if not first_chunk_received:
            loading_placeholder.empty()
        st.error(f"Error generating response: {e}")
        return ""
    finally:
        # Reset generation state
        st.session_state.generating = False
        st.session_state.stop_generation = False
        # Ensure loading spinner is cleared
        loading_placeholder.empty()


def render_chat_interface():
    """Render the main chat interface"""
    current_chat = st.session_state.chat_manager.get_current_chat()
    if not current_chat:
        return
    
    # Show current document info
    if current_chat.document_name:
        with st.expander(f"📄 {current_chat.document_name}", expanded=False):
            # Show RAG processing statistics (backend-aware)
            if current_chat.rag_processed:
                rag_stats = current_chat.rag_stats or {}
                chat_backend = current_chat.rag_backend
                col1, col2 = st.columns(2)
                if chat_backend == "pageindex":
                    with col1:
                        st.metric("Tree Nodes", rag_stats.get("total_nodes", 0))
                    with col2:
                        st.metric("Backend", "PageIndex")
                else:
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
                        annotations=[],  # Explicitly provide empty annotations list
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