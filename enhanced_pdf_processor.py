import streamlit as st
import fitz  # PyMuPDF
import io
from PIL import Image
import base64
from typing import List, Tuple, Dict, Optional
import re

class EnhancedPDFProcessor:
    """Enhanced PDF processor with highlighting capabilities using PyMuPDF"""
    
    def __init__(self, pdf_bytes: bytes):
        self.pdf_bytes = pdf_bytes
        self.doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        self.highlighted_doc = None
        
    def extract_text_with_positions(self) -> Dict[int, List[Dict]]:
        """Extract text with position information for each page"""
        text_data = {}
        
        for page_num in range(self.doc.page_count):
            page = self.doc[page_num]
            
            # Extract text blocks with positions
            blocks = page.get_text("dict")
            
            page_text_info = []
            for block in blocks["blocks"]:
                if "lines" in block:  # Text block
                    for line in block["lines"]:
                        for span in line["spans"]:
                            page_text_info.append({
                                "text": span["text"],
                                "bbox": span["bbox"],  # (x0, y0, x1, y1)
                                "page": page_num
                            })
            
            text_data[page_num] = page_text_info
            
        return text_data
    
    def extract_full_text(self) -> str:
        """Extract full text from PDF (similar to your current function)"""
        full_text = ""
        for page_num in range(self.doc.page_count):
            page = self.doc[page_num]
            full_text += page.get_text() + "\n"
        return full_text
    
    def search_and_highlight_text(self, search_terms: List[str], 
                                 highlight_color: Tuple[float, float, float] = (1, 1, 0)) -> bytes:
        """Search for terms and create highlighted PDF"""
        # Create a copy of the document for highlighting
        self.highlighted_doc = fitz.open(stream=self.pdf_bytes, filetype="pdf")
        
        highlight_info = []
        
        for page_num in range(self.highlighted_doc.page_count):
            page = self.highlighted_doc[page_num]
            
            for term in search_terms:
                # Search for text instances
                text_instances = page.search_for(term, quads=True)
                
                for inst in text_instances:
                    # Add highlight annotation
                    highlight = page.add_highlight_annot(inst)
                    highlight.set_colors(stroke=highlight_color)
                    highlight.update()
                    
                    highlight_info.append({
                        "term": term,
                        "page": page_num,
                        "bbox": inst.rect,
                        "quad": inst
                    })
        
        # Return the highlighted PDF as bytes
        return self.highlighted_doc.tobytes()
    
    def render_page_with_highlights(self, page_num: int, dpi: int = 150) -> bytes:
        """Render a specific page as PNG with highlights visible"""
        if self.highlighted_doc:
            page = self.highlighted_doc[page_num]
        else:
            page = self.doc[page_num]
        
        # Create pixmap with highlights
        pix = page.get_pixmap(dpi=dpi)
        return pix.tobytes("png")
    
    def create_ai_response_highlights(self, ai_response: str, original_text: str) -> List[str]:
        """Extract quoted text from AI response to highlight in PDF"""
        # Comprehensive regex patterns for various quote types including Unicode
        quoted_patterns = [
            r'"([^"]+)"',                    # ASCII double quotes
            r'\u201c([^\u201d]+)\u201d',     # Unicode smart double quotes
            r'\u201e([^\u201c]+)\u201c',     # German-style quotes  
            r'\u00bb([^\u00ab]+)\u00ab',     # French-style guillemets
            r"'([^']+)'",                    # ASCII single quotes
            r'\u2018([^\u2019]+)\u2019',     # Unicode smart single quotes
            r'`([^`]+)`',                    # Backticks
            r'\u201a([^\u2018]+)\u2018',     # German-style single quotes
        ]
        
        highlighted_terms = []
        
        for i, pattern in enumerate(quoted_patterns):
            try:
                matches = re.findall(pattern, ai_response)
                
                for match in matches:
                    # Clean up the match (remove extra whitespace)
                    cleaned_match = match.strip()
                    
                    # Check if the quoted text actually exists in the original document (case insensitive)
                    if len(cleaned_match) > 5:  # Reduced minimum length for testing
                        # Try multiple matching strategies
                        match_found = False
                        
                        # Strategy 1: Exact match
                        if self._fuzzy_text_match(cleaned_match, original_text):
                            highlighted_terms.append(cleaned_match)
                            match_found = True
                        
                        # Strategy 2: Significant word matching for longer quotes
                        if not match_found and len(cleaned_match.split()) >= 3:
                            significant_words = [w for w in cleaned_match.split() if len(w) > 3]
                            if len(significant_words) >= 2:
                                # Try combinations of significant words
                                for num_words in range(min(5, len(significant_words)), 1, -1):
                                    partial_match = ' '.join(significant_words[:num_words])
                                    if self._fuzzy_text_match(partial_match, original_text):
                                        highlighted_terms.append(partial_match)
                                        match_found = True
                                        break
            except Exception as e:
                # Silently skip patterns that cause errors
                continue
        
        return highlighted_terms
    
    def _fuzzy_text_match(self, search_text: str, document_text: str) -> bool:
        """Robust text matching that handles PDF extraction variations"""
        
        # Normalize both texts
        def normalize_text(text):
            import re
            # Convert to lowercase
            text = text.lower()
            # Normalize whitespace (replace multiple spaces/newlines with single space)
            text = re.sub(r'\s+', ' ', text)
            # Remove common punctuation differences
            text = re.sub(r'[^\w\s\u00c0-\u017f]', '', text)  # Keep only word chars and accented chars
            return text.strip()
        
        search_normalized = normalize_text(search_text)
        doc_normalized = normalize_text(document_text)
        
        # Try exact normalized match
        if search_normalized in doc_normalized:
            return True
        
        # Try word-by-word matching with threshold
        search_words = search_normalized.split()
        doc_words = doc_normalized.split()
        
        if len(search_words) < 2:
            return False
        
        # Check if we can find a sequence of words
        for i in range(len(doc_words) - len(search_words) + 1):
            window = doc_words[i:i + len(search_words)]
            matches = sum(1 for sw, dw in zip(search_words, window) if sw == dw)
            match_ratio = matches / len(search_words)
            
            if match_ratio >= 0.8:  # 80% of words must match
                return True
        
        # Try individual word presence with high threshold
        present_words = sum(1 for word in search_words if word in doc_words)
        presence_ratio = present_words / len(search_words)
        
        if presence_ratio >= 0.9 and len(search_words) >= 5:  # 90% presence for longer phrases
            return True
        
        return False
    
    def display_highlighted_pdf_in_streamlit(self, search_terms: List[str] = None):
        """Display PDF with highlights in Streamlit"""
        if search_terms:
            # Create highlighted version
            highlighted_pdf_bytes = self.search_and_highlight_text(search_terms)
            
            # Option 1: Use pdf_viewer with highlighted PDF
            from streamlit_pdf_viewer import pdf_viewer
            
            st.subheader("📄 Document with Highlights")
            pdf_viewer(
                input=highlighted_pdf_bytes,
                width="100%",
                height=800,
                render_text=True,
                key=f"highlighted_pdf"
            )
            
            # Option 2: Show page-by-page as images (alternative view)
            # Removed nested expander to avoid Streamlit API error
            st.markdown("### 📖 Alternative: Page-by-Page View")
            col1, col2 = st.columns([1, 3])
            
            with col1:
                page_num = st.selectbox(
                    "Select Page", 
                    range(1, self.doc.page_count + 1),
                    format_func=lambda x: f"Page {x}",
                    key="page_selector"
                ) - 1
            
            with col2:
                page_image_bytes = self.render_page_with_highlights(page_num, dpi=150)
                st.image(Image.open(io.BytesIO(page_image_bytes)), 
                        caption=f"Page {page_num + 1}", use_column_width=True)
        else:
            # Display original PDF
            from streamlit_pdf_viewer import pdf_viewer
            pdf_viewer(
                input=self.pdf_bytes,
                width="100%", 
                height=800,
                render_text=True,
                key="original_pdf"
            )
    
    def get_context_around_highlights(self, search_terms: List[str], 
                                    context_chars: int = 200) -> List[Dict]:
        """Get text context around highlighted terms for AI processing"""
        full_text = self.extract_full_text()
        contexts = []
        
        for term in search_terms:
            # Find all occurrences of the term
            start = 0
            while True:
                pos = full_text.lower().find(term.lower(), start)
                if pos == -1:
                    break
                
                # Extract context around the term
                context_start = max(0, pos - context_chars)
                context_end = min(len(full_text), pos + len(term) + context_chars)
                context = full_text[context_start:context_end]
                
                contexts.append({
                    "term": term,
                    "context": context,
                    "position": pos
                })
                
                start = pos + 1
        
        return contexts
    
    def get_highlighted_snippets(self, search_terms: List[str], context_chars: int = 150) -> List[Dict]:
        """Get highlighted text snippets with visual context for display below AI messages"""
        snippets = []
        
        for page_num in range(self.doc.page_count):
            page = self.doc[page_num]
            
            for term in search_terms:
                # Search for text instances
                text_instances = page.search_for(term, quads=True)
                
                for inst in text_instances:
                    # Get full page text for context
                    page_text = page.get_text()
                    
                    # Find the term in the page text
                    term_pos = page_text.lower().find(term.lower())
                    if term_pos != -1:
                        # Extract context around the term
                        context_start = max(0, term_pos - context_chars)
                        context_end = min(len(page_text), term_pos + len(term) + context_chars)
                        context = page_text[context_start:context_end]
                        
                        # Create snippet with page image
                        pix = page.get_pixmap(dpi=100)  # Lower DPI for snippets
                        
                        # Get bounding box for cropping
                        bbox = inst.rect
                        
                        snippets.append({
                            "term": term,
                            "page": page_num + 1,
                            "context": context,
                            "bbox": bbox,
                            "page_image": pix.tobytes("png"),
                            "highlighted_area": {
                                "x0": bbox.x0,
                                "y0": bbox.y0, 
                                "x1": bbox.x1,
                                "y1": bbox.y1
                            }
                        })
        
        return snippets

    def display_highlighted_snippets_below_message(self, ai_response: str, original_text: str):
        """Display highlighted PDF directly below an AI message"""
        
        # Extract terms to highlight
        highlight_terms = self.create_ai_response_highlights(ai_response, original_text)
        
        if highlight_terms:
            st.markdown("### 🎯 **Evidence from Document:**")
            
            # Create highlighted PDF
            highlighted_pdf_bytes = self._create_robust_highlighted_pdf(highlight_terms)
            
            if highlighted_pdf_bytes:
                # Display the highlighted PDF directly
                from streamlit_pdf_viewer import pdf_viewer
                
                pdf_viewer(
                    input=highlighted_pdf_bytes,
                    width="100%",
                    height=1200,
                    render_text=True,
                    key=f"evidence_pdf_{hash(ai_response)}"  # Unique key per response
                )
                
                # Show what's highlighted in a compact format
                st.caption(f"🎯 Highlighted: {', '.join([f'\"{term}\"' for term in highlight_terms[:3]])}" + 
                          (f" and {len(highlight_terms)-3} more" if len(highlight_terms) > 3 else ""))
            else:
                st.error("Could not create highlighted PDF")
                
        else:
            # Check if there are any quotes in the response at all
            has_quotes = ('"' in ai_response or '\u201c' in ai_response or '\u201d' in ai_response or 
                         "'" in ai_response or '\u2018' in ai_response or '\u2019' in ai_response)
            
            if has_quotes:
                st.info("🔍 Quotes detected but no matching text found in document")
            else:
                st.caption("💬 AI response contains no quoted text")
        
        return len(highlight_terms) if highlight_terms else 0

    def display_citation_based_references(self, ai_response: str, original_text: str):
        """Display highlighted document for citations found in AI response"""
        
        # Extract quotes directly from AI response instead of searching document
        citation_quotes = self._extract_quotes_from_ai_response(ai_response)
        
        if citation_quotes:
            # Skip showing redundant references section since AI response already shows them
            # Just create and show the highlighted PDF
            all_quotes = list(citation_quotes.values())
            if all_quotes:
                highlighted_pdf_bytes = self._create_robust_highlighted_pdf(all_quotes)
                
                if highlighted_pdf_bytes:
                    # Store for the document viewer to use
                    if 'current_chat_id' in st.session_state and st.session_state.current_chat_id:
                        chat_id = st.session_state.current_chat_id
                        if chat_id in st.session_state.chats:
                            st.session_state.chats[chat_id]['highlighted_pdf'] = highlighted_pdf_bytes
                            st.session_state.chats[chat_id]['highlight_terms'] = all_quotes
                    
                    # Display the highlighted PDF directly
                    from streamlit_pdf_viewer import pdf_viewer
                    pdf_viewer(
                        input=highlighted_pdf_bytes,
                        width="100%",
                        height=1200,
                        render_text=True,
                        key=f"inline_highlighted_pdf_{hash(ai_response)}"
                    )
            
            return len(citation_quotes)
        else:
            # Fallback to old method if no quotes found in AI response
            citations = self._extract_numbered_citations(ai_response)
            if citations:
                st.info("🔍 Citations found but AI didn't provide supporting quotes")
                return 0
            else:
                st.caption("💬 No citations found in response")
                return 0

    def _extract_quotes_from_ai_response(self, ai_response: str) -> Dict[int, str]:
        """Extract numbered quotes directly from AI response"""
        import re
        
        citation_quotes = {}
        
        # Look for patterns like [1] "quote text" or [2] "another quote"
        # This matches: [number] "quote content"
        pattern = r'\[(\d+)\]\s*"([^"]+)"'
        matches = re.findall(pattern, ai_response, re.MULTILINE | re.DOTALL)
        
        for match in matches:
            citation_num = int(match[0])
            quote_text = match[1].strip()
            citation_quotes[citation_num] = quote_text
        
        # Also try alternative patterns like [1]: "quote"
        pattern2 = r'\[(\d+)\]:\s*"([^"]+)"'
        matches2 = re.findall(pattern2, ai_response, re.MULTILINE | re.DOTALL)
        
        for match in matches2:
            citation_num = int(match[0])
            quote_text = match[1].strip()
            citation_quotes[citation_num] = quote_text
        
        return citation_quotes

    def _extract_numbered_citations(self, ai_response: str) -> List[int]:
        """Extract numbered citations from AI response in format [1], [2], etc."""
        import re
        
        # Find all numbered citations in square brackets
        pattern = r'\[(\d+)\]'
        matches = re.findall(pattern, ai_response)
        
        # Convert to integers and sort
        citations = sorted(list(set(int(match) for match in matches)))
        
        return citations

    def _get_search_terms_for_term(self, term: str) -> List[str]:
        """Generate specific search terms that users can use with Ctrl+F to find the exact location"""
        search_terms = []
        
        # Strategy 1: Use the exact term (cleaned up)
        cleaned_term = term.strip()
        if len(cleaned_term) > 10:
            search_terms.append(cleaned_term)
        
        # Strategy 2: Use the first few significant words
        words = cleaned_term.split()
        significant_words = [w for w in words if len(w) > 3]
        
        if len(significant_words) >= 2:
            # First 3-4 significant words
            short_phrase = ' '.join(significant_words[:min(4, len(significant_words))])
            if short_phrase not in search_terms:
                search_terms.append(short_phrase)
        
        # Strategy 3: Individual distinctive words
        distinctive_words = [w for w in significant_words if len(w) > 5]
        for word in distinctive_words[:2]:  # Max 2 distinctive words
            if word not in search_terms:
                search_terms.append(word)
        
        return search_terms
    
    def _create_highlighted_pdf_for_terms(self, highlight_terms: List[str]):
        """Create highlighted PDF and store it for the document viewer"""
        try:
            # Create highlighted version using robust search
            highlighted_pdf_bytes = self._create_robust_highlighted_pdf(highlight_terms)
            
            # Store in session state for the document viewer to use
            if 'current_chat_id' in st.session_state and st.session_state.current_chat_id:
                chat_id = st.session_state.current_chat_id
                if chat_id in st.session_state.chats:
                    st.session_state.chats[chat_id]['highlighted_pdf'] = highlighted_pdf_bytes
                    st.session_state.chats[chat_id]['highlight_terms'] = highlight_terms
        except Exception as e:
            print(f"DEBUG: Error creating highlighted PDF: {e}")
    
    def _create_robust_highlighted_pdf(self, search_terms: List[str]) -> bytes:
        """Create highlighted PDF using robust text matching"""
        # Create a copy of the document for highlighting
        highlighted_doc = fitz.open(stream=self.pdf_bytes, filetype="pdf")
        
        total_highlights = 0
        
        try:
            for page_num in range(highlighted_doc.page_count):
                page = highlighted_doc[page_num]
                page_text = page.get_text()
                
                for i, term in enumerate(search_terms):
                    # Strategy 1: Try exact search first
                    instances = page.search_for(term, quads=True)
                    
                    if instances:
                        for inst in instances:
                            highlight = page.add_highlight_annot(inst)
                            highlight.set_colors(stroke=(1, 1, 0))  # Yellow highlight
                            highlight.update()
                            total_highlights += 1
                    else:
                        # Strategy 2: Try to find the most distinctive parts of the quote
                        highlighted_parts = self._find_and_highlight_distinctive_parts(page, term)
                        total_highlights += highlighted_parts
            
            return highlighted_doc.tobytes()
            
        finally:
            highlighted_doc.close()
    
    def _find_and_highlight_distinctive_parts(self, page, term: str) -> int:
        """Find and highlight the most distinctive/important parts of a quote"""
        highlighted_count = 0
        words = term.split()
        
        if len(words) < 5:  # Only work with substantial quotes
            return 0
        
        # Strategy 1: Look for longer phrases (minimum 5 consecutive words)
        for phrase_length in range(min(len(words), 10), 4, -1):  # 10 words down to 5 words
            for start_idx in range(len(words) - phrase_length + 1):
                phrase = ' '.join(words[start_idx:start_idx + phrase_length])
                
                instances = page.search_for(phrase, quads=True)
                if instances:
                    for inst in instances:
                        highlight = page.add_highlight_annot(inst)
                        highlight.set_colors(stroke=(1, 0.9, 0))  # Light yellow for phrase matches
                        highlight.update()
                    highlighted_count += len(instances)
                    return highlighted_count  # Found a substantial phrase, stop here
        
        # Strategy 2: Only if no 5+ word phrases found, look for very specific distinctive phrases
        # But be much more conservative
        distinctive_phrases = self._extract_very_specific_phrases(term)
        
        for phrase in distinctive_phrases:
            if len(phrase.split()) >= 4:  # Only highlight phrases with 4+ words
                instances = page.search_for(phrase, quads=True)
                if instances:
                    for inst in instances:
                        highlight = page.add_highlight_annot(inst)
                        highlight.set_colors(stroke=(1, 0.8, 0))  # Orange for specific phrases
                        highlight.update()
                    highlighted_count += len(instances)
                    if highlighted_count > 0:
                        break  # Found something substantial, stop
        
        return highlighted_count
    
    def _extract_very_specific_phrases(self, text: str) -> list:
        """Extract only very specific and substantial phrases, avoiding single word matches"""
        distinctive_phrases = []
        words = text.split()
        
        # Only look for longer sequences that are likely to be unique/specific
        for length in range(min(len(words), 8), 3, -1):  # 8 words down to 4 words
            for i in range(len(words) - length + 1):
                phrase = ' '.join(words[i:i + length])
                
                # Only include if it's substantial and likely unique
                if self._is_substantial_phrase(phrase):
                    distinctive_phrases.append(phrase)
        
        return distinctive_phrases[:3]  # Limit to top 3 most promising phrases
    
    def _is_substantial_phrase(self, phrase: str) -> bool:
        """Check if a phrase is substantial enough to be worth highlighting"""
        words = phrase.split()
        
        # Must be at least 4 words
        if len(words) < 4:
            return False
        
        # Look for indicators of substantial/specific content
        indicators = [
            any(len(word) > 8 for word in words),  # Contains long technical words
            any(word[0].isupper() and len(word) > 3 for word in words),  # Contains proper nouns
            '"' in phrase or '(' in phrase or ')' in phrase,  # Contains specific formatting
            any(char in phrase for char in [':', '—', '/', '-']),  # Contains specific punctuation
        ]
        
        # Require at least 2 indicators of specificity
        return sum(indicators) >= 2
    
    def get_highlighted_pdf_bytes(self) -> bytes:
        """Get the highlighted PDF bytes for display"""
        chat_id = st.session_state.get('current_chat_id')
        if chat_id and chat_id in st.session_state.chats:
            return st.session_state.chats[chat_id].get('highlighted_pdf')
        return None
    
    def _get_text_context(self, search_term: str, document_text: str, context_chars: int = 300) -> str:
        """Get text context around a search term"""
        # Normalize search term and document for finding
        search_normalized = self._normalize_for_search(search_term)
        doc_normalized = self._normalize_for_search(document_text)
        
        # Find the position of the term
        pos = doc_normalized.find(search_normalized)
        
        if pos == -1:
            # Try finding by significant words
            words = search_normalized.split()
            significant_words = [w for w in words if len(w) > 3]
            
            if significant_words:
                # Look for the first significant word
                first_word_pos = doc_normalized.find(significant_words[0])
                if first_word_pos != -1:
                    pos = first_word_pos
        
        if pos == -1:
            return None
        
        # Extract context around the found position
        start = max(0, pos - context_chars // 2)
        end = min(len(document_text), pos + len(search_term) + context_chars // 2)
        
        # Use original document text for context (preserves formatting)
        context = document_text[start:end].strip()
        
        # Clean up context (remove excessive whitespace but preserve structure)
        context = re.sub(r'\n\s*\n', '\n\n', context)  # Preserve paragraph breaks
        context = re.sub(r'[ \t]+', ' ', context)  # Normalize spaces
        
        return context
    
    def _highlight_term_in_context(self, term: str, context: str) -> str:
        """Highlight the search term within the context"""
        # Try to find and highlight the term in the context
        term_normalized = self._normalize_for_search(term)
        context_normalized = self._normalize_for_search(context)
        
        # Find the best match within context
        pos = context_normalized.find(term_normalized)
        
        if pos != -1:
            # Find the actual text in the original context that matches
            # This is tricky because of normalization differences
            
            # Try a simple case-insensitive replacement first
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            match = pattern.search(context)
            
            if match:
                matched_text = match.group()
                highlighted_context = context.replace(matched_text, f"**{matched_text}**", 1)
                return highlighted_context
        
        # Fallback: highlight significant words
        words = term.split()
        highlighted_context = context
        
        for word in words:
            if len(word) > 3:  # Only highlight significant words
                pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
                highlighted_context = pattern.sub(f"**{word.upper()}**", highlighted_context)
        
        return highlighted_context
    
    def _normalize_for_search(self, text: str) -> str:
        """Normalize text for searching (same as in fuzzy match)"""
        # Convert to lowercase
        text = text.lower()
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove common punctuation
        text = re.sub(r'[^\w\s\u00c0-\u017f]', '', text)
        return text.strip()
    
    def __del__(self):
        """Clean up document resources"""
        if hasattr(self, 'doc') and self.doc:
            self.doc.close()
        if hasattr(self, 'highlighted_doc') and self.highlighted_doc:
            self.highlighted_doc.close()

# Streamlit integration functions
# @st.cache_data  # Removed - PyMuPDF docs can't be pickled
def process_pdf_with_highlighting(pdf_bytes: bytes) -> EnhancedPDFProcessor:
    """Create PDF processor (caching removed due to PyMuPDF serialization issues)"""
    return EnhancedPDFProcessor(pdf_bytes)

def highlight_ai_referenced_text(pdf_processor: EnhancedPDFProcessor, 
                                ai_response: str, 
                                original_text: str):
    """Highlight text that AI references in its response"""
    
    # Extract terms that should be highlighted
    highlight_terms = pdf_processor.create_ai_response_highlights(ai_response, original_text)
    
    if highlight_terms:
        st.info(f"🎯 Highlighting {len(highlight_terms)} referenced text sections in the document")
        
        # Show the highlighted PDF
        pdf_processor.display_highlighted_pdf_in_streamlit(highlight_terms)
        
        # Show what was highlighted (avoid nested expander)
        st.markdown("### 📝 Highlighted Text Sections")
        for i, term in enumerate(highlight_terms):
            st.markdown(f"**{i+1}.** \"{term}\"")
    else:
        # Show original PDF if no highlights
        pdf_processor.display_highlighted_pdf_in_streamlit()

# Example usage function for your existing app
def integrate_with_existing_chat_app():
    """
    Example of how to integrate this with your existing chat app
    
    Replace this in your app.py where you currently handle PDF display:
    """
    
    # After getting AI response, highlight referenced text
    if 'document_content' in st.session_state.chats[st.session_state.current_chat_id]:
        pdf_bytes = st.session_state.chats[st.session_state.current_chat_id]['document_content']
        original_text = st.session_state.chats[st.session_state.current_chat_id]['document_text']
        
        # Get the latest AI response
        messages = st.session_state.chats[st.session_state.current_chat_id]['messages']
        if messages and messages[-1]['role'] == 'assistant':
            latest_ai_response = messages[-1]['content']
            
            # Create enhanced PDF processor
            pdf_processor = process_pdf_with_highlighting(pdf_bytes)
            
            # Display with AI-referenced highlights
            highlight_ai_referenced_text(pdf_processor, latest_ai_response, original_text) 