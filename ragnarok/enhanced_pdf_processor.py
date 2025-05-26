"""
Simplified PDF processor with highlighting capabilities using PyMuPDF
"""

import streamlit as st
import fitz  # PyMuPDF
import re
from loguru import logger
from typing import List, Tuple, Dict, Optional


class EnhancedPDFProcessor:
    """Simplified PDF processor with highlighting capabilities"""

    def __init__(self, pdf_bytes: bytes):
        self.pdf_bytes = pdf_bytes
        self.doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    def extract_full_text(self) -> str:
        """Extract full text from PDF"""
        full_text = ""
        for page_num in range(self.doc.page_count):
            page = self.doc[page_num]
            full_text += page.get_text() + "\n"
        return full_text

    def create_ai_response_highlights(
        self, ai_response: str, original_text: str
    ) -> List[str]:
        """Extract quoted text from AI response to highlight in PDF"""
        # Common quote patterns
        quoted_patterns = [
            r'"([^"]+)"',  # ASCII double quotes
            r"\u201c([^\u201d]+)\u201d",  # Unicode smart double quotes
            r"'([^']+)'",  # ASCII single quotes
            r"\u2018([^\u2019]+)\u2019",  # Unicode smart single quotes
        ]

        highlighted_terms = []

        for pattern in quoted_patterns:
            try:
                matches = re.findall(pattern, ai_response)

                for match in matches:
                    cleaned_match = match.strip()

                    # Only process substantial quotes
                    if len(cleaned_match) > 10:
                        if self._fuzzy_text_match(cleaned_match, original_text):
                            highlighted_terms.append(cleaned_match)
            except Exception:
                continue

        return highlighted_terms

    def _fuzzy_text_match(self, search_text: str, document_text: str) -> bool:
        """Simple fuzzy text matching"""

        def normalize_text(text):
            text = text.lower()
            text = re.sub(r"\s+", " ", text)
            text = re.sub(r"[^\w\s]", "", text)
            return text.strip()

        search_normalized = normalize_text(search_text)
        doc_normalized = normalize_text(document_text)

        # Exact match
        if search_normalized in doc_normalized:
            return True

        # Word sequence matching
        search_words = search_normalized.split()
        doc_words = doc_normalized.split()

        if len(search_words) < 3:
            return False

        # Look for sequences of words
        for i in range(len(doc_words) - len(search_words) + 1):
            window = doc_words[i : i + len(search_words)]
            matches = sum(1 for sw, dw in zip(search_words, window) if sw == dw)
            match_ratio = matches / len(search_words)

            if match_ratio >= 0.7:  # 70% of words must match
                return True

        return False

    def display_citation_based_references(
        self,
        ai_response: str,
        original_text: str,
        user_question: str = "",
    ) -> int:
        """Display highlighted document for citations found in AI response"""
        # Extract quotes from AI response
        citation_quotes = self._extract_quotes_from_ai_response(ai_response, user_question)

        # Log debug information instead of showing in UI
        if not citation_quotes:
            logger.debug("No citations found, attempting pattern matching")
            logger.debug(f"AI Response (first 500 chars): {ai_response[:500]}")
            
            # Log what patterns we tried
            patterns = [
                (r'^\[(\d+)\]\s*"([^"]+)"', "Pattern 1: [1] \"quote\""),
                (r'^\[(\d+)\]:\s*"([^"]+)"', "Pattern 2: [1]: \"quote\""),
                (r'\[Exact quote:\s*"([^"]+)"\]', "Pattern 3: [Exact quote: \"text\"]"),
                (r'\["([^"]+)"\]', "Pattern 3b: [\"text\"]"),
                (r'"([^"]{20,})"', "Pattern 4: Any quotes 20+ chars")
            ]
            
            for pattern, description in patterns:
                matches = re.findall(pattern, ai_response, re.MULTILINE | re.IGNORECASE)
                logger.debug(f"{description}: {len(matches)} matches")
                if matches:
                    for i, match in enumerate(matches[:3]):  # Log first 3
                        logger.debug(f"  Match {i+1}: {str(match)[:100]}...")

        if citation_quotes:
            all_quotes = list(citation_quotes.values())
            
            # Show found citations
            st.caption(f"✅ Found {len(citation_quotes)} citation(s)")
            with st.expander("Found Citations", expanded=False):
                for num, quote in citation_quotes.items():
                    st.text(f"[{num}] \"{quote[:100]}{'...' if len(quote) > 100 else ''}\"")
            
            highlighted_pdf_bytes, first_highlight_page = self._create_highlighted_pdf(
                all_quotes
            )

            if highlighted_pdf_bytes:
                # Store for the document viewer
                if (
                    "current_chat_id" in st.session_state
                    and st.session_state.current_chat_id
                ):
                    chat_id = st.session_state.current_chat_id
                    if chat_id in st.session_state.chats:
                        st.session_state.chats[chat_id]["highlighted_pdf"] = (
                            highlighted_pdf_bytes
                        )
                        st.session_state.chats[chat_id]["highlight_terms"] = all_quotes

                # Display the highlighted PDF
                from streamlit_pdf_viewer import pdf_viewer

                viewer_params = {
                    "input": highlighted_pdf_bytes,
                    "width": "100%",
                    "height": 900,
                    "render_text": True,
                    "key": f"inline_highlighted_pdf_{hash(ai_response)}",
                }

                if first_highlight_page:
                    viewer_params["scroll_to_page"] = first_highlight_page

                pdf_viewer(**viewer_params)

            return len(citation_quotes)
        else:
            st.caption("💬 No citations found in response")
            return 0

    def _extract_quotes_from_ai_response(self, ai_response: str, user_question: str = "") -> Dict[int, str]:
        """Extract numbered quotes from AI response using multiple patterns"""
        citation_quotes = {}

        # Pattern 1: [1] "exact quote" - the preferred format
        pattern1 = r'^\[(\d+)\]\s*"([^"]+)"'
        matches1 = re.findall(pattern1, ai_response, re.MULTILINE)
        
        for match in matches1:
            citation_num = int(match[0])
            quote_text = match[1].strip()
            # Try to extract more focused quotes for long citations
            focused_quote = self._extract_focused_quote(quote_text, ai_response, user_question)
            citation_quotes[citation_num] = focused_quote

        # Pattern 2: [1]: "exact quote" - legacy format with colon
        if not citation_quotes:
            pattern2 = r'^\[(\d+)\]:\s*"([^"]+)"'
            matches2 = re.findall(pattern2, ai_response, re.MULTILINE)
            
            for match in matches2:
                citation_num = int(match[0])
                quote_text = match[1].strip()
                focused_quote = self._extract_focused_quote(quote_text, ai_response, user_question)
                citation_quotes[citation_num] = focused_quote

        # Pattern 3: [Exact quote: "text"] - current problematic format
        if not citation_quotes:
            pattern3 = r'\[Exact quote:\s*"([^"]+)"\]'
            matches3 = re.findall(pattern3, ai_response, re.IGNORECASE)
            
            for i, quote_text in enumerate(matches3, 1):
                focused_quote = self._extract_focused_quote(quote_text.strip(), ai_response, user_question)
                citation_quotes[i] = focused_quote

        # Pattern 3b: "text" in brackets without "Exact quote:" prefix
        if not citation_quotes:
            pattern3b = r'\["([^"]+)"\]'
            matches3b = re.findall(pattern3b, ai_response)
            
            for i, quote_text in enumerate(matches3b, 1):
                if len(quote_text.strip()) > 15:  # Only substantial quotes
                    focused_quote = self._extract_focused_quote(quote_text.strip(), ai_response, user_question)
                    citation_quotes[i] = focused_quote

        # Pattern 4: Any text in double quotes as fallback
        if not citation_quotes:
            pattern4 = r'"([^"]{20,})"'  # At least 20 characters
            matches4 = re.findall(pattern4, ai_response)
            
            for i, quote_text in enumerate(matches4, 1):
                # Only use if it looks like a substantial quote
                cleaned = quote_text.strip()
                if len(cleaned) > 15 and not cleaned.startswith('http'):
                    focused_quote = self._extract_focused_quote(cleaned, ai_response, user_question)
                    citation_quotes[i] = focused_quote

        return citation_quotes

    def _extract_focused_quote(self, quote_text: str, ai_response: str, user_question: str = "") -> str:
        """Extract the most relevant part of a long quote based on the question context"""
        # If quote is short enough, return as-is
        if len(quote_text.split()) <= 10:
            return quote_text
            
        # Try to identify what the user is asking about from both the question and AI response
        question_keywords = []
        
        # Analyze the user question first (more reliable)
        combined_text = f"{user_question} {ai_response}"
        
        # Look for common question patterns
        if re.search(r'\barrive\b|\barrival\b', combined_text, re.IGNORECASE):
            question_keywords.extend(['arrive', 'arrival', 'ankunft'])
        if re.search(r'\bdepart\b|\bdeparture\b', combined_text, re.IGNORECASE):
            question_keywords.extend(['depart', 'departure', 'abfahrt'])
        if re.search(r'\btime\b|\bwhen\b', combined_text, re.IGNORECASE):
            question_keywords.extend(['time', 'uhrzeit'])
        if re.search(r'\bdate\b', combined_text, re.IGNORECASE):
            question_keywords.extend(['date'])
        if re.search(r'\bprice\b|\bcost\b', combined_text, re.IGNORECASE):
            question_keywords.extend(['price', 'cost', 'euro', '€'])
            
        # If we have question keywords, try to find the most relevant part
        if question_keywords:
            words = quote_text.split()
            best_segment = quote_text  # fallback
            best_score = 0
            
            # Try different segment sizes
            for segment_size in [3, 5, 7, 10]:
                if segment_size >= len(words):
                    continue
                    
                for i in range(len(words) - segment_size + 1):
                    segment = " ".join(words[i:i + segment_size])
                    
                    # Score this segment based on keyword matches
                    score = 0
                    for keyword in question_keywords:
                        if keyword.lower() in segment.lower():
                            score += 1
                    
                    # Also look for time/date patterns
                    if re.search(r'\d{1,2}:\d{2}', segment):  # Time pattern
                        score += 2
                    if re.search(r'\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}', segment):  # Date pattern
                        score += 2
                    
                    if score > best_score:
                        best_score = score
                        best_segment = segment
            
            # If we found a good focused segment, use it
            if best_score > 0 and len(best_segment.split()) < len(words) * 0.7:
                return best_segment
        
        # If no good focused segment found, try to extract key information
        # Look for time patterns
        time_matches = re.findall(r'\d{1,2}:\d{2}(?:\s*-\s*\d{1,2}/\d{1,2}/\d{4})?', quote_text)
        if time_matches:
            # Return the time with some context
            for time_match in time_matches:
                time_pos = quote_text.find(time_match)
                if time_pos != -1:
                    # Get some words around the time
                    start = max(0, time_pos - 20)
                    end = min(len(quote_text), time_pos + len(time_match) + 20)
                    context = quote_text[start:end].strip()
                    if len(context.split()) <= 10:
                        return context
        
        # If still too long, just take the first part
        words = quote_text.split()
        if len(words) > 15:
            return " ".join(words[:15]) + "..."
            
        return quote_text

    def _create_highlighted_pdf(
        self, search_terms: List[str]
    ) -> Tuple[bytes, Optional[int]]:
        """Create highlighted PDF with smart highlighting"""
        highlighted_doc = fitz.open(stream=self.pdf_bytes, filetype="pdf")
        first_highlight_page = None

        try:
            for page_num in range(highlighted_doc.page_count):
                page = highlighted_doc[page_num]

                for term in search_terms:
                    # Try exact search first
                    instances = page.search_for(term, quads=True)

                    if instances:
                        for inst in instances:
                            highlight = page.add_highlight_annot(inst)
                            highlight.set_colors(stroke=(1, 1, 0))  # Yellow highlight
                            highlight.update()
                            if first_highlight_page is None:
                                first_highlight_page = page_num + 1
                    else:
                        # For long quotes, try smart highlighting
                        if len(term.split()) >= 5:
                            found = self._smart_highlight_long_quote(page, term)
                            if found and first_highlight_page is None:
                                first_highlight_page = page_num + 1

            return highlighted_doc.tobytes(), first_highlight_page

        finally:
            highlighted_doc.close()

    def _smart_highlight_long_quote(self, page, term: str) -> bool:
        """Smart highlighting for long quotes - tries to find key parts"""
        words = term.split()
        found_any = False
        
        # Extract key information patterns (times, dates, numbers, important words)
        key_patterns = []
        
        # Look for time patterns (HH:MM)
        time_pattern = r'\b\d{1,2}:\d{2}\b'
        times = re.findall(time_pattern, term)
        key_patterns.extend(times)
        
        # Look for date patterns (DD/MM/YYYY or similar)
        date_pattern = r'\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b'
        dates = re.findall(date_pattern, term)
        key_patterns.extend(dates)
        
        # Look for numbers that might be important
        number_pattern = r'\b\d+\b'
        numbers = re.findall(number_pattern, term)
        # Only include numbers that are likely important (not too common)
        important_numbers = [n for n in numbers if len(n) >= 2]
        key_patterns.extend(important_numbers)
        
        # Try to highlight key patterns first
        for pattern in key_patterns:
            instances = page.search_for(pattern, quads=True)
            if instances:
                for inst in instances:
                    highlight = page.add_highlight_annot(inst)
                    highlight.set_colors(stroke=(0, 1, 0))  # Green for key info
                    highlight.update()
                    found_any = True
        
        # If we found key patterns, we're done
        if found_any:
            return True
            
        # Otherwise, fall back to partial matching
        return self._highlight_partial_matches(page, term)

    def _highlight_partial_matches(self, page, term: str) -> bool:
        """Find and highlight partial matches for longer quotes"""
        words = term.split()

        # Try phrases of decreasing length
        for phrase_length in range(min(len(words), 8), 3, -1):
            for start_idx in range(len(words) - phrase_length + 1):
                phrase = " ".join(words[start_idx : start_idx + phrase_length])

                instances = page.search_for(phrase, quads=True)
                if instances:
                    for inst in instances:
                        highlight = page.add_highlight_annot(inst)
                        highlight.set_colors(
                            stroke=(1, 0.8, 0)
                        )  # Orange for partial matches
                        highlight.update()
                    return True  # Found something, stop here
        
        return False  # Nothing found

    def __del__(self):
        """Clean up document resources"""
        if hasattr(self, "doc") and self.doc:
            self.doc.close()


# Streamlit integration functions
def process_pdf_with_highlighting(pdf_bytes: bytes) -> EnhancedPDFProcessor:
    """Create PDF processor"""
    return EnhancedPDFProcessor(pdf_bytes)


def highlight_ai_referenced_text(
    pdf_processor: EnhancedPDFProcessor, ai_response: str, original_text: str, user_question: str = ""
):
    """Legacy function for backward compatibility"""
    return pdf_processor.display_citation_based_references(ai_response, original_text, user_question)
