"""
Simplified PDF processor with highlighting capabilities using PyMuPDF
Enhanced with PyMuPDF4LLM and Marker for high-quality markdown extraction
"""

import streamlit as st
import fitz  # PyMuPDF
import re
from loguru import logger
from typing import List, Tuple, Dict, Optional
import tempfile
import os

# High-quality markdown extraction library
import pymupdf4llm

class EnhancedPDFProcessor:
    """Simplified PDF processor with highlighting capabilities"""

    def __init__(self, pdf_bytes: bytes):
        self.pdf_bytes = pdf_bytes
        self.doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    def extract_full_text(self) -> str:
        """Extract full text from PDF with high-quality structure preservation using PyMuPDF4LLM"""
        try:
            # Save PDF bytes to temporary file for PyMuPDF4LLM
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(self.pdf_bytes)
                temp_path = temp_file.name
            
            try:
                # Try method 1: Auto-detected headers with optimized parameters
                markdown_text = self._extract_with_auto_headers(temp_path)
                if markdown_text and len(markdown_text.strip()) > 100:
                    logger.info("Successfully extracted text using auto-header detection")
                    return markdown_text.strip()
                
                # Try method 2: TOC-based headers
                markdown_text = self._extract_with_toc_headers(temp_path)
                if markdown_text and len(markdown_text.strip()) > 100:
                    logger.info("Successfully extracted text using TOC-based headers")
                    return markdown_text.strip()
                
                # Try method 3: No header detection (plain text with structure)
                markdown_text = self._extract_with_no_headers(temp_path)
                if markdown_text and len(markdown_text.strip()) > 100:
                    logger.info("Successfully extracted text with no header detection")
                    return markdown_text.strip()
                
                logger.warning("All PyMuPDF4LLM methods produced insufficient text, falling back to basic extraction")
                return self._extract_basic_fallback()
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            logger.warning(f"PyMuPDF4LLM extraction failed: {e}, falling back to basic extraction")
            return self._extract_basic_fallback()
    
    def _extract_with_auto_headers(self, temp_path: str) -> str:
        """Extract with automatic header detection based on font sizes"""
        try:
            return pymupdf4llm.to_markdown(
                temp_path,
                # Page processing
                page_chunks=False,
                
                # Text quality settings
                force_text=True,
                ignore_code=False,
                
                # Table detection - crucial for structure
                table_strategy='lines',  # More aggressive table detection
                
                # Image handling
                write_images=False,
                ignore_images=False,
                ignore_graphics=False,
                image_size_limit=0.05,
                
                # Layout and margins
                margins=10,
                
                # Header detection - auto-detect based on font sizes
                hdr_info=None,  # This triggers automatic font size analysis
                
                # Performance settings
                graphics_limit=5000,
                show_progress=False,
            )
        except Exception as e:
            logger.debug(f"Auto-header extraction failed: {e}")
            return ""
    
    def _extract_with_toc_headers(self, temp_path: str) -> str:
        """Extract using Table of Contents for header detection"""
        try:
            # Open document to check for TOC
            doc = fitz.open(temp_path)
            toc = doc.get_toc()
            doc.close()
            
            if toc:  # Only use TOC headers if TOC exists
                toc_headers = pymupdf4llm.TocHeaders(temp_path)
                return pymupdf4llm.to_markdown(
                    temp_path,
                    page_chunks=False,
                    force_text=True,
                    ignore_code=False,
                    table_strategy='lines',
                    write_images=False,
                    ignore_images=False,
                    ignore_graphics=False,
                    image_size_limit=0.05,
                    margins=10,
                    hdr_info=toc_headers,  # Use TOC-based headers
                    graphics_limit=5000,
                    show_progress=False,
                )
            else:
                return ""  # No TOC available
        except Exception as e:
            logger.debug(f"TOC-header extraction failed: {e}")
            return ""
    
    def _extract_with_no_headers(self, temp_path: str) -> str:
        """Extract with no header detection - plain structured text"""
        try:
            return pymupdf4llm.to_markdown(
                temp_path,
                page_chunks=False,
                force_text=True,
                ignore_code=False,
                table_strategy='lines',
                write_images=False,
                ignore_images=False,
                ignore_graphics=False,
                image_size_limit=0.05,
                margins=10,
                hdr_info=False,  # Disable header detection completely
                graphics_limit=5000,
                show_progress=False,
            )
        except Exception as e:
            logger.debug(f"No-header extraction failed: {e}")
            return ""
    
    def _extract_basic_fallback(self) -> str:
        """Basic text extraction fallback (only used if PyMuPDF4LLM fails)"""
        full_text = ""
        for page_num in range(self.doc.page_count):
            page = self.doc[page_num]
            full_text += page.get_text() + "\n"
        return full_text
    

    

    
    def extract_table_of_contents(self) -> List[Dict[str, any]]:
        """Extract table of contents/outline from PDF"""
        toc = []
        try:
            # Try to get built-in TOC first
            outline = self.doc.get_toc()
            if outline:
                for item in outline:
                    level, title, page = item
                    toc.append({
                        'level': level,
                        'title': title.strip(),
                        'page': page,
                        'type': 'outline'
                    })
            else:
                # Extract TOC from document structure
                toc = self._extract_structural_toc()
        except Exception as e:
            logger.warning(f"Could not extract TOC: {e}")
            toc = self._extract_structural_toc()
        
        return toc
    
    def _extract_structural_toc(self) -> List[Dict[str, any]]:
        """Extract TOC from high-quality markdown structure"""
        toc = []
        
        # Use high-quality extraction to get structured text
        markdown_text = self.extract_full_text()
        lines = markdown_text.split('\n')
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            
            # PyMuPDF4LLM already detected headings with # markers
            if re.match(r'^#{1,6}\s+', line):
                # Count the number of # to determine level
                level = len(line) - len(line.lstrip('#'))
                title = re.sub(r'^#{1,6}\s+', '', line).strip()
                
                toc.append({
                    'level': level,
                    'title': title,
                    'page': 1,  # Approximate page (would need more complex logic for exact page)
                    'type': 'structural'
                })
        
        return toc
    
    def _is_toc_entry(self, text: str) -> bool:
        """Check if text looks like a table of contents entry"""
        # Common TOC patterns
        toc_patterns = [
            r'^\d+\.?\s+.+\s+\d+$',  # "1. Title 5" or "1 Title 5"
            r'^.+\.{3,}\s*\d+$',     # "Title....5"
            r'^\d+\.\d+\s+.+',       # "1.1 Subtitle"
            r'^(Chapter|Section|Part)\s+\d+',  # "Chapter 1"
        ]
        
        for pattern in toc_patterns:
            if re.match(pattern, text.strip()):
                return True
        
        return False
    
    def extract_sections(self) -> Dict[str, str]:
        """Extract document sections as separate text blocks using high-quality extraction"""
        sections = {}
        current_section = "Introduction"
        current_text = ""
        
        # Use high-quality markdown extraction which already handles structure
        full_text = self.extract_full_text()
        lines = full_text.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Check if this is a heading (starts with #) - PyMuPDF4LLM already detected these
            if re.match(r'^#{1,6}\s+', line):
                # Save previous section
                if current_text.strip():
                    sections[current_section] = current_text.strip()
                
                # Start new section
                current_section = re.sub(r'^#{1,6}\s+', '', line).strip()
                current_text = ""
            else:
                # Add to current section
                if line:
                    current_text += line + "\n"
        
        # Save final section
        if current_text.strip():
            sections[current_section] = current_text.strip()
        
        return sections
    
    def get_document_metadata(self) -> Dict[str, any]:
        """Extract document metadata and structure information"""
        metadata = {
            'page_count': self.doc.page_count,
            'title': self.doc.metadata.get('title', 'Unknown'),
            'author': self.doc.metadata.get('author', 'Unknown'),
            'subject': self.doc.metadata.get('subject', ''),
            'creator': self.doc.metadata.get('creator', ''),
            'producer': self.doc.metadata.get('producer', ''),
            'creation_date': self.doc.metadata.get('creationDate', ''),
            'modification_date': self.doc.metadata.get('modDate', ''),
            'table_of_contents': self.extract_table_of_contents(),
            'sections': list(self.extract_sections().keys()),
            'has_outline': bool(self.doc.get_toc()),
        }
        
        return metadata

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
                (r'\[(\d+)\]\s*"([^"]+)"', "Pattern 1: [1] \"quote\""),
                (r'\[(\d+)\]:\s*"([^"]+)"', "Pattern 2: [1]: \"quote\""),
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

        # Pattern 1: [1] "exact quote" - the preferred format (anywhere in line, not just start)
        pattern1 = r'\[(\d+)\]\s*"([^"]+)"'
        matches1 = re.findall(pattern1, ai_response, re.MULTILINE)
        
        for match in matches1:
            citation_num = int(match[0])
            quote_text = match[1].strip()
            # For citation highlighting, preserve the full quote text
            # Only use focused extraction for very long quotes (>20 words)
            if len(quote_text.split()) > 20:
                focused_quote = self._extract_focused_quote(quote_text, ai_response, user_question)
                citation_quotes[citation_num] = focused_quote
            else:
                citation_quotes[citation_num] = quote_text

        # Pattern 2: [1]: "exact quote" - legacy format with colon (anywhere in line)
        if not citation_quotes:
            pattern2 = r'\[(\d+)\]:\s*"([^"]+)"'
            matches2 = re.findall(pattern2, ai_response, re.MULTILINE)
            
            for match in matches2:
                citation_num = int(match[0])
                quote_text = match[1].strip()
                # For citation highlighting, preserve the full quote text
                # Only use focused extraction for very long quotes (>20 words)
                if len(quote_text.split()) > 20:
                    focused_quote = self._extract_focused_quote(quote_text, ai_response, user_question)
                    citation_quotes[citation_num] = focused_quote
                else:
                    citation_quotes[citation_num] = quote_text

        # Pattern 3: [Exact quote: "text"] - current problematic format
        if not citation_quotes:
            pattern3 = r'\[Exact quote:\s*"([^"]+)"\]'
            matches3 = re.findall(pattern3, ai_response, re.IGNORECASE)
            
            for i, quote_text in enumerate(matches3, 1):
                quote_text = quote_text.strip()
                # For citation highlighting, preserve the full quote text
                # Only use focused extraction for very long quotes (>20 words)
                if len(quote_text.split()) > 20:
                    focused_quote = self._extract_focused_quote(quote_text, ai_response, user_question)
                    citation_quotes[i] = focused_quote
                else:
                    citation_quotes[i] = quote_text

        # Pattern 3b: "text" in brackets without "Exact quote:" prefix
        if not citation_quotes:
            pattern3b = r'\["([^"]+)"\]'
            matches3b = re.findall(pattern3b, ai_response)
            
            for i, quote_text in enumerate(matches3b, 1):
                if len(quote_text.strip()) > 15:  # Only substantial quotes
                    quote_text = quote_text.strip()
                    # For citation highlighting, preserve the full quote text
                    # Only use focused extraction for very long quotes (>20 words)
                    if len(quote_text.split()) > 20:
                        focused_quote = self._extract_focused_quote(quote_text, ai_response, user_question)
                        citation_quotes[i] = focused_quote
                    else:
                        citation_quotes[i] = quote_text

        # Pattern 4: Any text in double quotes as fallback
        if not citation_quotes:
            pattern4 = r'"([^"]{20,})"'  # At least 20 characters
            matches4 = re.findall(pattern4, ai_response)
            
            for i, quote_text in enumerate(matches4, 1):
                # Only use if it looks like a substantial quote
                cleaned = quote_text.strip()
                if len(cleaned) > 15 and not cleaned.startswith('http'):
                    # For citation highlighting, preserve the full quote text
                    # Only use focused extraction for very long quotes (>20 words)
                    if len(cleaned.split()) > 20:
                        focused_quote = self._extract_focused_quote(cleaned, ai_response, user_question)
                        citation_quotes[i] = focused_quote
                    else:
                        citation_quotes[i] = cleaned

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
        if re.search(r'\bprice\b|\bcost\b|\bpercent\b|\b%\b', combined_text, re.IGNORECASE):
            question_keywords.extend(['price', 'cost', 'euro', '€', 'percent', '%'])
        if re.search(r'\badvantage\b|\bbenefit\b|\bfaster\b|\bsmaller\b|\bimprove\b', combined_text, re.IGNORECASE):
            question_keywords.extend(['advantage', 'benefit', 'faster', 'smaller', 'improve', 'times'])
            
        # If we have question keywords, try to find the most relevant part
        if question_keywords:
            words = quote_text.split()
            best_segment = quote_text  # fallback
            best_score = 0
            
            # Try different segment sizes, prioritizing meaningful phrases
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
                    # Look for percentage patterns with context
                    if re.search(r'\d+%\s+\w+|\w+\s+\d+%', segment):  # Percentage with context
                        score += 3
                    # Look for comparative patterns (6 times faster, 50% smaller)
                    if re.search(r'\d+\s+times\s+\w+|\d+%\s+\w+', segment):  # Comparative patterns
                        score += 3
                    
                    if score > best_score:
                        best_score = score
                        best_segment = segment
            
            # If we found a good focused segment, use it
            if best_score > 0 and len(best_segment.split()) < len(words) * 0.7:
                return best_segment
        
        # If no good focused segment found, try to extract key information patterns
        # Look for percentage patterns with context (this is key for the "50%" issue)
        percentage_matches = re.findall(r'\w*\s*\d+%\s*\w*', quote_text)
        if percentage_matches:
            for match in percentage_matches:
                if len(match.strip()) > 3:  # Must have some context
                    return match.strip()
        
        # Look for comparative patterns (6 times faster, etc.)
        comparative_matches = re.findall(r'\d+\s+times\s+\w+|\w+\s+\d+\s+times', quote_text)
        if comparative_matches:
            for match in comparative_matches:
                return match.strip()
        
        # Look for time patterns with context
        time_matches = re.findall(r'\w*\s*\d{1,2}:\d{2}\s*\w*', quote_text)
        if time_matches:
            for match in time_matches:
                if len(match.strip()) > 5:  # Must have some context
                    return match.strip()
        
        # If still too long, try to find the most meaningful part
        # Look for sentences with numbers or key information
        sentences = re.split(r'[.!?]', quote_text)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence.split()) <= 12 and (
                re.search(r'\d+%', sentence) or  # Contains percentage
                re.search(r'\d+\s+times', sentence) or  # Contains "X times"
                re.search(r'\d{1,2}:\d{2}', sentence) or  # Contains time
                any(keyword.lower() in sentence.lower() for keyword in question_keywords)  # Contains question keywords
            ):
                return sentence
        
        # Final fallback - just take the first part
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
                    # Try exact search first for the complete term
                    instances = page.search_for(term, quads=True)

                    if instances:
                        # Found exact match - highlight it
                        for inst in instances:
                            highlight = page.add_highlight_annot(inst)
                            highlight.set_colors(stroke=(1, 1, 0))  # Yellow highlight
                            highlight.update()
                            if first_highlight_page is None:
                                first_highlight_page = page_num + 1
                    else:
                        # No exact match found - try smart highlighting for longer quotes
                        if len(term.split()) >= 3:  # Lowered threshold for better coverage
                            found = self._smart_highlight_long_quote(page, term)
                            if found and first_highlight_page is None:
                                first_highlight_page = page_num + 1
                        else:
                            # For short terms, try case-insensitive search
                            instances_case_insensitive = page.search_for(term, quads=True, flags=fitz.TEXT_DEHYPHENATE | fitz.TEXT_PRESERVE_WHITESPACE)
                            if instances_case_insensitive:
                                for inst in instances_case_insensitive:
                                    highlight = page.add_highlight_annot(inst)
                                    highlight.set_colors(stroke=(1, 0.8, 0))  # Orange for case-insensitive matches
                                    highlight.update()
                                    if first_highlight_page is None:
                                        first_highlight_page = page_num + 1

            return highlighted_doc.tobytes(), first_highlight_page

        finally:
            highlighted_doc.close()

    def _smart_highlight_long_quote(self, page, term: str) -> bool:
        """Smart highlighting for long quotes - tries to find key parts with context"""
        words = term.split()
        found_any = False
        
        # First, try to find the exact quote or substantial parts of it
        # This is more reliable than extracting individual patterns
        
        # Try to find phrases with context (3-8 words)
        for phrase_length in range(min(len(words), 8), 2, -1):
            for start_idx in range(len(words) - phrase_length + 1):
                phrase = " ".join(words[start_idx : start_idx + phrase_length])
                
                # Skip very generic phrases
                if len(phrase.strip()) < 10:
                    continue
                    
                instances = page.search_for(phrase, quads=True)
                if instances:
                    for inst in instances:
                        highlight = page.add_highlight_annot(inst)
                        highlight.set_colors(stroke=(1, 1, 0))  # Yellow for exact matches
                        highlight.update()
                        found_any = True
                    # Found substantial phrase match, we're done
                    return True
        
        # If no substantial phrases found, try contextual pattern matching
        # Extract key information patterns with surrounding context
        key_patterns_with_context = []
        
        # Look for time patterns with context (e.g., "at 14:30" or "14:30 departure")
        time_pattern = r'\b\w*\s*\d{1,2}:\d{2}\s*\w*\b'
        time_matches = re.findall(time_pattern, term)
        key_patterns_with_context.extend([match.strip() for match in time_matches if len(match.strip()) > 5])
        
        # Look for date patterns with context
        date_pattern = r'\b\w*\s*\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\s*\w*\b'
        date_matches = re.findall(date_pattern, term)
        key_patterns_with_context.extend([match.strip() for match in date_matches if len(match.strip()) > 8])
        
        # Look for percentage patterns with context (e.g., "50% smaller", "increased by 23%")
        percentage_pattern = r'\b\w*\s*\d+%\s*\w*\b'
        percentage_matches = re.findall(percentage_pattern, term)
        key_patterns_with_context.extend([match.strip() for match in percentage_matches if len(match.strip()) > 3])
        
        # Look for number patterns with meaningful context (avoid standalone numbers)
        number_with_context_pattern = r'\b\w+\s+\d+\s+\w+\b|\b\d+\s+\w+\s+\w+\b'
        number_matches = re.findall(number_with_context_pattern, term)
        key_patterns_with_context.extend([match.strip() for match in number_matches if len(match.strip()) > 8])
        
        # Try to highlight contextual patterns
        for pattern in key_patterns_with_context:
            instances = page.search_for(pattern, quads=True)
            if instances:
                for inst in instances:
                    highlight = page.add_highlight_annot(inst)
                    highlight.set_colors(stroke=(0, 1, 0))  # Green for contextual matches
                    highlight.update()
                    found_any = True
        
        # If we found contextual patterns, we're done
        if found_any:
            return True
            
        # Otherwise, fall back to partial matching
        return self._highlight_partial_matches(page, term)

    def _highlight_partial_matches(self, page, term: str) -> bool:
        """Find and highlight partial matches for longer quotes with better context preservation"""
        words = term.split()

        # Try phrases of decreasing length, but be more selective
        for phrase_length in range(min(len(words), 8), 4, -1):  # Start from 4 words minimum
            for start_idx in range(len(words) - phrase_length + 1):
                phrase = " ".join(words[start_idx : start_idx + phrase_length])
                
                # Skip phrases that are just numbers or very generic
                if (re.match(r'^\d+$', phrase.strip()) or  # Just a number
                    phrase.strip().lower() in ['the', 'and', 'of', 'to', 'in', 'for', 'is', 'on', 'that', 'by'] or
                    len(phrase.strip()) < 8):  # Too short
                    continue

                instances = page.search_for(phrase, quads=True)
                if instances:
                    for inst in instances:
                        highlight = page.add_highlight_annot(inst)
                        highlight.set_colors(
                            stroke=(1, 0.8, 0)
                        )  # Orange for partial matches
                        highlight.update()
                    return True  # Found something meaningful, stop here
        
        # If we still haven't found anything, try to find meaningful 3-word phrases
        # but only if they contain important information
        for start_idx in range(len(words) - 2):
            phrase = " ".join(words[start_idx : start_idx + 3])
            
            # Only highlight 3-word phrases if they contain meaningful information
            if (re.search(r'\d+%', phrase) or  # Contains percentage
                re.search(r'\d+\s+\w+', phrase) or  # Contains "number word"
                re.search(r'\w+\s+\d+', phrase) or  # Contains "word number"
                re.search(r'\d{1,2}:\d{2}', phrase) or  # Contains time
                len(phrase.strip()) >= 12):  # Or is substantial enough
                
                # Additional check: avoid standalone numbers
                if not re.match(r'^\d+$', phrase.strip()):
                    instances = page.search_for(phrase, quads=True)
                    if instances:
                        for inst in instances:
                            highlight = page.add_highlight_annot(inst)
                            highlight.set_colors(
                                stroke=(1, 0.6, 0)
                            )  # Darker orange for 3-word contextual matches
                            highlight.update()
                        return True  # Found something, stop here
        
        return False  # Nothing meaningful found

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
