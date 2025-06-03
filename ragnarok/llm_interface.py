"""
LLM interface for Document Q&A

This module handles AI response generation, prompt creation, and streaming
without any UI dependencies.
"""

import ollama
from typing import Iterator, Dict, Any, Optional
from loguru import logger


class PromptBuilder:
    """Builds system prompts for different scenarios"""
    
    @staticmethod
    def create_system_prompt(document_content: str, is_rag: bool = False) -> str:
        """
        Create a unified system prompt by assembling reusable components
        
        Args:
            document_content: Either full document text or retrieved chunks text
            is_rag: Boolean indicating if this is for RAG (chunks) or traditional (full doc)
        
        Returns:
            Formatted system prompt string
        """
        # Base role and instruction (shared)
        base_role = "You are a document analysis assistant. Answer questions ONLY using information from this"
        
        # Content section (varies by method)
        if is_rag:
            content_section = f"""relevant document excerpts:

RELEVANT DOCUMENT EXCERPTS:
{document_content}"""
            content_type = "relevant document excerpts"
            content_plural = "excerpts are"
            error_msg = "Error: No relevant document content found for this question."
            decline_msg = "I cannot answer this based on the available document excerpts"
        else:
            content_section = f"""document:

DOCUMENT CONTENT:
{document_content}"""
            content_type = "document"
            content_plural = "document is"
            error_msg = "Error: No document content received, cannot proceed."
            decline_msg = "I cannot answer this based on the document"
        
        # Initial check (varies by method)
        initial_check = f"""INITIAL CHECK:
First, verify you have received {content_type} above. If the {content_plural} empty or missing, respond: "{error_msg}\""""
        
        # Response rules (mostly shared, slight variation in decline message)
        response_rules = f"""RESPONSE RULES:
Choose ONE approach based on whether the {content_type} contain{'s' if not is_rag else ''} relevant information:

1. **IF ANSWERABLE**: Provide a complete answer with citations
- Every factual claim must have a citation [1], [2], etc.
- List citations at the end using this exact format:
    [1] "exact quote from document"
    [2] "another exact quote"

2. **IF NOT ANSWERABLE**: Decline to answer
- State: "{decline_msg}"
- Do NOT include any citations when declining
- Do not attempt to answer the question with your own knowledge."""
        
        # Citation guidelines (shared)
        citation_guidelines = """CITATION GUIDELINES:
- Use verbatim quotes in their original language (never translate)
- Quote meaningful phrases (5-15 words) that provide sufficient context
- Include descriptive context around numbers/measurements (e.g., "increased by 50% compared to" not just "50%")
- Avoid very short snippets like single numbers, dates, or isolated words
- Each citation should be substantial enough to be meaningful on its own
- Each citation on its own line"""
        
        # Language rules (shared)
        language_rules = """LANGUAGE RULES:
- Respond in the user's language
- Keep citations in the document's original language"""
        
        # Assemble the complete prompt
        return f"""{base_role} {content_section}

{initial_check}

{response_rules}

{citation_guidelines}

{language_rules}
"""


class LLMInterface:
    """Interface for interacting with LLM models"""
    
    def __init__(self, ollama_base_url: str = "http://localhost:11434", use_docker: bool = False):
        """
        Initialize LLM interface
        
        Args:
            ollama_base_url: Ollama server URL
            use_docker: Whether running in Docker environment
        """
        self.ollama_base_url = ollama_base_url
        self.use_docker = use_docker
    
    def generate_response_stream(
        self, 
        prompt: str, 
        document_content: str, 
        model_name: str,
        is_rag: bool = False
    ) -> Iterator[Dict[str, Any]]:
        """
        Generate AI response using streaming
        
        Args:
            prompt: User's question
            document_content: Either full document text or retrieved chunks text
            model_name: Name of the model to use
            is_rag: Boolean indicating if this is RAG method
            
        Yields:
            Dictionary containing chunk data and metadata
        """
        # Check if document content is empty
        if not document_content or not document_content.strip():
            yield {
                'content': "I apologize, but I cannot answer your question because the document could not be processed or contains no readable text. Please try uploading a different PDF document.",
                'final': True,
                'error': True
            }
            return
        
        # Create system prompt
        system_prompt = PromptBuilder.create_system_prompt(document_content, is_rag)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        try:
            # Determine Ollama client based on environment
            if self.use_docker:
                client = ollama.Client(host=self.ollama_base_url)
                chat_stream = client.chat(
                    model=model_name,
                    messages=messages,
                    stream=True
                )
            else:
                chat_stream = ollama.chat(
                    model=model_name,
                    messages=messages,
                    stream=True
                )
            
            # Stream response
            for chunk in chat_stream:
                if chunk['message']['content']:
                    yield {
                        'content': chunk['message']['content'],
                        'final': False,
                        'error': False
                    }
            
            # Signal completion
            yield {
                'content': '',
                'final': True,
                'error': False
            }
            
        except Exception as e:
            logger.error(f"Error during streaming: {e}")
            yield {
                'content': f"Error generating response: {e}",
                'final': True,
                'error': True
            }
    
    def generate_response_simple(
        self, 
        prompt: str, 
        document_content: str, 
        model_name: str,
        is_rag: bool = False
    ) -> str:
        """
        Generate AI response without streaming (simple version)
        
        Args:
            prompt: User's question
            document_content: Either full document text or retrieved chunks text
            model_name: Name of the model to use
            is_rag: Boolean indicating if this is RAG method
            
        Returns:
            Generated response text
        """
        # Check if document content is empty
        if not document_content or not document_content.strip():
            return "I apologize, but I cannot answer your question because the document could not be processed or contains no readable text. Please try uploading a different PDF document."
        
        # Create system prompt
        system_prompt = PromptBuilder.create_system_prompt(document_content, is_rag)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        try:
            # Determine Ollama client based on environment
            if self.use_docker:
                client = ollama.Client(host=self.ollama_base_url)
                response = client.chat(
                    model=model_name,
                    messages=messages,
                    stream=False
                )
            else:
                response = ollama.chat(
                    model=model_name,
                    messages=messages,
                    stream=False
                )
            
            return response['message']['content']
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {e}"


class ReasoningParser:
    """Parser for handling reasoning in LLM responses"""
    
    @staticmethod
    def parse_reasoning_response(full_response: str) -> Dict[str, str]:
        """
        Parse response that may contain <think> tags for reasoning
        
        Args:
            full_response: Complete response from LLM
            
        Returns:
            Dictionary with 'reasoning' and 'answer' keys
        """
        # Check for reasoning tags
        think_start = full_response.find('<think>')
        think_end = full_response.find('</think>')
        
        if think_start != -1 and think_end != -1:
            # Response contains reasoning
            reasoning = full_response[think_start + 7:think_end].strip()
            answer = full_response[think_end + 8:].strip()
            
            return {
                'reasoning': reasoning,
                'answer': answer,
                'has_reasoning': True
            }
        else:
            # No reasoning tags
            return {
                'reasoning': '',
                'answer': full_response,
                'has_reasoning': False
            }
    
    @staticmethod
    def extract_reasoning_from_stream(content_chunks: list[str]) -> Dict[str, str]:
        """
        Extract reasoning from a list of streamed content chunks
        
        Args:
            content_chunks: List of content chunks from streaming
            
        Returns:
            Dictionary with parsed reasoning and answer
        """
        full_response = ''.join(content_chunks)
        return ReasoningParser.parse_reasoning_response(full_response) 