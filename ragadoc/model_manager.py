"""
Model management for Ollama integration

This module handles all interactions with Ollama models including:
- Model discovery and listing
- Model information retrieval
- Context window analysis
"""

import ollama
import re
from typing import List, Dict, Any, Optional
from loguru import logger


class ModelManager:
    """Manages Ollama models and their configurations"""
    
    def __init__(self, ollama_base_url: str = "http://localhost:11434", use_docker: bool = False):
        """
        Initialize ModelManager
        
        Args:
            ollama_base_url: Ollama server URL
            use_docker: Whether running in Docker environment
        """
        self.ollama_base_url = ollama_base_url
        self.use_docker = use_docker
        
    def get_available_models(self) -> List[str]:
        """Get list of available Ollama models"""
        try:
            if self.use_docker:
                # Docker - use explicit client configuration
                client = ollama.Client(host=self.ollama_base_url)
                models_info = client.list()
            else:
                # Direct execution - use default ollama client
                models_info = ollama.list()
                
            # Handle both dict and ListResponse object
            if hasattr(models_info, 'models'):
                models_list = models_info.models
                return [model.model if hasattr(model, 'model') else model.get('model', model.get('name', '')) 
                       for model in models_list]
            elif isinstance(models_info, dict) and 'models' in models_info:
                return [model.get('model', model.get('name', '')) 
                       for model in models_info['models']]
            return []
        except Exception as e:
            # Provide helpful error message based on likely causes
            error_msg = f"Error connecting to Ollama: {e}"
            if "Connection refused" in str(e) or "No connection" in str(e):
                error_msg += "\n\nPlease ensure Ollama is running:\n"
                if self.use_docker:
                    error_msg += "- For Docker access: `OLLAMA_HOST=0.0.0.0:11434 ollama serve`"
                else:
                    error_msg += "- Direct execution: `ollama serve`"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed model information including context length"""
        try:
            if self.use_docker:
                # Docker - use explicit client configuration
                client = ollama.Client(host=self.ollama_base_url)
                model_info = client.show(model_name)
            else:
                # Direct execution - use default ollama client
                model_info = ollama.show(model_name)
            
            return model_info
        except Exception as e:
            logger.warning(f"Could not get model info for {model_name}: {e}")
            return None
    
    def get_context_length(self, model_name: str) -> Optional[int]:
        """Get the context length for a specific model - returns None if cannot be determined"""
        model_info = self.get_model_info(model_name)
        
        if not model_info:
            return None
            
        # Try to get context length from model parameters first
        try:
            # Check in parameters first (handle both dict and object)
            parameters = None
            if hasattr(model_info, 'parameters'):
                parameters = model_info.parameters
            elif isinstance(model_info, dict) and 'parameters' in model_info:
                parameters = model_info['parameters']
            
            if parameters:
                num_ctx = None
                if hasattr(parameters, 'get') and parameters.get('num_ctx'):
                    num_ctx = parameters['num_ctx']
                elif hasattr(parameters, 'num_ctx'):
                    num_ctx = parameters.num_ctx
                elif isinstance(parameters, dict) and 'num_ctx' in parameters:
                    num_ctx = parameters['num_ctx']
                
                if num_ctx:
                    return int(num_ctx)
            
            # Check in modelfile for PARAMETER num_ctx
            modelfile = None
            if hasattr(model_info, 'modelfile'):
                modelfile = model_info.modelfile
            elif isinstance(model_info, dict) and 'modelfile' in model_info:
                modelfile = model_info['modelfile']
            
            if modelfile:
                ctx_match = re.search(r'PARAMETER\s+num_ctx\s+(\d+)', modelfile, re.IGNORECASE)
                if ctx_match:
                    return int(ctx_match.group(1))
                    
        except Exception as e:
            logger.warning(f"Error parsing model info for {model_name}: {e}")
        
        # Return None if we truly cannot determine it
        logger.warning(f"Could not determine context length for {model_name} - no explicit parameter found")
        return None


class ContextChecker:
    """Utility class for checking context window compatibility"""
    
    @staticmethod
    def estimate_token_count(text: str) -> int:
        """Estimate token count using multiple methods for better accuracy"""
        if not text:
            return 0
        
        # Try to use tiktoken for more accurate counting (if available)
        try:
            import tiktoken
            # Use cl100k_base encoding (used by GPT-4, similar to most modern LLMs)
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except ImportError:
            pass
        
        # Fallback: Multiple estimation methods
        char_count = len(text)
        word_count = len(text.split())
        
        # Different estimation approaches
        char_based = char_count // 4  # ~4 chars per token (conservative)
        word_based = int(word_count * 1.3)  # ~1.3 tokens per word (average)
        
        # Use the higher estimate to be conservative
        return max(char_based, word_based)
    
    @staticmethod
    def check_document_fits_context(
        document_text: str, 
        model_manager: ModelManager, 
        model_name: str, 
        user_prompt: str = "",
        system_prompt: str = ""
    ) -> tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Check if document + system prompt fits in model's context window
        
        Args:
            document_text: The document content
            model_manager: ModelManager instance
            model_name: Name of the model to check
            user_prompt: User's question
            system_prompt: System prompt to use (optional)
            
        Returns:
            Tuple of (fits, context_info_dict, error_message)
        """
        if not document_text or not model_name:
            return True, None, None
        
        context_length = model_manager.get_context_length(model_name)
        if context_length is None:
            return True, None, f"Cannot determine context length for model '{model_name}'. Context checking skipped."
        
        # Use provided system prompt or generate default one
        if not system_prompt:
            system_prompt = f"""You are a document analysis assistant. Answer questions ONLY using information from this document:

DOCUMENT CONTENT:
{document_text}

RESPONSE RULES:
1. **IF ANSWERABLE**: Provide a complete answer with citations
2. **IF NOT ANSWERABLE**: Decline to answer - do not use your own knowledge
"""
        
        # Measure actual token counts
        system_tokens = ContextChecker.estimate_token_count(system_prompt)
        user_tokens = ContextChecker.estimate_token_count(user_prompt)
        
        # Reserve space for response (conservative estimate)
        response_reserve = 1000
        
        total_tokens = system_tokens + user_tokens + response_reserve
        
        fits = total_tokens <= context_length
        usage_percent = (total_tokens / context_length) * 100
        
        context_info = {
            'context_length': context_length,
            'system_tokens': system_tokens,
            'user_tokens': user_tokens,
            'response_reserve': response_reserve,
            'total_estimated_tokens': total_tokens,
            'usage_percent': usage_percent,
            'available_tokens': context_length - total_tokens,
            'document_tokens': ContextChecker.estimate_token_count(document_text)
        }
        
        return fits, context_info, None 