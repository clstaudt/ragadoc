"""
Test utilities for ensuring proper test environment setup
"""
import subprocess
import sys
import time
from typing import List
from loguru import logger


def is_ollama_running() -> bool:
    """Check if Ollama service is running."""
    try:
        result = subprocess.run(
            ["ollama", "list"], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        return result.returncode == 0
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False


def get_available_models() -> List[str]:
    """Get list of currently available Ollama models."""
    try:
        result = subprocess.run(
            ["ollama", "list"], 
            capture_output=True, 
            text=True, 
            check=True,
            timeout=10
        )
        # Parse model names from output, skip header line
        lines = result.stdout.strip().split('\n')[1:]
        models = []
        for line in lines:
            if line.strip():
                # Extract model name (first column)
                model_name = line.split()[0]
                models.append(model_name)
        return models
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.error(f"Failed to get model list: {e}")
        return []


def pull_model(model: str, timeout: int = 300) -> bool:
    """Pull a specific Ollama model with timeout."""
    logger.info(f"Pulling model: {model}")
    try:
        result = subprocess.run(
            ["ollama", "pull", model], 
            capture_output=True,
            text=True,
            timeout=timeout
        )
        if result.returncode == 0:
            logger.info(f"Successfully pulled {model}")
            return True
        else:
            logger.error(f"Failed to pull {model}: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout pulling {model} after {timeout} seconds")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Error pulling {model}: {e}")
        return False


def ensure_test_models() -> bool:
    """
    Ensure all required models for testing are available.
    
    Returns:
        bool: True if all models are available, False otherwise
    """
    required_models = [
        "tinyllama:latest",
        "nomic-embed-text:latest"
    ]
    
    if not is_ollama_running():
        logger.error("Ollama is not running. Please start Ollama service first.")
        return False
    
    try:
        available_models = get_available_models()
        logger.info(f"Currently available models: {available_models}")
        
        missing_models = []
        for model in required_models:
            if model not in available_models:
                missing_models.append(model)
            else:
                logger.info(f"✓ Model {model} already available")
        
        if not missing_models:
            logger.info("All required models are available!")
            return True
        
        logger.info(f"Missing models: {missing_models}")
        
        # Pull missing models
        success_count = 0
        for model in missing_models:
            if pull_model(model):
                success_count += 1
            else:
                logger.error(f"Failed to pull required model: {model}")
        
        if success_count == len(missing_models):
            logger.info("All required models successfully pulled!")
            return True
        else:
            logger.error(f"Failed to pull {len(missing_models) - success_count} models")
            return False
            
    except Exception as e:
        logger.error(f"Error ensuring test models: {e}")
        return False


def check_test_environment() -> bool:
    """
    Comprehensive check of the test environment.
    
    Returns:
        bool: True if environment is ready for testing
    """
    logger.info("Checking test environment...")
    
    # Check if Ollama is installed
    try:
        subprocess.run(["ollama", "--version"], capture_output=True, check=True)
        logger.info("✓ Ollama is installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("✗ Ollama is not installed or not in PATH")
        return False
    
    # Check if Ollama is running
    if not is_ollama_running():
        logger.error("✗ Ollama service is not running")
        return False
    
    logger.info("✓ Ollama service is running")
    
    # Ensure required models
    if not ensure_test_models():
        return False
    
    logger.info("✓ Test environment is ready!")
    return True 