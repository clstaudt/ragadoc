#!/usr/bin/env python3
"""
Debug script to check context window calculations
"""

import ollama
import os
from app import ModelManager, ContextChecker

def debug_context_detection():
    """Debug context length detection for available models"""
    print("🔍 Debugging Context Length Detection\n")
    
    # Check if running in Docker
    in_docker = os.path.exists('/.dockerenv') or os.environ.get('STREAMLIT_SERVER_ADDRESS') == '0.0.0.0'
    
    if in_docker:
        ollama_base_url = os.environ.get('OLLAMA_BASE_URL', 'http://host.docker.internal:11434')
        print(f"🐳 Docker mode - using {ollama_base_url}")
        client = ollama.Client(host=ollama_base_url)
    else:
        print("💻 Direct mode - using localhost:11434")
        client = None
    
    try:
        # Get available models
        if client:
            models_info = client.list()
        else:
            models_info = ollama.list()
        
        if 'models' not in models_info:
            print("❌ No models found")
            return
        
        models = [model.get('model', model.get('name', '')) for model in models_info['models']]
        print(f"📋 Found {len(models)} models:")
        
        for i, model in enumerate(models):
            print(f"  {i+1}. {model}")
        
        print("\n" + "="*60)
        
        # Analyze each model
        for model_name in models:
            print(f"\n🔍 Analyzing: {model_name}")
            print("-" * 40)
            
            # Try to get actual model info
            try:
                if client:
                    model_info = client.show(model_name)
                else:
                    model_info = ollama.show(model_name)
                
                print("✅ Successfully retrieved model info")
                
                # Check for context length in parameters
                if 'parameters' in model_info and 'num_ctx' in model_info['parameters']:
                    actual_ctx = model_info['parameters']['num_ctx']
                    print(f"📊 Actual num_ctx parameter: {actual_ctx:,} tokens")
                else:
                    print("⚠️  No num_ctx parameter found in model info")
                
                # Show what our detection returns
                detected_ctx = ModelManager.get_context_length(model_name)
                if detected_ctx:
                    print(f"🔄 Detected context: {detected_ctx:,} tokens")
                    
                    # Check if they match
                    if 'parameters' in model_info and 'num_ctx' in model_info['parameters']:
                        actual_ctx = int(model_info['parameters']['num_ctx'])
                        if actual_ctx != detected_ctx:
                            print(f"⚠️  MISMATCH! Actual: {actual_ctx:,}, Detected: {detected_ctx:,}")
                        else:
                            print("✅ Actual and detected match")
                else:
                    print("❌ Could not detect context length")
                
            except Exception as e:
                print(f"❌ Failed to get model info: {e}")
                detected_ctx = ModelManager.get_context_length(model_name)
                if detected_ctx:
                    print(f"🔄 Detected context: {detected_ctx:,} tokens")
                else:
                    print("❌ Could not detect context length")
        
        print("\n" + "="*60)
        print("\n💡 To test with a sample document:")
        print("   python debug_context.py test <model_name> <text>")
        
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")

def test_document_context(model_name, sample_text):
    """Test context calculation with sample text"""
    print(f"\n🧪 Testing context calculation for: {model_name}")
    print("-" * 50)
    
    # Get context length
    context_length = ModelManager.get_context_length(model_name)
    if context_length is None:
        print("❌ Cannot determine context length for this model")
        print("   Context checking would be skipped in the app")
        return
    
    print(f"📏 Detected context length: {context_length:,} tokens")
    
    # Use the actual context checking function
    fits, context_info, error = ContextChecker.check_document_fits_context(sample_text, model_name)
    
    if error:
        print(f"❌ Error: {error}")
        return
    
    if context_info:
        print(f"📄 System prompt tokens: {context_info['system_tokens']:,}")
        print(f"📝 User prompt tokens: {context_info['user_tokens']:,}")
        print(f"💬 Response reserve: {context_info['response_reserve']:,}")
        print(f"📊 Total estimated usage: {context_info['total_estimated_tokens']:,} tokens")
        print(f"📈 Usage percentage: {context_info['usage_percent']:.1f}%")
        
        if context_info['usage_percent'] > 100:
            excess = context_info['total_estimated_tokens'] - context_info['context_length']
            print(f"⚠️  EXCEEDS CONTEXT by {excess:,} tokens")
        elif context_info['usage_percent'] > 80:
            print(f"⚠️  HIGH USAGE - {context_info['available_tokens']:,} tokens remaining")
        else:
            print(f"✅ GOOD FIT - {context_info['available_tokens']:,} tokens remaining")
        
        # Show character/word stats for reference
        char_count = len(sample_text)
        word_count = len(sample_text.split())
        print(f"\n📝 Document stats:")
        print(f"   Characters: {char_count:,}")
        print(f"   Words: {word_count:,}")
        if context_info['system_tokens'] > 0:
            print(f"   Chars per token: {char_count/context_info['system_tokens']:.1f}")
            print(f"   Words per token: {word_count/context_info['system_tokens']:.2f}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        if len(sys.argv) < 4:
            print("Usage: python debug_context.py test <model_name> <sample_text>")
            sys.exit(1)
        
        model_name = sys.argv[2]
        sample_text = " ".join(sys.argv[3:])
        test_document_context(model_name, sample_text)
    else:
        debug_context_detection() 