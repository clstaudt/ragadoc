import streamlit as st
import ollama

st.title("Ollama Chatbot")

# Function to get available Ollama models
def get_ollama_models():
    try:
        models_info = ollama.list()
        model_names = []
        if 'models' in models_info and isinstance(models_info['models'], list):
            for model_details in models_info['models']:
                # Handle both dictionary and object formats
                if hasattr(model_details, 'model'):
                    # Object with model attribute
                    model_names.append(model_details.model)
                elif isinstance(model_details, dict) and 'model' in model_details:
                    # Dictionary with model key
                    model_names.append(model_details['model'])
                else:
                    st.warning(f"Found an entry without a 'model' key/attribute: {model_details}")
        else:
            st.warning("Ollama list response did not contain a 'models' list or it was malformed.")
        return model_names
    except ollama.ResponseError as e:
        st.error(f"Ollama API Error: {e.error} (Status code: {e.status_code}). Make sure Ollama is running and accessible.")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching Ollama models: {e}. Make sure Ollama is running.")
        return []

# Initialize chat history and selected model
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

# Get available models
available_models = get_ollama_models()

if not available_models:
    st.warning("No usable Ollama models found. Please ensure Ollama is running, models are installed, and they are correctly configured.")
    # We still allow the app to run so the user can see error messages.
    # Consider st.stop() if you want to halt execution completely.

# Model selection dropdown
# Only show dropdown if models are available
if available_models:
    st.session_state.selected_model = st.selectbox(
        "Choose an Ollama model:",
        available_models,
        index=available_models.index(st.session_state.selected_model) if st.session_state.selected_model and st.session_state.selected_model in available_models else 0
    )
else:
    st.session_state.selected_model = None # Ensure no model is selected if none are available

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    if not st.session_state.selected_model:
        st.warning("Please select a model from the dropdown above. If no models are listed, check Ollama.")
    else:
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            # Display assistant response with streaming
            with st.chat_message("assistant"):
                # Prepare messages for Ollama API (list of dicts)
                current_conversation = []
                for msg in st.session_state.messages:
                    current_conversation.append({'role': msg['role'], 'content': msg['content']})
                
                def generate_response():
                    """Generator function for streaming response"""
                    for chunk in ollama.chat(
                        model=st.session_state.selected_model,
                        messages=current_conversation,
                        stream=True
                    ):
                        if chunk['message']['content']:
                            yield chunk['message']['content']
                
                # Stream the response
                assistant_response = st.write_stream(generate_response())
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        except ollama.ResponseError as e:
            st.error(f"Ollama API Error during chat: {e.error} (Status code: {e.status_code})")
        except Exception as e:
            st.error(f"Error communicating with Ollama during chat: {e}") 