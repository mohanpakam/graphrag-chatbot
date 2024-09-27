import streamlit as st
import requests
import yaml
from src.common.database_manager import DatabaseManager

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()

# Set page config
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")

# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Function to send user input to the backend API and get response
def get_ai_response(user_input, reset=False):
    api_url = f"{config['backend_api_url']}/chat"
    payload = {"message": user_input, "reset": reset}
    response = requests.post(api_url, json=payload)
    print("Request sent to API:", payload)
    print("Response received from API:", response.json())
    return response.json()["response"]

# Header
st.title("AI Chatbot")

# Chat history display
chat_container = st.container()

# User input
user_input = st.text_input("Enter your message:", key="user_input")

# Submit button
if st.button("Send"):
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append(("user", user_input))
        
        # Get AI response
        with st.spinner("AI is thinking..."):
            ai_response = get_ai_response(user_input)
        
        # Add AI response to chat history
        st.session_state.chat_history.append(("ai", ai_response))
        
        # Clear user input by rerunning the app
        st.rerun()

# Display chat history
with chat_container:
    for role, message in st.session_state.chat_history:
        if role == "user":
            st.text_input("You:", value=message, disabled=True)
        else:
            st.text_area("AI:", value=message, disabled=True)

# Reset Chat button
if st.button("Reset Chat"):
    # Clear chat history in UI
    st.session_state.chat_history = []
    
    # Send reset signal to backend
    get_ai_response("", reset=True)
    
    st.success("Chat has been reset!")
    st.rerun()

# Footer
st.markdown("---")
st.caption("Disclaimer: This chatbot uses AI-generated content. Use the information provided at your own discretion.")