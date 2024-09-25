import streamlit as st
import requests
import yaml
from database_manager import DatabaseManager

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
def get_ai_response(user_input):
    api_url = f"{config['backend_api_url']}/chat"
    response = requests.post(api_url, json={"message": user_input})
    print(response)
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
        
        # Clear user input
        st.session_state.user_input = ""

# Display chat history
with chat_container:
    for role, message in st.session_state.chat_history:
        if role == "user":
            st.text_input("You:", value=message, disabled=True)
        else:
            st.text_area("AI:", value=message, disabled=True)

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.chat_history = []

# Footer
st.markdown("---")
st.caption("Disclaimer: This chatbot uses AI-generated content. Use the information provided at your own discretion.")