import streamlit as st
import requests
import yaml
from database_manager import DatabaseManager
import os
import matplotlib.pyplot as plt
import base64
import io

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()

# Set page config
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")

# Function to send user input to the backend API and get response
def get_ai_response(user_input, reset=False):
    api_url = f"{config['backend_api_url']}/chat"
    payload = {
        "message": user_input, 
        "reset": reset,
        "vertex_ai_token": st.session_state.vertex_ai_token
    }
    response = requests.post(api_url, json=payload)
    print("Request sent to API:", {**payload, "vertex_ai_token": "***"})  # Hide token in logs
    print("Response received from API:", response.json())
    return response.json()["response"]

def chatbot_tab():
    st.header("AI Chatbot")

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

# New tab for Production Support Issue Analysis
def production_support_tab():
    st.header("Production Support Issue Analysis")

    if st.button("Reset Production Chat"):
        response = requests.post(f"{config['backend_api_url']}/reset_production_chat")
        st.success(response.json()["message"])

    user_query = st.text_input("Enter your production support query:")

    if user_query:
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Run Query"):
                response = requests.post(f"{config['backend_api_url']}/production_issue_query", json={"query": user_query})
                data = response.json()
                st.subheader("SQL Query")
                st.code(data["sql_query"], language="sql")
                st.subheader("Query Result")
                st.table(data["result"])

        with col2:
            if st.button("Get Analysis Summary"):
                response = requests.post(f"{config['backend_api_url']}/production_issue_analysis", json={"query": user_query})
                data = response.json()
                st.subheader("Analysis Summary")
                st.write(data["analysis_summary"])

        if st.button("Show Trend Graph"):
            response = requests.post(f"{config['backend_api_url']}/production_issue_trend", json={"query": user_query})
            data = response.json()
            graph_base64 = data["graph"]
            graph_bytes = base64.b64decode(graph_base64)
            graph_image = plt.imread(io.BytesIO(graph_bytes), format='png')
            st.image(graph_image, caption="Trend Graph", use_column_width=True)
            st.write(f"X-axis: {data['x_axis']}")
            st.write(f"Y-axis: {data['y_axis']}")

# Main app layout
def main():
    # Ask for Google Vertex AI token
    if 'vertex_ai_token' not in st.session_state:
        st.session_state.vertex_ai_token = st.text_input("Enter your Google Vertex AI token:", type="password")
        if st.session_state.vertex_ai_token:
            os.environ['VERTEX_AI_TOKEN'] = st.session_state.vertex_ai_token
            st.success("Token saved. You can now use the chatbot.")
        else:
            st.warning("Please enter a valid token to use the chatbot.")
            st.stop()

    # Initialize session state for chat history if it doesn't exist
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Create tabs
    tab1, tab2 = st.tabs(["AI Chatbot", "Production Support Analysis"])

    with tab1:
        chatbot_tab()

    with tab2:
        production_support_tab()

    # Footer
    st.markdown("---")
    st.caption("Disclaimer: This chatbot uses AI-generated content. Use the information provided at your own discretion.")

if __name__ == "__main__":
    main()