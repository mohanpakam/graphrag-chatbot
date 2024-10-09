import streamlit as st
import requests
import yaml
import os
import matplotlib.pyplot as plt
import base64
import io
from config import LoggerConfig

config = LoggerConfig.load_config()
logger = LoggerConfig.setup_logger(__name__)

# Set page config
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")

# Function to send user input to the backend API and get response
def get_ai_response(user_input, reset=False):
    api_url = f"{config['backend_api_url']}/chat"
    payload = {
        "message": user_input, 
        "reset": reset
    }
    response = requests.post(api_url, json=payload)
    logger.debug("Request sent to API:", {**payload, "vertex_ai_token": "***"})  # Hide token in logs
    logger.debug("Response received from API:", response.json())
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
                try:
                    ai_response = get_ai_response(user_input)
                except Exception as e:
                    logger.error(f"Error getting AI response: {str(e)}")
                    ai_response = "Oops, something went wrong. Please try again later."
            
            # Add AI response to chat history
            st.session_state.chat_history.append(("ai", ai_response))
            
            # Clear user input by rerunning the app
            st.rerun()

    # Display chat history
    with chat_container:
        for i, (role, message) in enumerate(st.session_state.chat_history):
            if role == "user":
                st.text_input("You:", value=message, disabled=True, key=f"user_message_{i}")
            else:
                st.text_area("AI:", value=message, disabled=True, key=f"ai_message_{i}")

    # Reset Chat button
    if st.button("Reset Chat"):
        # Clear chat history in UI
        st.session_state.chat_history = []
        
        # Send reset signal to backend
        get_ai_response("", reset=True)
        
        st.success("Chat has been reset!")
        st.rerun()

def production_support_tab():
    st.header("Production Support Issue Analysis")

    if st.button("Reset Production Chat"):
        try:
            response = requests.post(f"{config['backend_api_url']}/reset_production_chat")
            st.success(response.json()["message"])
        except Exception as e:
            logger.error(f"Error resetting production chat: {str(e)}")
            st.error("Oops, something went wrong. Please try again later.")

    user_query = st.text_input("Enter your production support query:")

    if user_query:
        if st.button("Analyze Query"):
            try:
                response = requests.post(f"{config['backend_api_url']}/production_issue_query", json={"query": user_query})
                data = response.json()
                
                if data["query_type"] == "specific_issue":
                    st.subheader("SQL Query")
                    st.code(data["sql_query"], language="sql")
                    st.subheader("Query Result")
                    st.table(data["query_results"])
                elif data["query_type"] == "trend_analysis":
                    st.subheader("Trend Summary")
                    st.write(data["trend_summary"])
                    st.subheader("Trend Data")
                    st.write(data["trend_data"])
                    st.subheader("Trend Axes")
                    st.write(f"X-axis: {data['trend_axes']['x']}")
                    st.write(f"Y-axis: {data['trend_axes']['y']}")
                elif data["query_type"] == "general_question":
                    st.subheader("SQL Query")
                    st.code(data["sql_query"], language="sql")
                    st.subheader("Query Result")
                    st.table(data["query_results"])
                    st.subheader("Analysis Summary")
                    st.write(data["analysis_summary"])
                else:
                    st.error("Oops, something went wrong. Please try again later.")
            except Exception as e:
                logger.error(f"Error analyzing production query: {str(e)}")
                st.error("Oops, something went wrong. Please try again later.")

def main():
    # Ask for Google Vertex AI token
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