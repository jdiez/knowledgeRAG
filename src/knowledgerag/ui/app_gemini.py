import os

import google.generativeai as gpt
import streamlit as st

from knowledgerag.pipeline.local.rag_pipeline import create_rag_pipe

# Load environment variables

rag_pipeline = create_rag_pipe(llm=None)


# Function to translate roles between Gemini and Streamlit terminology
def map_role(role):
    if role == "model":
        return "assistant"
    else:
        return role


def fetch_gemini_response(user_query):
    # Use the session's model to generate a response
    res = rag_pipeline(query=user_query)
    # res = res.candidates[0].content.parts[0].text
    print(res)
    response = st.session_state.chat_session.model.generate_content(res)
    print(f"Gemini's Response: {response}")
    return response.parts[0].text


# Configure Streamlit page settings
st.set_page_config(
    page_title="Chat with Domino Docs.",
    page_icon=":robot_face:",  # Favicon emoji
    layout="wide",  # Page layout option
)

API_KEY = os.getenv("GOOGLE_API_KEY")

# Set up Google Gemini-Pro AI model
gpt.configure(api_key=API_KEY)
model = gpt.GenerativeModel("gemini-2.0-flash")

# Initialize chat session in Streamlit if not already present
if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat(history=[])

# Display the chatbot's title on the page
st.title("🤖 Chat with Gemini-Pro")

# Display the chat history
for msg in st.session_state.chat_session.history:
    with st.chat_message(map_role(msg["role"])):
        st.markdown(msg["content"])

# Input field for user's message
user_input = st.chat_input("Ask Gemini-Pro...")
if user_input:
    # Add user's message to chat and display it
    st.chat_message("user").markdown(user_input)

    # Send user's message to Gemini and get the response
    gemini_response = fetch_gemini_response(user_input)

    # Display Gemini's response
    with st.chat_message("assistant"):
        st.markdown(gemini_response)

    # Add user and assistant messages to the chat history
    st.session_state.chat_session.history.append({"role": "user", "content": user_input})
    st.session_state.chat_session.history.append({"role": "model", "content": gemini_response})
