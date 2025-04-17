import os
from collections.abc import Generator

import streamlit as st
from openai import AzureOpenAI

# App title
st.set_page_config(page_title="🤗💬 HugChat")

st.title("AZ ODSP ChatRag")

MODEL = os.environ.get("AZURE_OPENAI_API_MODEL")
API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
BASE_URL = os.environ.get("AZURE_OPENAI_API_ENDPOINT")
VERSION = os.environ.get("AZURE_OPENAI_API_VERSION")

client = AzureOpenAI(
    default_headers={"Ocp-Apim-Subscription-Key": API_KEY},
    azure_endpoint=BASE_URL,  # do not add "/openai" at the end here because this will be automatically added by this SDK
    api_key=API_KEY,
    azure_deployment=MODEL,
    api_version=VERSION,
)
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = MODEL

# Store LLM generated responses
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


def generate_response(prompt_input: str) -> Generator[str, None, None]:
    """AI is creating summary for generate_response

    Args:
        prompt_input (str): [description]

    Returns:
        [type]: [description]

    Yields:
        Generator[str, None, None]: [description]
    """
    stream = client.chat.completions.create(
        model=st.session_state["openai_model"],
        messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
        stream=True,
    )
    return stream


# User-provided prompt
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"), st.spinner("Thinking"):
        # response = generate_response(prompt, hf_email, hf_pass)
        # st.write(response)
        response = generate_response(prompt)
        st.write_stream(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
