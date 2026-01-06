import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="GPT-5 Chatbot", layout="centered")

st.title("💬 GPT-5 Chatbot")

# Sidebar – API Key
with st.sidebar:
    st.header("🔑 OpenAI API Key")
    api_key = st.text_input(
        "Enter your OpenAI API key",
        type="password",
        help="Your key is not stored"
    )
    st.markdown("---")
    st.caption("Model: GPT-5")

# Stop if no API key
if not api_key:
    st.info("Please enter your OpenAI API key in the sidebar to begin.")
    st.stop()

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

# Display chat history
for msg in st.session_state.messages[1:]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Type your message...")

if user_input:
    # Add user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # GPT-5 response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model="gpt-5",
                messages=st.session_state.messages,
            )

            assistant_reply = response.choices[0].message.content
            st.markdown(assistant_reply)

    # Save assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_reply}
    )
