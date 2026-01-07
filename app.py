import streamlit as st

from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai

st.set_page_config(page_title="Multi-LLM Chatbot", layout="centered")
st.title("🤖 Multi-LLM Chatbot")

# =====================
# Sidebar – API Keys
# =====================
with st.sidebar:
    st.header("🔑 API Keys")

    openai_key = st.text_input("OpenAI API Key", type="password")
    anthropic_key = st.text_input("Anthropic API Key", type="password")
    google_key = st.text_input("Google Gemini API Key", type="password")
    together_key = st.text_input("Together / Llama API Key", type="password")

    st.divider()

    model_choice = st.selectbox(
        "Select Model",
        [
            "GPT-5 (OpenAI)",
            "Claude 3.5 Sonnet",
            "Gemini 1.5 Pro",
            "Llama 3.1 70B"
        ]
    )

    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    max_tokens = st.slider("Max Tokens", 256, 4096, 1024)

    if st.button("🔄 Reset Chat"):
        st.session_state.messages = []

# =====================
# Session State
# =====================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful, concise assistant."}
    ]

# Display messages
for msg in st.session_state.messages[1:]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Type your message...")

# =====================
# LLM Call Logic
# =====================
def call_llm(messages):
    last_user_message = messages[-1]["content"]

    # --- OpenAI GPT-5 ---
    if model_choice == "GPT-5 (OpenAI)":
        if not openai_key:
            return "❌ OpenAI API key missing."

        client = OpenAI(api_key=openai_key)
        response = client.chat.completions.create(
            model="gpt-5",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    # --- Claude ---
    if model_choice == "Claude 3.5 Sonnet":
        if not anthropic_key:
            return "❌ Anthropic API key missing."

        client = Anthropic(api_key=anthropic_key)
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": last_user_message}],
        )
        return response.content[0].text

    # --- Gemini ---
    if model_choice == "Gemini 1.5 Pro":
        if not google_key:
            return "❌ Google API key missing."

        genai.configure(api_key=google_key)
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(last_user_message)
        return response.text

    # --- Llama 3.1 via Together ---
    if model_choice == "Llama 3.1 70B":
        if not together_key:
            return "❌ Together API key missing."

        client = OpenAI(
            api_key=together_key,
            base_url="https://api.together.xyz/v1"
        )
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-70B-Instruct",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    return "Model not supported."

# =====================
# Chat Flow
# =====================
if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = call_llm(st.session_state.messages)
            st.markdown(reply)

    st.session_state.messages.append(
        {"role": "assistant", "content": reply}
    )
