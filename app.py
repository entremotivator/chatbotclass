import streamlit as st
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai
import json
from datetime import datetime
import time

# =====================
# Page Configuration
# =====================
st.set_page_config(
    page_title="Multi-LLM Chatbot Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .chat-stats {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🤖 Multi-LLM Chatbot Pro</p>', unsafe_allow_html=True)

# =====================
# Initialize Session State
# =====================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "total_tokens_used" not in st.session_state:
    st.session_state.total_tokens_used = 0

if "conversation_count" not in st.session_state:
    st.session_state.conversation_count = 0

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "You are a helpful, knowledgeable, and friendly AI assistant."

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

if "streaming_enabled" not in st.session_state:
    st.session_state.streaming_enabled = True

# =====================
# Sidebar Configuration
# =====================
with st.sidebar:
    st.header("🔑 API Configuration")
    
    # API Keys in expandable sections
    with st.expander("🔐 API Keys", expanded=False):
        openai_key = st.text_input("OpenAI API Key", type="password", key="openai_key")
        anthropic_key = st.text_input("Anthropic API Key", type="password", key="anthropic_key")
        google_key = st.text_input("Google Gemini API Key", type="password", key="google_key")
        together_key = st.text_input("Together AI API Key", type="password", key="together_key")
    
    st.divider()
    
    # Model Selection
    st.subheader("🎯 Model Settings")
    
    model_category = st.selectbox(
        "Model Provider",
        ["OpenAI", "Anthropic", "Google", "Together AI"]
    )
    
    # Dynamic model selection based on provider
    model_options = {
        "OpenAI": ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
        "Anthropic": ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
        "Google": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"],
        "Together AI": ["meta-llama/Meta-Llama-3.1-70B-Instruct", "meta-llama/Meta-Llama-3.1-8B-Instruct", "mistralai/Mixtral-8x7B-Instruct-v0.1"]
    }
    
    selected_model = st.selectbox(
        "Select Model",
        model_options[model_category]
    )
    
    st.divider()
    
    # Advanced Parameters
    st.subheader("⚙️ Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
        top_p = st.slider("Top P", 0.0, 1.0, 0.9, 0.05)
    
    with col2:
        max_tokens = st.slider("Max Tokens", 256, 8192, 2048, 256)
        presence_penalty = st.slider("Presence Penalty", -2.0, 2.0, 0.0, 0.1)
    
    frequency_penalty = st.slider("Frequency Penalty", -2.0, 2.0, 0.0, 0.1)
    
    streaming_enabled = st.checkbox("Enable Streaming", value=True)
    st.session_state.streaming_enabled = streaming_enabled
    
    st.divider()
    
    # System Prompt Configuration
    st.subheader("📝 System Prompt")
    
    preset_prompts = {
        "Default Assistant": "You are a helpful, knowledgeable, and friendly AI assistant.",
        "Professional": "You are a professional AI assistant. Provide clear, concise, and well-structured responses.",
        "Creative Writer": "You are a creative writing assistant. Help users with storytelling, poetry, and creative content.",
        "Code Expert": "You are an expert programmer. Provide clean, efficient code with explanations.",
        "Teacher": "You are a patient teacher. Explain concepts clearly with examples and encourage learning.",
        "Analyzer": "You are an analytical assistant. Provide detailed analysis with data-driven insights.",
        "Custom": ""
    }
    
    prompt_choice = st.selectbox("Select Preset", list(preset_prompts.keys()))
    
    if prompt_choice == "Custom":
        system_prompt = st.text_area("Custom System Prompt", value=st.session_state.system_prompt, height=100)
    else:
        system_prompt = preset_prompts[prompt_choice]
    
    st.session_state.system_prompt = system_prompt
    
    st.divider()
    
    # Chat Management
    st.subheader("💬 Chat Management")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 New Chat", use_container_width=True):
            # Save current session
            if st.session_state.messages:
                st.session_state.chat_sessions[st.session_state.current_session_id] = {
                    "messages": st.session_state.messages.copy(),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "model": selected_model
                }
            # Reset for new chat
            st.session_state.messages = []
            st.session_state.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Chat History
    if st.session_state.chat_sessions:
        st.subheader("📚 Chat History")
        for session_id, session_data in sorted(st.session_state.chat_sessions.items(), reverse=True):
            if st.button(f"📅 {session_data['timestamp']}", key=session_id, use_container_width=True):
                st.session_state.messages = session_data["messages"].copy()
                st.session_state.current_session_id = session_id
                st.rerun()
    
    st.divider()
    
    # Export Options
    st.subheader("💾 Export")
    
    if st.session_state.messages:
        # Export as JSON
        chat_export = {
            "session_id": st.session_state.current_session_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": selected_model,
            "messages": st.session_state.messages
        }
        
        st.download_button(
            label="📥 Download Chat (JSON)",
            data=json.dumps(chat_export, indent=2),
            file_name=f"chat_{st.session_state.current_session_id}.json",
            mime="application/json",
            use_container_width=True
        )
        
        # Export as Text
        text_export = f"Chat Session: {st.session_state.current_session_id}\n"
        text_export += f"Model: {selected_model}\n"
        text_export += f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        text_export += "="*50 + "\n\n"
        
        for msg in st.session_state.messages:
            text_export += f"{msg['role'].upper()}:\n{msg['content']}\n\n"
        
        st.download_button(
            label="📄 Download Chat (TXT)",
            data=text_export,
            file_name=f"chat_{st.session_state.current_session_id}.txt",
            mime="text/plain",
            use_container_width=True
        )

# =====================
# Main Chat Area
# =====================

# Statistics Dashboard
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="stat-card">
        <h3>{len(st.session_state.messages)}</h3>
        <p>Messages</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="stat-card">
        <h3>{len(st.session_state.chat_sessions)}</h3>
        <p>Saved Chats</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="stat-card">
        <h3>{selected_model.split('/')[-1] if '/' in selected_model else selected_model}</h3>
        <p>Current Model</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# Display chat messages
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Add timestamp and token info if available
        if "timestamp" in msg:
            st.caption(f"🕒 {msg['timestamp']}")
        if "tokens" in msg:
            st.caption(f"🎯 Tokens: {msg['tokens']}")

# =====================
# LLM Call Functions
# =====================

def estimate_tokens(text):
    """Rough token estimation (4 chars ≈ 1 token)"""
    return len(text) // 4

def call_openai(messages, model, api_key):
    """Call OpenAI API with streaming support"""
    try:
        client = OpenAI(api_key=api_key)
        
        # Prepare messages with system prompt
        full_messages = [{"role": "system", "content": st.session_state.system_prompt}]
        full_messages.extend(messages)
        
        if st.session_state.streaming_enabled:
            response = client.chat.completions.create(
                model=model,
                messages=full_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stream=True
            )
            return response
        else:
            response = client.chat.completions.create(
                model=model,
                messages=full_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty
            )
            return response.choices[0].message.content
    except Exception as e:
        return f"❌ OpenAI Error: {str(e)}"

def call_anthropic(messages, model, api_key):
    """Call Anthropic API"""
    try:
        client = Anthropic(api_key=api_key)
        
        # Convert messages format for Claude
        claude_messages = [{"role": msg["role"], "content": msg["content"]} 
                          for msg in messages if msg["role"] != "system"]
        
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=st.session_state.system_prompt,
            messages=claude_messages
        )
        return response.content[0].text
    except Exception as e:
        return f"❌ Anthropic Error: {str(e)}"

def call_google(messages, model, api_key):
    """Call Google Gemini API"""
    try:
        genai.configure(api_key=api_key)
        model_obj = genai.GenerativeModel(
            model_name=model,
            generation_config={
                "temperature": temperature,
                "top_p": top_p,
                "max_output_tokens": max_tokens,
            }
        )
        
        # Use the last user message
        last_message = messages[-1]["content"]
        response = model_obj.generate_content(last_message)
        return response.text
    except Exception as e:
        return f"❌ Google Error: {str(e)}"

def call_together(messages, model, api_key):
    """Call Together AI API"""
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.together.xyz/v1"
        )
        
        full_messages = [{"role": "system", "content": st.session_state.system_prompt}]
        full_messages.extend(messages)
        
        response = client.chat.completions.create(
            model=model,
            messages=full_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Together AI Error: {str(e)}"

def call_llm(messages, provider, model):
    """Route to appropriate LLM based on provider"""
    
    if provider == "OpenAI":
        if not openai_key:
            return "❌ Please enter your OpenAI API key in the sidebar."
        return call_openai(messages, model, openai_key)
    
    elif provider == "Anthropic":
        if not anthropic_key:
            return "❌ Please enter your Anthropic API key in the sidebar."
        return call_anthropic(messages, model, anthropic_key)
    
    elif provider == "Google":
        if not google_key:
            return "❌ Please enter your Google API key in the sidebar."
        return call_google(messages, model, google_key)
    
    elif provider == "Together AI":
        if not together_key:
            return "❌ Please enter your Together AI API key in the sidebar."
        return call_together(messages, model, together_key)
    
    return "❌ Unsupported provider."

# =====================
# Chat Input and Response
# =====================

# Quick action buttons
st.markdown("### 🎯 Quick Actions")
quick_actions = st.columns(4)

with quick_actions[0]:
    if st.button("💡 Explain", use_container_width=True):
        st.session_state.quick_prompt = "Explain this concept in simple terms: "

with quick_actions[1]:
    if st.button("📝 Summarize", use_container_width=True):
        st.session_state.quick_prompt = "Summarize the following: "

with quick_actions[2]:
    if st.button("🔍 Analyze", use_container_width=True):
        st.session_state.quick_prompt = "Analyze this in detail: "

with quick_actions[3]:
    if st.button("💻 Code", use_container_width=True):
        st.session_state.quick_prompt = "Write code for: "

# Chat input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Add quick prompt prefix if selected
    if "quick_prompt" in st.session_state:
        user_input = st.session_state.quick_prompt + user_input
        del st.session_state.quick_prompt
    
    # Add user message
    timestamp = datetime.now().strftime("%H:%M:%S")
    user_message = {
        "role": "user",
        "content": user_input,
        "timestamp": timestamp,
        "tokens": estimate_tokens(user_input)
    }
    
    st.session_state.messages.append(user_message)
    st.session_state.conversation_count += 1
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
        st.caption(f"🕒 {timestamp}")
    
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            start_time = time.time()
            
            response = call_llm(
                st.session_state.messages,
                model_category,
                selected_model
            )
            
            # Handle streaming for OpenAI
            if model_category == "OpenAI" and st.session_state.streaming_enabled and not isinstance(response, str):
                response_text = ""
                response_placeholder = st.empty()
                
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        response_text += chunk.choices[0].delta.content
                        response_placeholder.markdown(response_text + "▌")
                
                response_placeholder.markdown(response_text)
                response = response_text
            else:
                st.markdown(response)
            
            end_time = time.time()
            response_time = round(end_time - start_time, 2)
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            st.caption(f"🕒 {timestamp} | ⏱️ {response_time}s | 🎯 Tokens: ~{estimate_tokens(response)}")
    
    # Add assistant message
    assistant_message = {
        "role": "assistant",
        "content": response,
        "timestamp": timestamp,
        "tokens": estimate_tokens(response),
        "response_time": response_time
    }
    
    st.session_state.messages.append(assistant_message)
    st.session_state.total_tokens_used += estimate_tokens(response) + estimate_tokens(user_input)
    
    st.rerun()

# =====================
# Footer
# =====================

st.divider()

footer_cols = st.columns(3)

with footer_cols[0]:
    st.info(f"💬 **Total Messages:** {len(st.session_state.messages)}")

with footer_cols[1]:
    st.info(f"🎯 **Est. Tokens Used:** ~{st.session_state.total_tokens_used}")

with footer_cols[2]:
    st.info(f"🤖 **Active Model:** {model_category}")

st.caption("Built with Streamlit 🎈 | Multi-LLM Chatbot Pro v2.0")
