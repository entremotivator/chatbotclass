import streamlit as st
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai
import json
from datetime import datetime
import time
import requests

# =====================
# Page Configuration
# =====================
st.set_page_config(
    page_title="Multi-LLM Chatbot Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🤖 Multi-LLM Chatbot Pro + Image Generator</p>', unsafe_allow_html=True)

# =====================
# Initialize Session State
# =====================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
if "total_tokens_used" not in st.session_state:
    st.session_state.total_tokens_used = 0
if "images_generated" not in st.session_state:
    st.session_state.images_generated = 0
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "You are a helpful, knowledgeable, and friendly AI assistant."
if "streaming_enabled" not in st.session_state:
    st.session_state.streaming_enabled = True
if "interaction_mode" not in st.session_state:
    st.session_state.interaction_mode = "chat"
if "auto_detect_image" not in st.session_state:
    st.session_state.auto_detect_image = True

# =====================
# Sidebar
# =====================
with st.sidebar:
    st.header("🔑 API Configuration")
    
    with st.expander("🔐 API Keys", expanded=False):
        openai_key = st.text_input("OpenAI API Key", type="password", key="openai_key")
        anthropic_key = st.text_input("Anthropic API Key", type="password", key="anthropic_key")
        google_key = st.text_input("Google Gemini API Key", type="password", key="google_key")
        together_key = st.text_input("Together AI API Key", type="password", key="together_key")
    
    st.divider()
    
    st.subheader("🎨 Interaction Mode")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💬 Chat", use_container_width=True, type="primary" if st.session_state.interaction_mode == "chat" else "secondary"):
            st.session_state.interaction_mode = "chat"
            st.rerun()
    with col2:
        if st.button("🎨 Image", use_container_width=True, type="primary" if st.session_state.interaction_mode == "image" else "secondary"):
            st.session_state.interaction_mode = "image"
            st.rerun()
    
    auto_detect = st.checkbox("🔍 Auto-detect image requests", value=st.session_state.auto_detect_image)
    st.session_state.auto_detect_image = auto_detect
    st.info("💡 Keywords: 'generate', 'create', 'draw', 'image', 'picture'")
    
    st.divider()
    
    if st.session_state.interaction_mode == "chat":
        st.subheader("🎯 Chat Settings")
        model_category = st.selectbox("Provider", ["OpenAI", "Anthropic", "Google", "Together AI"])
        
        model_options = {
            "OpenAI": ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
            "Anthropic": ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229"],
            "Google": ["gemini-1.5-pro", "gemini-1.5-flash"],
            "Together AI": ["meta-llama/Meta-Llama-3.1-70B-Instruct"]
        }
        selected_model = st.selectbox("Model", model_options[model_category])
        
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
        max_tokens = st.slider("Max Tokens", 256, 8192, 2048, 256)
        streaming_enabled = st.checkbox("Enable Streaming", value=True)
        st.session_state.streaming_enabled = streaming_enabled
        
        st.divider()
        st.subheader("📝 System Prompt")
        preset_prompts = {
            "Default": "You are a helpful, knowledgeable, and friendly AI assistant.",
            "Professional": "You are a professional AI assistant. Provide clear, concise responses.",
            "Creative": "You are a creative writing assistant.",
            "Code Expert": "You are an expert programmer.",
            "Custom": ""
        }
        prompt_choice = st.selectbox("Preset", list(preset_prompts.keys()))
        if prompt_choice == "Custom":
            system_prompt = st.text_area("Custom Prompt", value=st.session_state.system_prompt, height=100)
        else:
            system_prompt = preset_prompts[prompt_choice]
        st.session_state.system_prompt = system_prompt
    else:
        st.subheader("🎨 Image Settings")
        image_model = st.selectbox("Model", ["dall-e-3", "dall-e-2"])
        
        if image_model == "dall-e-3":
            image_size = st.selectbox("Size", ["1024x1024", "1792x1024", "1024x1792"])
            image_quality = st.selectbox("Quality", ["standard", "hd"])
            image_style = st.selectbox("Style", ["vivid", "natural"])
        else:
            image_size = st.selectbox("Size", ["256x256", "512x512", "1024x1024"])
            image_quality = "standard"
            image_style = "vivid"
        
        num_images = st.slider("Images (DALL-E 2)", 1, 4, 1)
        enhance_prompt = st.checkbox("✨ Enhance prompt", value=True)
        
        model_category = "OpenAI"
        selected_model = image_model
        temperature = 0.7
        max_tokens = 2048
        streaming_enabled = False
    
    st.divider()
    st.subheader("💬 Management")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 New Chat", use_container_width=True):
            if st.session_state.messages:
                st.session_state.chat_sessions[st.session_state.current_session_id] = {
                    "messages": st.session_state.messages.copy(),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            st.session_state.messages = []
            st.session_state.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.rerun()
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    if st.session_state.chat_sessions:
        st.subheader("📚 History")
        for session_id, data in sorted(st.session_state.chat_sessions.items(), reverse=True)[:5]:
            if st.button(f"📅 {data['timestamp']}", key=session_id, use_container_width=True):
                st.session_state.messages = data["messages"].copy()
                st.rerun()
    
    st.divider()
    if st.session_state.messages:
        chat_export = {
            "session_id": st.session_state.current_session_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "messages": st.session_state.messages
        }
        st.download_button(
            "📥 Export JSON",
            json.dumps(chat_export, indent=2),
            f"chat_{st.session_state.current_session_id}.json",
            "application/json",
            use_container_width=True
        )

# =====================
# Statistics
# =====================
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="stat-card"><h3>{len(st.session_state.messages)}</h3><p>Messages</p></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="stat-card"><h3>{st.session_state.images_generated}</h3><p>Images</p></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="stat-card"><h3>{len(st.session_state.chat_sessions)}</h3><p>Chats</p></div>', unsafe_allow_html=True)
with col4:
    mode_display = "🎨 Image" if st.session_state.interaction_mode == "image" else "💬 Chat"
    st.markdown(f'<div class="stat-card"><h3>{mode_display}</h3><p>Mode</p></div>', unsafe_allow_html=True)

st.divider()

# =====================
# Display Messages
# =====================
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg.get("type") == "image":
            st.markdown(f"**Prompt:** {msg.get('original_prompt', 'N/A')}")
            if msg.get("enhanced_prompt"):
                with st.expander("✨ Enhanced Prompt"):
                    st.markdown(msg["enhanced_prompt"])
            
            if "images" in msg:
                cols = st.columns(min(len(msg["images"]), 2))
                for i, img_data in enumerate(msg["images"]):
                    with cols[i % 2]:
                        st.image(img_data["url"], caption=f"Image {i+1}", use_container_width=True)
                        try:
                            response = requests.get(img_data["url"])
                            st.download_button(
                                f"⬇️ Download {i+1}",
                                response.content,
                                f"image_{idx}_{i}.png",
                                "image/png",
                                key=f"dl_{idx}_{i}"
                            )
                        except:
                            pass
        else:
            st.markdown(msg["content"])
        
        if "timestamp" in msg:
            st.caption(f"🕒 {msg['timestamp']}")

# =====================
# Helper Functions
# =====================
def estimate_tokens(text):
    return len(str(text)) // 4

def detect_image_request(text):
    keywords = ['generate', 'create', 'draw', 'make', 'image', 'picture', 'photo', 'illustration']
    return any(k in str(text).lower() for k in keywords)

def enhance_image_prompt(prompt, api_key):
    try:
        client = OpenAI(api_key=api_key)
        system = "You are an expert at DALL-E prompts. Enhance this into a detailed, vivid prompt under 400 chars. Include style, lighting, composition, mood, colors."
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Enhance: {prompt}"}
            ],
            max_tokens=150,
            temperature=0.8
        )
        return response.choices[0].message.content.strip()
    except:
        return prompt

def call_openai(messages, model, api_key):
    try:
        client = OpenAI(api_key=api_key)
        full_messages = [{"role": "system", "content": str(st.session_state.system_prompt)}]
        for msg in messages:
            if msg.get("type") != "image":
                full_messages.append({"role": str(msg["role"]), "content": str(msg["content"])})
        
        if st.session_state.streaming_enabled:
            return client.chat.completions.create(
                model=model,
                messages=full_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=full_messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
    except Exception as e:
        return f"❌ OpenAI Error: {str(e)}"

def generate_image_openai(prompt, api_key, model, size, quality, style, n=1, enhance=True):
    try:
        client = OpenAI(api_key=api_key)
        final_prompt = str(prompt)
        enhanced_prompt = None
        
        if enhance and model == "dall-e-3":
            enhanced_prompt = enhance_image_prompt(prompt, api_key)
            final_prompt = enhanced_prompt
        
        if model == "dall-e-3":
            response = client.images.generate(
                model=model,
                prompt=final_prompt,
                size=size,
                quality=quality,
                style=style,
                n=1
            )
        else:
            response = client.images.generate(
                model=model,
                prompt=final_prompt,
                size=size,
                n=min(n, 4)
            )
        
        images = [{"url": img.url} for img in response.data]
        return {
            "success": True,
            "images": images,
            "original_prompt": prompt,
            "enhanced_prompt": enhanced_prompt,
            "model": model
        }
    except Exception as e:
        return {"success": False, "error": str(e), "original_prompt": prompt}

def call_anthropic(messages, model, api_key):
    try:
        client = Anthropic(api_key=api_key)
        claude_messages = [{"role": str(msg["role"]), "content": str(msg["content"])} 
                          for msg in messages if msg["role"] != "system" and msg.get("type") != "image"]
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=str(st.session_state.system_prompt),
            messages=claude_messages
        )
        return response.content[0].text
    except Exception as e:
        return f"❌ Anthropic Error: {str(e)}"

def call_google(messages, model, api_key):
    try:
        genai.configure(api_key=api_key)
        model_obj = genai.GenerativeModel(
            model_name=model,
            generation_config={"temperature": temperature, "max_output_tokens": max_tokens}
        )
        last_message = str(messages[-1]["content"])
        response = model_obj.generate_content(last_message)
        return response.text
    except Exception as e:
        return f"❌ Google Error: {str(e)}"

def call_together(messages, model, api_key):
    try:
        client = OpenAI(api_key=api_key, base_url="https://api.together.xyz/v1")
        full_messages = [{"role": "system", "content": str(st.session_state.system_prompt)}]
        for msg in messages:
            if msg.get("type") != "image":
                full_messages.append({"role": str(msg["role"]), "content": str(msg["content"])})
        response = client.chat.completions.create(
            model=model,
            messages=full_messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Together Error: {str(e)}"

def call_llm(messages, provider, model):
    if provider == "OpenAI":
        if not openai_key:
            return "❌ Please enter OpenAI API key"
        return call_openai(messages, model, openai_key)
    elif provider == "Anthropic":
        if not anthropic_key:
            return "❌ Please enter Anthropic API key"
        return call_anthropic(messages, model, anthropic_key)
    elif provider == "Google":
        if not google_key:
            return "❌ Please enter Google API key"
        return call_google(messages, model, google_key)
    elif provider == "Together AI":
        if not together_key:
            return "❌ Please enter Together API key"
        return call_together(messages, model, together_key)
    return "❌ Unsupported provider"

# =====================
# Quick Actions
# =====================
st.markdown("### 🎯 Quick Actions")
if st.session_state.interaction_mode == "chat":
    qa = st.columns(4)
    with qa[0]:
        if st.button("💡 Explain", use_container_width=True):
            st.session_state.quick_prompt = "Explain: "
    with qa[1]:
        if st.button("📝 Summarize", use_container_width=True):
            st.session_state.quick_prompt = "Summarize: "
    with qa[2]:
        if st.button("🔍 Analyze", use_container_width=True):
            st.session_state.quick_prompt = "Analyze: "
    with qa[3]:
        if st.button("💻 Code", use_container_width=True):
            st.session_state.quick_prompt = "Write code for: "
else:
    qa = st.columns(4)
    with qa[0]:
        if st.button("🎨 Artistic", use_container_width=True):
            st.session_state.quick_prompt = "Artistic illustration of: "
    with qa[1]:
        if st.button("📸 Photo", use_container_width=True):
            st.session_state.quick_prompt = "Photorealistic image of: "
    with qa[2]:
        if st.button("🌈 Fantasy", use_container_width=True):
            st.session_state.quick_prompt = "Fantasy scene with: "
    with qa[3]:
        if st.button("🏙️ Landscape", use_container_width=True):
            st.session_state.quick_prompt = "Beautiful landscape of: "

# =====================
# Chat Input
# =====================
user_input = st.chat_input("Type your message...")

if user_input:
    if "quick_prompt" in st.session_state:
        user_input = st.session_state.quick_prompt + user_input
        del st.session_state.quick_prompt
    
    if st.session_state.auto_detect_image and detect_image_request(user_input):
        st.session_state.interaction_mode = "image"
        st.info("🎨 Auto-switched to Image Mode!")
        time.sleep(1)
        st.rerun()
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    user_message = {
        "role": "user",
        "content": user_input,
        "timestamp": timestamp,
        "tokens": estimate_tokens(user_input)
    }
    st.session_state.messages.append(user_message)
    
    with st.chat_message("user"):
        st.markdown(user_input)
        st.caption(f"🕒 {timestamp}")
    
    if st.session_state.interaction_mode == "image":
        if not openai_key:
            with st.chat_message("assistant"):
                st.error("❌ Enter OpenAI API key for images")
        else:
            with st.chat_message("assistant"):
                with st.spinner("🎨 Generating..."):
                    start_time = time.time()
                    result = generate_image_openai(
                        user_input, openai_key, image_model,
                        image_size, image_quality, image_style,
                        num_images if image_model == "dall-e-2" else 1,
                        enhance_prompt
                    )
                    response_time = round(time.time() - start_time, 2)
                    
                    if result["success"]:
                        st.success(f"✨ Generated {len(result['images'])} image(s) in {response_time}s!")
                        st.markdown(f"**Prompt:** {result['original_prompt']}")
                        
                        if result.get("enhanced_prompt"):
                            with st.expander("✨ Enhanced Prompt"):
                                st.markdown(result["enhanced_prompt"])
                        
                        cols = st.columns(min(len(result["images"]), 2))
                        for i, img_data in enumerate(result["images"]):
                            with cols[i % 2]:
                                st.image(img_data["url"], caption=f"Image {i+1}", use_container_width=True)
                                try:
                                    r = requests.get(img_data["url"])
                                    st.download_button(
                                        f"⬇️ Download {i+1}",
                                        r.content,
                                        f"image_{st.session_state.current_session_id}_{i}.png",
                                        "image/png",
                                        key=f"new_{i}"
                                    )
                                except:
                                    pass
                        
                        assistant_message = {
                            "role": "assistant",
                            "type": "image",
                            "images": result["images"],
                            "original_prompt": result["original_prompt"],
                            "enhanced_prompt": result.get("enhanced_prompt"),
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "response_time": response_time
                        }
                        st.session_state.messages.append(assistant_message)
                        st.session_state.images_generated += len(result["images"])
                    else:
                        st.error(f"❌ Failed: {result['error']}")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Failed: {result['error']}",
                            "timestamp": datetime.now().strftime("%H:%M:%S")
                        })
    else:
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                start_time = time.time()
                response = call_llm(st.session_state.messages, model_category, selected_model)
                
                if model_category == "OpenAI" and st.session_state.streaming_enabled and not isinstance(response, str):
                    response_text = ""
                    placeholder = st.empty()
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            response_text += chunk.choices[0].delta.content
                            placeholder.markdown(response_text + "▌")
                    placeholder.markdown(response_text)
                    response = response_text
                else:
                    st.markdown(response)
                
                response_time = round(time.time() - start_time, 2)
                timestamp = datetime.now().strftime("%H:%M:%S")
                st.caption(f"🕒 {timestamp} | ⏱️ {response_time}s | 🎯 ~{estimate_tokens(response)} tokens")
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "timestamp": timestamp,
            "tokens": estimate_tokens(response),
            "response_time": response_time
        })
        st.session_state.total_tokens_used += estimate_tokens(response) + estimate_tokens(user_input)
    
    st.rerun()

# =====================
# Footer
# =====================
st.divider()
fc = st.columns(4)
with fc[0]:
    st.info(f"💬 **Messages:** {len(st.session_state.messages)}")
with fc[1]:
    st.info(f"🎨 **Images:** {st.session_state.images_generated}")
with fc[2]:
    st.info(f"🎯 **Tokens:** ~{st.session_state.total_tokens_used}")
with fc[3]:
    mode_text = "Image" if st.session_state.interaction_mode == "image" else "Chat"
    st.info(f"🤖 **Mode:** {mode_text}")

st.caption("Built with Streamlit 🎈 | Multi-LLM Chatbot Pro v3.0")
