import streamlit as st
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseUpload
import json
from datetime import datetime
import time
import requests
import io
import base64
from PIL import Image
import PyPDF2
import docx
import csv

# =====================
# Page Configuration
# =====================
st.set_page_config(
    page_title="D Hudson's AI Assistant Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Enhanced Design
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        text-align: center;
    }
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s;
    }
    .stat-card:hover {
        transform: translateY(-5px);
    }
    .success-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .file-upload-box {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        background: #f8f9fa;
    }
    .drive-status {
        padding: 0.5rem;
        border-radius: 8px;
        font-weight: bold;
    }
    .connected {
        background: #d4edda;
        color: #155724;
    }
    .disconnected {
        background: #f8d7da;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🚀 D Hudson\'s AI Assistant Pro</p>', unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #666;'>Advanced Multi-LLM Chatbot with Cloud Integration</h4>", unsafe_allow_html=True)

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
    st.session_state.system_prompt = """You are D Hudson's advanced personal AI assistant. Your personality traits:

🎯 **Core Characteristics:**
- Highly intelligent, analytical, and detail-oriented
- Professional yet approachable and personable
- Quick-witted with a subtle sense of humor
- Proactive in anticipating needs and offering solutions
- Adaptive to different contexts (business, creative, technical, casual)

💼 **Professional Demeanor:**
- Provide comprehensive, well-structured responses
- Think strategically and offer multiple perspectives
- Back up claims with reasoning and examples
- Be direct and efficient, avoiding unnecessary fluff
- Maintain confidentiality and discretion

🚀 **Capabilities Focus:**
- Excel at problem-solving and critical thinking
- Break down complex topics into digestible insights
- Offer creative solutions and innovative approaches
- Provide actionable recommendations
- Learn and adapt from conversation context

🤝 **Communication Style:**
- Use clear, concise language with sophisticated vocabulary when appropriate
- Be confident but humble, admitting when uncertain
- Show enthusiasm for interesting topics and challenges
- Maintain a warm, supportive tone while staying professional
- Use emojis strategically to enhance communication

Remember: You're not just an assistant—you're D Hudson's trusted AI partner for tackling challenges, exploring ideas, and achieving goals."""
if "streaming_enabled" not in st.session_state:
    st.session_state.streaming_enabled = True
if "interaction_mode" not in st.session_state:
    st.session_state.interaction_mode = "chat"
if "auto_detect_image" not in st.session_state:
    st.session_state.auto_detect_image = True
if "drive_service" not in st.session_state:
    st.session_state.drive_service = None
if "drive_folder_id" not in st.session_state:
    st.session_state.drive_folder_id = None
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "file_contents" not in st.session_state:
    st.session_state.file_contents = {}
if "conversation_context" not in st.session_state:
    st.session_state.conversation_context = []

# =====================
# Google Drive Functions
# =====================
def init_drive_service(service_account_json):
    """Initialize Google Drive API service"""
    try:
        credentials = service_account.Credentials.from_service_account_info(
            service_account_json,
            scopes=['https://www.googleapis.com/auth/drive.file']
        )
        service = build('drive', 'v3', credentials=credentials)
        return service
    except Exception as e:
        st.error(f"Failed to initialize Drive: {str(e)}")
        return None

def create_drive_folder(service, folder_name):
    """Create a folder in Google Drive"""
    try:
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        folder = service.files().create(body=file_metadata, fields='id').execute()
        return folder.get('id')
    except Exception as e:
        st.error(f"Failed to create folder: {str(e)}")
        return None

def upload_to_drive(service, file_content, filename, folder_id=None, mime_type='text/plain'):
    """Upload file to Google Drive"""
    try:
        file_metadata = {'name': filename}
        if folder_id:
            file_metadata['parents'] = [folder_id]
        
        if isinstance(file_content, str):
            fh = io.BytesIO(file_content.encode('utf-8'))
        else:
            fh = io.BytesIO(file_content)
        
        media = MediaIoBaseUpload(fh, mimetype=mime_type, resumable=True)
        file = service.files().create(body=file_metadata, media_body=media, fields='id,webViewLink').execute()
        return file
    except Exception as e:
        st.error(f"Upload failed: {str(e)}")
        return None

def save_chat_to_drive(service, messages, session_id, folder_id):
    """Save chat history to Google Drive"""
    try:
        chat_data = {
            "session_id": session_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "messages": messages,
            "total_messages": len(messages)
        }
        content = json.dumps(chat_data, indent=2)
        filename = f"chat_history_{session_id}.json"
        return upload_to_drive(service, content, filename, folder_id, 'application/json')
    except Exception as e:
        st.error(f"Failed to save chat: {str(e)}")
        return None

def save_image_to_drive(service, image_url, prompt, session_id, folder_id):
    """Download and save image to Google Drive"""
    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            filename = f"image_{session_id}_{datetime.now().strftime('%H%M%S')}.png"
            return upload_to_drive(service, response.content, filename, folder_id, 'image/png')
        return None
    except Exception as e:
        st.error(f"Failed to save image: {str(e)}")
        return None

# =====================
# File Processing Functions
# =====================
def process_uploaded_file(uploaded_file):
    """Process different file types and extract text content"""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        content = ""
        
        if file_extension == 'pdf':
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            for page in pdf_reader.pages:
                content += page.extract_text() + "\n"
        
        elif file_extension == 'docx':
            doc = docx.Document(io.BytesIO(uploaded_file.read()))
            content = "\n".join([para.text for para in doc.paragraphs])
        
        elif file_extension == 'txt':
            content = uploaded_file.read().decode('utf-8')
        
        elif file_extension == 'csv':
            csv_data = uploaded_file.read().decode('utf-8')
            content = f"CSV Data:\n{csv_data}"
        
        elif file_extension in ['png', 'jpg', 'jpeg', 'gif', 'bmp']:
            image = Image.open(uploaded_file)
            # Convert image to base64 for storage
            buffered = io.BytesIO()
            image.save(buffered, format=image.format if image.format else "PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            content = f"[IMAGE: {uploaded_file.name}]"
            return {
                'type': 'image',
                'name': uploaded_file.name,
                'content': content,
                'data': img_str,
                'size': len(buffered.getvalue())
            }
        
        elif file_extension == 'json':
            content = uploaded_file.read().decode('utf-8')
            json_data = json.loads(content)
            content = json.dumps(json_data, indent=2)
        
        else:
            content = uploaded_file.read().decode('utf-8', errors='ignore')
        
        return {
            'type': 'text',
            'name': uploaded_file.name,
            'content': content,
            'size': len(content.encode('utf-8'))
        }
    
    except Exception as e:
        return {
            'type': 'error',
            'name': uploaded_file.name,
            'content': f"Error processing file: {str(e)}",
            'size': 0
        }

# =====================
# Sidebar
# =====================
with st.sidebar:
    st.header("⚙️ Configuration Center")
    
    # Google Drive Setup
    with st.expander("☁️ Google Drive Integration", expanded=True):
        st.markdown("**Upload Service Account JSON**")
        service_file = st.file_uploader("Service Account Key", type=['json'], key="service_json")
        
        if service_file:
            try:
                service_account_json = json.load(service_file)
                if st.button("🔗 Connect to Drive", use_container_width=True):
                    with st.spinner("Connecting..."):
                        service = init_drive_service(service_account_json)
                        if service:
                            st.session_state.drive_service = service
                            # Create main folder
                            folder_id = create_drive_folder(service, f"DHudson_AI_Assistant_{datetime.now().strftime('%Y%m')}")
                            st.session_state.drive_folder_id = folder_id
                            st.success("✅ Connected to Google Drive!")
                            time.sleep(1)
                            st.rerun()
            except Exception as e:
                st.error(f"Invalid JSON: {str(e)}")
        
        if st.session_state.drive_service:
            st.markdown('<div class="drive-status connected">✅ Drive Connected</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save Chat", use_container_width=True):
                    if st.session_state.messages:
                        result = save_chat_to_drive(
                            st.session_state.drive_service,
                            st.session_state.messages,
                            st.session_state.current_session_id,
                            st.session_state.drive_folder_id
                        )
                        if result:
                            st.success("Saved to Drive!")
            with col2:
                auto_save = st.checkbox("Auto-save", value=False)
        else:
            st.markdown('<div class="drive-status disconnected">❌ Drive Not Connected</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # API Keys
    with st.expander("🔐 API Keys", expanded=False):
        openai_key = st.text_input("OpenAI API Key", type="password", key="openai_key")
        anthropic_key = st.text_input("Anthropic API Key", type="password", key="anthropic_key")
        google_key = st.text_input("Google Gemini API Key", type="password", key="google_key")
        together_key = st.text_input("Together AI API Key", type="password", key="together_key")
    
    st.divider()
    
    # File Upload Section
    with st.expander("📁 Upload Files", expanded=False):
        st.markdown("**Supported formats:** PDF, DOCX, TXT, CSV, JSON, Images")
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt', 'csv', 'json', 'png', 'jpg', 'jpeg', 'gif'],
            key="file_uploader"
        )
        
        if uploaded_files:
            if st.button("📥 Process Files", use_container_width=True):
                with st.spinner("Processing files..."):
                    for file in uploaded_files:
                        processed = process_uploaded_file(file)
                        if processed['type'] != 'error':
                            st.session_state.uploaded_files.append(processed)
                            st.session_state.file_contents[file.name] = processed['content']
                            st.success(f"✅ {file.name} ({processed['size']} bytes)")
                        else:
                            st.error(f"❌ {file.name}: {processed['content']}")
        
        if st.session_state.uploaded_files:
            st.markdown(f"**📚 Files Loaded: {len(st.session_state.uploaded_files)}**")
            for file_info in st.session_state.uploaded_files:
                st.text(f"• {file_info['name']} ({file_info['type']})")
            if st.button("🗑️ Clear Files", use_container_width=True):
                st.session_state.uploaded_files = []
                st.session_state.file_contents = {}
                st.rerun()
    
    st.divider()
    
    # Interaction Mode
    st.subheader("🎨 Mode Selection")
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
    
    # Chat Settings
    if st.session_state.interaction_mode == "chat":
        st.subheader("🎯 Chat Configuration")
        model_category = st.selectbox("Provider", ["OpenAI", "Anthropic", "Google", "Together AI"])
        
        model_options = {
            "OpenAI": ["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
            "Anthropic": ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-sonnet-20240229"],
            "Google": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"],
            "Together AI": ["meta-llama/Meta-Llama-3.1-70B-Instruct", "meta-llama/Meta-Llama-3.1-8B-Instruct"]
        }
        selected_model = st.selectbox("Model", model_options[model_category])
        
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
        max_tokens = st.slider("Max Tokens", 512, 16384, 4096, 512)
        streaming_enabled = st.checkbox("Enable Streaming", value=True)
        st.session_state.streaming_enabled = streaming_enabled
        
        st.divider()
        st.subheader("📝 System Prompt")
        
        preset_prompts = {
            "D Hudson AI": st.session_state.system_prompt,
            "Professional": "You are a professional AI assistant. Provide clear, concise, and well-structured responses.",
            "Creative Writer": "You are a creative writing assistant with expertise in storytelling, poetry, and narrative development.",
            "Code Expert": "You are an expert programmer proficient in multiple languages. Provide clean, efficient, well-documented code.",
            "Research Assistant": "You are a research assistant. Provide detailed, fact-based responses with citations when possible.",
            "Custom": ""
        }
        
        prompt_choice = st.selectbox("Preset", list(preset_prompts.keys()))
        if prompt_choice == "Custom":
            system_prompt = st.text_area("Custom Prompt", value="", height=150)
        else:
            system_prompt = preset_prompts[prompt_choice]
            if prompt_choice == "D Hudson AI":
                with st.expander("View D Hudson AI Personality"):
                    st.markdown(system_prompt)
        
        st.session_state.system_prompt = system_prompt
        
        # Context Options
        st.divider()
        st.subheader("🧠 Context Settings")
        include_files = st.checkbox("Include uploaded files in context", value=True)
        context_length = st.slider("Conversation history", 5, 50, 20, 5)
        
    else:
        st.subheader("🎨 Image Configuration")
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
        save_to_drive = st.checkbox("☁️ Auto-save to Drive", value=True if st.session_state.drive_service else False)
        
        model_category = "OpenAI"
        selected_model = image_model
        temperature = 0.7
        max_tokens = 4096
        streaming_enabled = False
        include_files = False
        context_length = 20
    
    st.divider()
    
    # Session Management
    st.subheader("💬 Session Management")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 New Chat", use_container_width=True):
            if st.session_state.messages:
                st.session_state.chat_sessions[st.session_state.current_session_id] = {
                    "messages": st.session_state.messages.copy(),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "model": selected_model if st.session_state.interaction_mode == "chat" else image_model
                }
                # Auto-save to Drive if connected
                if st.session_state.drive_service and st.session_state.drive_folder_id:
                    save_chat_to_drive(
                        st.session_state.drive_service,
                        st.session_state.messages,
                        st.session_state.current_session_id,
                        st.session_state.drive_folder_id
                    )
            st.session_state.messages = []
            st.session_state.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.session_state.conversation_context = []
            st.rerun()
    with col2:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_context = []
            st.rerun()
    
    # Chat History
    if st.session_state.chat_sessions:
        st.subheader("📚 Chat History")
        for session_id, data in sorted(st.session_state.chat_sessions.items(), reverse=True)[:10]:
            session_label = f"📅 {data['timestamp']}"
            if 'model' in data:
                session_label += f" | {data['model']}"
            if st.button(session_label, key=session_id, use_container_width=True):
                st.session_state.messages = data["messages"].copy()
                st.rerun()
    
    st.divider()
    
    # Export Options
    if st.session_state.messages:
        st.subheader("📤 Export Options")
        
        chat_export = {
            "session_id": st.session_state.current_session_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": selected_model if st.session_state.interaction_mode == "chat" else image_model,
            "messages": st.session_state.messages,
            "statistics": {
                "total_messages": len(st.session_state.messages),
                "images_generated": st.session_state.images_generated,
                "estimated_tokens": st.session_state.total_tokens_used
            }
        }
        
        st.download_button(
            "📥 Download JSON",
            json.dumps(chat_export, indent=2),
            f"chat_{st.session_state.current_session_id}.json",
            "application/json",
            use_container_width=True
        )
        
        # Text Export
        text_export = f"Chat Session: {st.session_state.current_session_id}\n"
        text_export += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        text_export += "="*50 + "\n\n"
        for msg in st.session_state.messages:
            if msg.get("type") != "image":
                text_export += f"{msg['role'].upper()}: {msg['content']}\n\n"
        
        st.download_button(
            "📝 Download Text",
            text_export,
            f"chat_{st.session_state.current_session_id}.txt",
            "text/plain",
            use_container_width=True
        )

# =====================
# Statistics Dashboard
# =====================
st.markdown("### 📊 Dashboard")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(f'<div class="stat-card"><h2>{len(st.session_state.messages)}</h2><p>💬 Messages</p></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="stat-card"><h2>{st.session_state.images_generated}</h2><p>🎨 Images</p></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="stat-card"><h2>{len(st.session_state.chat_sessions)}</h2><p>📚 Sessions</p></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="stat-card"><h2>{len(st.session_state.uploaded_files)}</h2><p>📁 Files</p></div>', unsafe_allow_html=True)
with col5:
    mode_display = "🎨 Image" if st.session_state.interaction_mode == "image" else "💬 Chat"
    st.markdown(f'<div class="stat-card"><h2>{mode_display}</h2><p>Mode</p></div>', unsafe_allow_html=True)

st.divider()

# =====================
# Display Messages
# =====================
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg.get("type") == "image":
            st.markdown(f"**🎨 Prompt:** {msg.get('original_prompt', 'N/A')}")
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
            caption_text = f"🕒 {msg['timestamp']}"
            if "response_time" in msg:
                caption_text += f" | ⏱️ {msg['response_time']}s"
            if "tokens" in msg:
                caption_text += f" | 📊 ~{msg['tokens']} tokens"
            st.caption(caption_text)

# =====================
# Helper Functions
# =====================
def estimate_tokens(text):
    """Estimate token count"""
    return len(str(text)) // 4

def detect_image_request(text):
    """Detect if user wants to generate an image"""
    keywords = ['generate', 'create', 'draw', 'make', 'design', 'image', 'picture', 'photo', 'illustration', 'artwork']
    return any(k in str(text).lower() for k in keywords)

def enhance_image_prompt(prompt, api_key):
    """Enhance image generation prompt using GPT-4"""
    try:
        client = OpenAI(api_key=api_key)
        system = """You are an expert at creating detailed, vivid DALL-E prompts. 
        Enhance the user's prompt to include:
        - Specific artistic style and medium
        - Lighting and atmosphere
        - Color palette and mood
        - Composition and perspective
        - Additional details that bring the image to life
        Keep it under 400 characters. Be creative and descriptive."""
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Enhance this image prompt: {prompt}"}
            ],
            max_tokens=200,
            temperature=0.8
        )
        return response.choices[0].message.content.strip()
    except:
        return prompt

def build_context_messages(messages, include_files=False, context_length=20):
    """Build context from conversation history and files"""
    context = []
    
    # Add file contents if requested
    if include_files and st.session_state.file_contents:
        file_context = "📁 **Available Files:**\n\n"
        for filename, content in st.session_state.file_contents.items():
            file_context += f"**{filename}:**\n{content[:500]}...\n\n"
        context.append({"role": "system", "content": file_context})
    
    # Add recent conversation history
    recent_messages = messages[-context_length:] if len(messages) > context_length else messages
    for msg in recent_messages:
        if msg.get("type") != "image":
            context.append({"role": msg["role"], "content": msg["content"]})
    
    return context

# =====================
# LLM Call Functions
# =====================
def call_openai(messages, model, api_key, system_prompt, streaming=True):
    """Call OpenAI API"""
    try:
        client = OpenAI(api_key=api_key)
        full_messages = [{"role": "system", "content": str(system_prompt)}]
        
        for msg in messages:
            if msg.get("type") != "image":
                full_messages.append({"role": str(msg["role"]), "content": str(msg["content"])})
        
        if streaming and st.session_state.streaming_enabled:
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
    """Generate images using DALL-E"""
    try:
        client = OpenAI(api_key=api_key)
        final_prompt = str(prompt)
        enhanced_prompt = None
        
        if enhance and model == "dall-e-3":
            with st.spinner("✨ Enhancing prompt..."):
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

def call_anthropic(messages, model, api_key, system_prompt):
    """Call Anthropic Claude API"""
    try:
        client = Anthropic(api_key=api_key)
        claude_messages = [{"role": str(msg["role"]), "content": str(msg["content"])} 
                          for msg in messages if msg["role"] != "system" and msg.get("type") != "image"]
        
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=str(system_prompt),
            messages=claude_messages
        )
        return response.content[0].text
    except Exception as e:
        return f"❌ Anthropic Error: {str(e)}"

def call_google(messages, model, api_key, system_prompt):
    """Call Google Gemini API"""
    try:
        genai.configure(api_key=api_key)
        model_obj = genai.GenerativeModel(
            model_name=model,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens
            },
            system_instruction=str(system_prompt)
        )
        
        # Build conversation history
        chat = model_obj.start_chat(history=[])
        for msg in messages[:-1]:
            if msg.get("type") != "image":
                chat.send_message(msg["content"])
        
        # Send latest message
        response = chat.send_message(messages[-1]["content"])
        return response.text
    except Exception as e:
        return f"❌ Google Error: {str(e)}"

def call_together(messages, model, api_key, system_prompt):
    """Call Together AI API"""
    try:
        client = OpenAI(api_key=api_key, base_url="https://api.together.xyz/v1")
        full_messages = [{"role": "system", "content": str(system_prompt)}]
        
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
        return f"❌ Together AI Error: {str(e)}"

def call_llm(messages, provider, model, system_prompt):
    """Route to appropriate LLM provider"""
    if provider == "OpenAI":
        if not openai_key:
            return "❌ Please enter OpenAI API key in the sidebar"
        return call_openai(messages, model, openai_key, system_prompt)
    elif provider == "Anthropic":
        if not anthropic_key:
            return "❌ Please enter Anthropic API key in the sidebar"
        return call_anthropic(messages, model, anthropic_key, system_prompt)
    elif provider == "Google":
        if not google_key:
            return "❌ Please enter Google API key in the sidebar"
        return call_google(messages, model, google_key, system_prompt)
    elif provider == "Together AI":
        if not together_key:
            return "❌ Please enter Together AI API key in the sidebar"
        return call_together(messages, model, together_key, system_prompt)
    return "❌ Unsupported provider"

# =====================
# Quick Actions
# =====================
st.markdown("### 🎯 Quick Actions")
if st.session_state.interaction_mode == "chat":
    qa = st.columns(5)
    with qa[0]:
        if st.button("💡 Explain", use_container_width=True):
            st.session_state.quick_prompt = "Explain in detail: "
    with qa[1]:
        if st.button("📝 Summarize", use_container_width=True):
            st.session_state.quick_prompt = "Summarize this: "
    with qa[2]:
        if st.button("🔍 Analyze", use_container_width=True):
            st.session_state.quick_prompt = "Analyze and provide insights on: "
    with qa[3]:
        if st.button("💻 Code", use_container_width=True):
            st.session_state.quick_prompt = "Write well-documented code for: "
    with qa[4]:
        if st.button("📊 Compare", use_container_width=True):
            st.session_state.quick_prompt = "Compare and contrast: "
else:
    qa = st.columns(5)
    with qa[0]:
        if st.button("🎨 Artistic", use_container_width=True):
            st.session_state.quick_prompt = "Create an artistic illustration of: "
    with qa[1]:
        if st.button("📸 Photo", use_container_width=True):
            st.session_state.quick_prompt = "Generate a photorealistic image of: "
    with qa[2]:
        if st.button("🌈 Fantasy", use_container_width=True):
            st.session_state.quick_prompt = "Design a fantasy scene featuring: "
    with qa[3]:
        if st.button("🏙️ Landscape", use_container_width=True):
            st.session_state.quick_prompt = "Create a stunning landscape of: "
    with qa[4]:
        if st.button("🎭 Abstract", use_container_width=True):
            st.session_state.quick_prompt = "Generate abstract art representing: "

# =====================
# Chat Input Handler
# =====================
user_input = st.chat_input("💬 Message D Hudson's AI Assistant...")

if user_input:
    # Apply quick prompt if exists
    if "quick_prompt" in st.session_state:
        user_input = st.session_state.quick_prompt + user_input
        del st.session_state.quick_prompt
    
    # Auto-detect image generation request
    if st.session_state.auto_detect_image and detect_image_request(user_input):
        st.session_state.interaction_mode = "image"
        st.info("🎨 Auto-switched to Image Mode based on your request!")
        time.sleep(1)
        st.rerun()
    
    # Create user message
    timestamp = datetime.now().strftime("%H:%M:%S")
    user_message = {
        "role": "user",
        "content": user_input,
        "timestamp": timestamp,
        "tokens": estimate_tokens(user_input)
    }
    st.session_state.messages.append(user_message)
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
        st.caption(f"🕒 {timestamp}")
    
    # Handle based on mode
    if st.session_state.interaction_mode == "image":
        # Image Generation Mode
        if not openai_key:
            with st.chat_message("assistant"):
                st.error("❌ Please enter your OpenAI API key in the sidebar to generate images")
        else:
            with st.chat_message("assistant"):
                with st.spinner("🎨 Creating your masterpiece..."):
                    start_time = time.time()
                    result = generate_image_openai(
                        user_input, openai_key, image_model,
                        image_size, image_quality, image_style,
                        num_images if image_model == "dall-e-2" else 1,
                        enhance_prompt
                    )
                    response_time = round(time.time() - start_time, 2)
                    
                    if result["success"]:
                        st.success(f"✨ Successfully generated {len(result['images'])} image(s) in {response_time}s!")
                        st.markdown(f"**🎨 Original Prompt:** {result['original_prompt']}")
                        
                        if result.get("enhanced_prompt"):
                            with st.expander("✨ View Enhanced Prompt"):
                                st.markdown(result["enhanced_prompt"])
                        
                        # Display images
                        cols = st.columns(min(len(result["images"]), 2))
                        for i, img_data in enumerate(result["images"]):
                            with cols[i % 2]:
                                st.image(img_data["url"], caption=f"Image {i+1}", use_container_width=True)
                                
                                # Download button
                                try:
                                    r = requests.get(img_data["url"])
                                    st.download_button(
                                        f"⬇️ Download Image {i+1}",
                                        r.content,
                                        f"dhudson_image_{st.session_state.current_session_id}_{i}.png",
                                        "image/png",
                                        key=f"new_img_{i}"
                                    )
                                except:
                                    pass
                                
                                # Save to Google Drive
                                if st.session_state.drive_service and save_to_drive:
                                    with st.spinner(f"☁️ Saving to Drive..."):
                                        drive_result = save_image_to_drive(
                                            st.session_state.drive_service,
                                            img_data["url"],
                                            result["original_prompt"],
                                            st.session_state.current_session_id,
                                            st.session_state.drive_folder_id
                                        )
                                        if drive_result:
                                            st.success(f"✅ Saved to Drive!")
                        
                        # Save message
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
                        st.error(f"❌ Image generation failed: {result['error']}")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Image generation failed: {result['error']}",
                            "timestamp": datetime.now().strftime("%H:%M:%S")
                        })
    else:
        # Chat Mode
        with st.chat_message("assistant"):
            with st.spinner("🤔 D Hudson's AI is thinking..."):
                start_time = time.time()
                
                # Build context
                context_messages = build_context_messages(
                    st.session_state.messages,
                    include_files,
                    context_length
                )
                
                # Get response
                response = call_llm(
                    context_messages,
                    model_category,
                    selected_model,
                    st.session_state.system_prompt
                )
                
                # Handle streaming
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
                
                # Display metadata
                st.caption(f"🕒 {timestamp} | ⏱️ {response_time}s | 🎯 ~{estimate_tokens(response)} tokens | 🤖 {selected_model}")
        
        # Save assistant message
        assistant_message = {
            "role": "assistant",
            "content": response,
            "timestamp": timestamp,
            "tokens": estimate_tokens(response),
            "response_time": response_time,
            "model": selected_model
        }
        st.session_state.messages.append(assistant_message)
        st.session_state.total_tokens_used += estimate_tokens(response) + estimate_tokens(user_input)
        
        # Auto-save to Drive if enabled
        if st.session_state.drive_service and auto_save:
            save_chat_to_drive(
                st.session_state.drive_service,
                st.session_state.messages,
                st.session_state.current_session_id,
                st.session_state.drive_folder_id
            )
    
    st.rerun()

# =====================
# Footer & Info
# =====================
st.divider()

# Statistics Footer
fc = st.columns(6)
with fc[0]:
    st.info(f"💬 **Messages:** {len(st.session_state.messages)}")
with fc[1]:
    st.info(f"🎨 **Images:** {st.session_state.images_generated}")
with fc[2]:
    st.info(f"🎯 **Tokens:** ~{st.session_state.total_tokens_used:,}")
with fc[3]:
    st.info(f"📚 **Sessions:** {len(st.session_state.chat_sessions)}")
with fc[4]:
    st.info(f"📁 **Files:** {len(st.session_state.uploaded_files)}")
with fc[5]:
    mode_text = "🎨 Image" if st.session_state.interaction_mode == "image" else "💬 Chat"
    st.info(f"**Mode:** {mode_text}")

st.markdown("---")

# Help Section
with st.expander("ℹ️ Help & Features"):
    st.markdown("""
    ### 🚀 D Hudson's AI Assistant Pro - Features
    
    **💬 Multi-LLM Chat:**
    - OpenAI GPT-4, Claude, Gemini, Together AI
    - Customizable system prompts
    - Streaming responses
    - Context-aware conversations
    
    **🎨 Image Generation:**
    - DALL-E 3 & DALL-E 2 support
    - Automatic prompt enhancement
    - HD quality options
    - Multiple sizes and styles
    
    **☁️ Google Drive Integration:**
    - Auto-save chat history
    - Save generated images
    - Organized folder structure
    - Easy retrieval
    
    **📁 File Upload:**
    - PDF, DOCX, TXT, CSV, JSON
    - Image files (PNG, JPG, etc.)
    - Files included in chat context
    - Automatic text extraction
    
    **🎯 Advanced Features:**
    - Session management
    - Export to JSON/Text
    - Quick action templates
    - Auto-detect image requests
    - Token estimation
    - Response timing
    
    **💡 Tips:**
    - Use quick actions for common tasks
    - Upload files to provide context
    - Enable auto-save for important chats
    - Try different models for varied responses
    - Use D Hudson AI preset for personalized assistance
    """)

st.caption("🚀 D Hudson's AI Assistant Pro v4.0 | Built with Streamlit | Powered by Multiple AI Models")
st.caption("💡 Created for enhanced productivity and creative workflows")
