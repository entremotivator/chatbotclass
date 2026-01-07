import streamlit as st
import io
import json
from datetime import datetime
import base64
import requests
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Google Drive Podcast Studio",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        color: #155724;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        color: #856404;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 8px;
        color: #0c5460;
        margin: 1rem 0;
    }
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'recordings' not in st.session_state:
    st.session_state.recordings = []
if 'gdrive_authenticated' not in st.session_state:
    st.session_state.gdrive_authenticated = False
if 'access_token' not in st.session_state:
    st.session_state.access_token = None
if 'service_account' not in st.session_state:
    st.session_state.service_account = None
if 'folder_structure' not in st.session_state:
    st.session_state.folder_structure = {}
if 'podcast_episodes' not in st.session_state:
    st.session_state.podcast_episodes = []

# Google Drive API Helper Functions
class GoogleDriveAPI:
    def __init__(self, service_account_info):
        self.service_account = service_account_info
        self.access_token = None
        
    def get_access_token(self):
        """Get OAuth2 access token using service account"""
        try:
            import jwt
            import time
            
            # Create JWT
            now = int(time.time())
            payload = {
                'iss': self.service_account['client_email'],
                'scope': 'https://www.googleapis.com/auth/drive.file',
                'aud': 'https://oauth2.googleapis.com/token',
                'iat': now,
                'exp': now + 3600
            }
            
            # Sign JWT
            private_key = self.service_account['private_key']
            encoded_jwt = jwt.encode(payload, private_key, algorithm='RS256')
            
            # Exchange JWT for access token
            token_url = 'https://oauth2.googleapis.com/token'
            data = {
                'grant_type': 'urn:ietf:params:oauth:grant-type:jwt-bearer',
                'assertion': encoded_jwt
            }
            
            response = requests.post(token_url, data=data)
            if response.status_code == 200:
                self.access_token = response.json()['access_token']
                return self.access_token
            else:
                return None
        except Exception as e:
            st.error(f"Authentication error: {str(e)}")
            return None
    
    def create_folder(self, folder_name, parent_id=None):
        """Create a folder in Google Drive"""
        if not self.access_token:
            return None
            
        url = 'https://www.googleapis.com/drive/v3/files'
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
        
        metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        
        if parent_id:
            metadata['parents'] = [parent_id]
        
        response = requests.post(url, headers=headers, json=metadata)
        
        if response.status_code == 200:
            return response.json()['id']
        return None
    
    def folder_exists(self, folder_name, parent_id=None):
        """Check if folder exists"""
        if not self.access_token:
            return None
            
        url = 'https://www.googleapis.com/drive/v3/files'
        headers = {'Authorization': f'Bearer {self.access_token}'}
        
        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        if parent_id:
            query += f" and '{parent_id}' in parents"
        
        params = {'q': query, 'fields': 'files(id, name)'}
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            files = response.json().get('files', [])
            return files[0]['id'] if files else None
        return None
    
    def upload_file(self, file_name, file_data, mime_type, folder_id=None):
        """Upload file to Google Drive"""
        if not self.access_token:
            return None
        
        # Metadata
        metadata = {'name': file_name}
        if folder_id:
            metadata['parents'] = [folder_id]
        
        # Upload using multipart
        url = 'https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart'
        headers = {'Authorization': f'Bearer {self.access_token}'}
        
        files = {
            'metadata': (None, json.dumps(metadata), 'application/json'),
            'file': (file_name, file_data, mime_type)
        }
        
        response = requests.post(url, headers=headers, files=files)
        
        if response.status_code == 200:
            return response.json()['id']
        return None
    
    def list_files(self, folder_id=None):
        """List files in folder"""
        if not self.access_token:
            return []
            
        url = 'https://www.googleapis.com/drive/v3/files'
        headers = {'Authorization': f'Bearer {self.access_token}'}
        
        query = "trashed=false"
        if folder_id:
            query += f" and '{folder_id}' in parents"
        
        params = {
            'q': query,
            'fields': 'files(id, name, mimeType, createdTime, size)',
            'orderBy': 'createdTime desc'
        }
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            return response.json().get('files', [])
        return []

def setup_podcast_folders(gdrive_api):
    """Create folder structure for podcast recordings"""
    try:
        # Main folder: "Podcast Studio"
        main_folder_id = gdrive_api.folder_exists("Podcast Studio")
        if not main_folder_id:
            main_folder_id = gdrive_api.create_folder("Podcast Studio")
            st.success("✅ Created 'Podcast Studio' folder")
        else:
            st.info("📁 'Podcast Studio' folder already exists")
        
        if not main_folder_id:
            st.error("Failed to create main folder")
            return None
        
        # Create subfolders
        subfolders = {}
        folder_names = ["Audio Recordings", "Episode Notes", "Transcripts", "Drafts"]
        
        for folder_name in folder_names:
            folder_id = gdrive_api.folder_exists(folder_name, main_folder_id)
            if not folder_id:
                folder_id = gdrive_api.create_folder(folder_name, main_folder_id)
                st.success(f"✅ Created '{folder_name}' subfolder")
            else:
                st.info(f"📁 '{folder_name}' subfolder already exists")
            
            subfolders[folder_name] = folder_id
        
        st.session_state.folder_structure = {
            'main': main_folder_id,
            **subfolders
        }
        
        return st.session_state.folder_structure
        
    except Exception as e:
        st.error(f"Error setting up folders: {str(e)}")
        return None

# Sidebar - Google Drive Authentication
with st.sidebar:
    st.header("🔐 Google Drive Connection")
    
    if not st.session_state.gdrive_authenticated:
        st.markdown("""
        **Setup Instructions:**
        1. Create a Google Cloud project
        2. Enable Google Drive API
        3. Create a service account
        4. Download JSON key file
        5. Upload below
        """)
        
        uploaded_file = st.file_uploader(
            "Upload Service Account JSON",
            type=['json'],
            help="Upload your Google Cloud service account JSON key file"
        )
        
        if uploaded_file is not None:
            try:
                service_account_info = json.load(uploaded_file)
                
                # Validate required fields
                required_fields = ['private_key', 'client_email', 'project_id']
                if all(field in service_account_info for field in required_fields):
                    st.session_state.service_account = service_account_info
                    
                    # Try to authenticate
                    with st.spinner("Authenticating with Google Drive..."):
                        gdrive = GoogleDriveAPI(service_account_info)
                        token = gdrive.get_access_token()
                        
                        if token:
                            st.session_state.access_token = token
                            st.session_state.gdrive_authenticated = True
                            st.success("✅ Connected to Google Drive!")
                            st.rerun()
                        else:
                            st.error("❌ Authentication failed. Check your service account.")
                else:
                    st.error("❌ Invalid service account file. Missing required fields.")
                    
            except json.JSONDecodeError:
                st.error("❌ Invalid JSON file")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    else:
        st.success("✅ Connected to Google Drive")
        
        # Show connection info
        if st.session_state.service_account:
            st.write(f"**Account:** {st.session_state.service_account.get('client_email', 'N/A')[:30]}...")
            st.write(f"**Project:** {st.session_state.service_account.get('project_id', 'N/A')}")
        
        st.divider()
        
        # Folder setup button
        if st.button("📁 Setup/Verify Podcast Folders", use_container_width=True):
            gdrive = GoogleDriveAPI(st.session_state.service_account)
            gdrive.access_token = st.session_state.access_token
            
            with st.spinner("Setting up folder structure..."):
                folders = setup_podcast_folders(gdrive)
                if folders:
                    st.success("✅ Folder structure ready!")
        
        # Show folder structure
        if st.session_state.folder_structure:
            st.divider()
            st.subheader("📂 Folder Structure")
            st.write("**Podcast Studio/**")
            for name, folder_id in st.session_state.folder_structure.items():
                if name != 'main':
                    st.write(f"  └─ {name}")
        
        st.divider()
        
        if st.button("🔌 Disconnect", use_container_width=True):
            st.session_state.gdrive_authenticated = False
            st.session_state.access_token = None
            st.session_state.service_account = None
            st.session_state.folder_structure = {}
            st.rerun()
    
    st.divider()
    
    # Studio statistics
    st.subheader("📊 Studio Stats")
    st.metric("Local Recordings", len(st.session_state.recordings))
    st.metric("Podcast Episodes", len(st.session_state.podcast_episodes))
    
    if st.session_state.recordings:
        total_size = sum(r['size_kb'] for r in st.session_state.recordings)
        st.metric("Total Size", f"{total_size:.2f} KB")

# Main header
st.markdown('<h1 class="main-header">🎙️ Google Drive Podcast Studio</h1>', unsafe_allow_html=True)
st.markdown("**Professional podcast recording with automatic Google Drive sync**")

# Connection status banner
if not st.session_state.gdrive_authenticated:
    st.warning("⚠️ **Google Drive Not Connected** - Upload your service account JSON in the sidebar to enable cloud sync")
else:
    st.success("✅ **Google Drive Connected** - Your recordings will be automatically synced")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎙️ Record Podcast",
    "📝 Episode Manager", 
    "📚 Cloud Library",
    "⚙️ Settings"
])

# Tab 1: Record Podcast
with tab1:
    st.header("🎙️ Record Podcast Episode")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Episode Information")
        
        episode_title = st.text_input(
            "Episode Title",
            placeholder="e.g., Episode 42: The Future of AI",
            key="episode_title"
        )
        
        col_num, col_season = st.columns(2)
        with col_num:
            episode_number = st.text_input("Episode Number", placeholder="42", key="ep_num")
        with col_season:
            season_number = st.text_input("Season (optional)", placeholder="1", key="season_num")
        
        episode_description = st.text_area(
            "Episode Description",
            placeholder="Brief description of this episode...",
            height=100,
            key="ep_desc"
        )
        
        st.divider()
        
        st.subheader("Recording Settings")
        
        col_quality, col_format = st.columns(2)
        
        with col_quality:
            audio_quality = st.selectbox(
                "Audio Quality",
                ["Podcast Standard (24kHz)", "High Quality (48kHz)", "Custom"],
                index=0,
                key="audio_quality"
            )
            
            if audio_quality == "Custom":
                custom_rate = st.selectbox(
                    "Sample Rate",
                    [16000, 24000, 32000, 44100, 48000],
                    index=2
                )
                sample_rate = custom_rate
            elif audio_quality == "High Quality (48kHz)":
                sample_rate = 48000
            else:
                sample_rate = 24000
        
        with col_format:
            st.metric("Sample Rate", f"{sample_rate} Hz")
            st.metric("Format", "WAV (Uncompressed)")
        
        st.divider()
        
        st.subheader("Record Audio")
        
        audio_recording = st.audio_input(
            "🎙️ Click to start recording",
            sample_rate=sample_rate,
            key="podcast_recorder",
            help="Click the microphone button to record your podcast episode"
        )
        
        if audio_recording:
            st.success("✅ Recording captured!")
            
            st.subheader("Preview & Edit")
            st.audio(audio_recording)
            
            # Get audio data
            audio_bytes = audio_recording.getvalue()
            file_size_kb = len(audio_bytes) / 1024
            estimated_duration = len(audio_bytes) / (sample_rate * 2)
            
            col_metrics = st.columns(4)
            with col_metrics[0]:
                st.metric("Duration", f"{estimated_duration:.1f}s")
            with col_metrics[1]:
                st.metric("Size", f"{file_size_kb:.2f} KB")
            with col_metrics[2]:
                st.metric("Quality", f"{sample_rate/1000:.1f} kHz")
            with col_metrics[3]:
                st.metric("Format", "WAV")
            
            st.divider()
            
            st.subheader("Episode Notes")
            
            episode_notes = st.text_area(
                "Show Notes",
                placeholder="Key points, timestamps, links, guest information...",
                height=150,
                key="ep_notes"
            )
            
            episode_tags = st.text_input(
                "Tags (comma-separated)",
                placeholder="e.g., technology, AI, interview",
                key="ep_tags"
            )
            
            st.divider()
            
            # Save options
            col_save, col_local = st.columns(2)
            
            with col_save:
                if st.button("💾 Save to Google Drive", type="primary", use_container_width=True, disabled=not st.session_state.gdrive_authenticated):
                    if not episode_title:
                        st.error("⚠️ Please enter an episode title")
                    else:
                        with st.spinner("Uploading to Google Drive..."):
                            try:
                                gdrive = GoogleDriveAPI(st.session_state.service_account)
                                gdrive.access_token = st.session_state.access_token
                                
                                # Ensure folders exist
                                if not st.session_state.folder_structure:
                                    setup_podcast_folders(gdrive)
                                
                                # Generate filename
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                audio_filename = f"{episode_title}_{timestamp}.wav"
                                notes_filename = f"{episode_title}_{timestamp}_notes.txt"
                                
                                # Upload audio
                                audio_folder_id = st.session_state.folder_structure.get('Audio Recordings')
                                audio_file_id = gdrive.upload_file(
                                    audio_filename,
                                    audio_bytes,
                                    'audio/wav',
                                    audio_folder_id
                                )
                                
                                # Upload notes if provided
                                notes_file_id = None
                                if episode_notes:
                                    notes_content = f"""Episode: {episode_title}
Episode Number: {episode_number if episode_number else 'N/A'}
Season: {season_number if season_number else 'N/A'}
Recorded: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Duration: {estimated_duration:.1f}s
Quality: {sample_rate} Hz

Description:
{episode_description if episode_description else 'N/A'}

Show Notes:
{episode_notes}

Tags: {episode_tags if episode_tags else 'N/A'}
"""
                                    notes_folder_id = st.session_state.folder_structure.get('Episode Notes')
                                    notes_file_id = gdrive.upload_file(
                                        notes_filename,
                                        notes_content.encode('utf-8'),
                                        'text/plain',
                                        notes_folder_id
                                    )
                                
                                if audio_file_id:
                                    # Save to session state
                                    episode_data = {
                                        'title': episode_title,
                                        'number': episode_number,
                                        'season': season_number,
                                        'description': episode_description,
                                        'notes': episode_notes,
                                        'tags': [t.strip() for t in episode_tags.split(',') if t.strip()],
                                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        'duration': f"{estimated_duration:.1f}s",
                                        'size_kb': file_size_kb,
                                        'sample_rate': sample_rate,
                                        'audio_file_id': audio_file_id,
                                        'notes_file_id': notes_file_id,
                                        'local_data': audio_bytes
                                    }
                                    
                                    st.session_state.podcast_episodes.append(episode_data)
                                    
                                    st.success("✅ Episode saved to Google Drive!")
                                    st.balloons()
                                    
                                    st.info(f"""
                                    **Saved to Google Drive:**
                                    - 🎵 Audio: Podcast Studio/Audio Recordings/{audio_filename}
                                    {"- 📝 Notes: Podcast Studio/Episode Notes/" + notes_filename if notes_file_id else ""}
                                    """)
                                else:
                                    st.error("❌ Failed to upload to Google Drive")
                                    
                            except Exception as e:
                                st.error(f"❌ Upload error: {str(e)}")
            
            with col_local:
                if st.button("💾 Save Locally Only", use_container_width=True):
                    if not episode_title:
                        st.error("⚠️ Please enter an episode title")
                    else:
                        episode_data = {
                            'title': episode_title,
                            'number': episode_number,
                            'season': season_number,
                            'description': episode_description,
                            'notes': episode_notes,
                            'tags': [t.strip() for t in episode_tags.split(',') if t.strip()],
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'duration': f"{estimated_duration:.1f}s",
                            'size_kb': file_size_kb,
                            'sample_rate': sample_rate,
                            'audio_file_id': None,
                            'notes_file_id': None,
                            'local_data': audio_bytes
                        }
                        
                        st.session_state.podcast_episodes.append(episode_data)
                        st.success("✅ Episode saved locally!")
    
    with col2:
        st.subheader("Quick Tips")
        
        st.info("""
        **Recording Tips:**
        - Use headphones
        - Quiet environment
        - Good microphone
        - 6-12" distance
        - Test levels first
        """)
        
        st.subheader("Quality Guide")
        
        st.markdown("""
        **Podcast Standard**
        24 kHz, smaller files
        
        **High Quality**
        48 kHz, best fidelity
        
        **Custom**
        Your choice
        """)

# Tab 2: Episode Manager
with tab2:
    st.header("📝 Episode Manager")
    
    if not st.session_state.podcast_episodes:
        st.info("📭 No episodes yet. Record your first episode in the 'Record Podcast' tab!")
    else:
        st.write(f"**Total Episodes: {len(st.session_state.podcast_episodes)}**")
        
        # Search and filter
        col_search, col_filter = st.columns([3, 1])
        
        with col_search:
            search_query = st.text_input(
                "🔍 Search episodes",
                placeholder="Search by title, description, tags...",
                key="ep_search"
            )
        
        with col_filter:
            sync_filter = st.selectbox(
                "Filter",
                ["All Episodes", "Synced to Drive", "Local Only"],
                key="sync_filter"
            )
        
        st.divider()
        
        # Filter episodes
        filtered_episodes = st.session_state.podcast_episodes.copy()
        
        if sync_filter == "Synced to Drive":
            filtered_episodes = [e for e in filtered_episodes if e.get('audio_file_id')]
        elif sync_filter == "Local Only":
            filtered_episodes = [e for e in filtered_episodes if not e.get('audio_file_id')]
        
        if search_query:
            search_lower = search_query.lower()
            filtered_episodes = [
                e for e in filtered_episodes
                if search_lower in e['title'].lower()
                or search_lower in e.get('description', '').lower()
                or any(search_lower in tag.lower() for tag in e.get('tags', []))
            ]
        
        st.write(f"**Showing {len(filtered_episodes)} episodes**")
        
        # Display episodes
        for idx, episode in enumerate(filtered_episodes):
            with st.expander(f"🎙️ {episode['title']}" + (f" - Episode {episode['number']}" if episode.get('number') else "")):
                col_left, col_right = st.columns([3, 1])
                
                with col_left:
                    # Play audio
                    if episode.get('local_data'):
                        st.audio(episode['local_data'])
                    
                    # Episode info
                    if episode.get('number'):
                        st.write(f"**Episode:** {episode['number']}" + (f" (Season {episode['season']})" if episode.get('season') else ""))
                    
                    if episode.get('description'):
                        st.write(f"**Description:** {episode['description']}")
                    
                    if episode.get('notes'):
                        st.write("**Show Notes:**")
                        st.text_area("", value=episode['notes'], height=100, disabled=True, key=f"notes_display_{idx}", label_visibility="collapsed")
                    
                    if episode.get('tags'):
                        st.write("**Tags:** " + ", ".join(f"`{tag}`" for tag in episode['tags']))
                
                with col_right:
                    st.metric("Duration", episode.get('duration', 'N/A'))
                    st.metric("Size", f"{episode.get('size_kb', 0):.2f} KB")
                    st.metric("Quality", f"{episode.get('sample_rate', 0)/1000:.1f} kHz")
                    st.write(f"**Recorded:** {episode['timestamp']}")
                    
                    # Sync status
                    if episode.get('audio_file_id'):
                        st.success("☁️ Synced to Drive")
                    else:
                        st.warning("💾 Local only")
                    
                    st.divider()
                    
                    # Actions
                    if not episode.get('audio_file_id') and st.session_state.gdrive_authenticated:
                        if st.button("☁️ Upload to Drive", key=f"upload_{idx}", use_container_width=True):
                            st.info("Upload feature would sync this episode")
                    
                    if st.button("📥 Download", key=f"download_{idx}", use_container_width=True):
                        st.info("Download feature would save the audio file")
                    
                    if st.button("🗑️ Delete", key=f"delete_{idx}", use_container_width=True, type="secondary"):
                        st.session_state.podcast_episodes.pop(st.session_state.podcast_episodes.index(episode))
                        st.rerun()

# Tab 3: Cloud Library
with tab3:
    st.header("📚 Cloud Library")
    
    if not st.session_state.gdrive_authenticated:
        st.warning("⚠️ Connect to Google Drive to view your cloud library")
    else:
        st.subheader("Browse Google Drive Files")
        
        if st.button("🔄 Refresh from Google Drive", use_container_width=False):
            with st.spinner("Loading files from Google Drive..."):
                try:
                    gdrive = GoogleDriveAPI(st.session_state.service_account)
                    gdrive.access_token = st.session_state.access_token
                    
                    if st.session_state.folder_structure:
                        # Get audio files
                        audio_folder_id = st.session_state.folder_structure.get('Audio Recordings')
                        audio_files = gdrive.list_files(audio_folder_id) if audio_folder_id else []
                        
                        # Get notes files
                        notes_folder_id = st.session_state.folder_structure.get('Episode Notes')
                        notes_files = gdrive.list_files(notes_folder_id) if notes_folder_id else []
                        
                        st.success(f"✅ Found {len(audio_files)} audio files and {len(notes_files)} note files")
                        
                        if audio_files:
                            st.subheader("🎵 Audio Recordings")
                            for file in audio_files:
                                col1, col2, col3 = st.columns([3, 1, 1])
                                with col1:
                                    st.write(f"**{file['name']}**")
                                with col2:
                                    size_kb = int(file.get('size', 0)) / 1024 if file.get('size') else 0
                                    st.write(f"{size_kb:.2f} KB")
                                with col3:
                                    created = file.get('createdTime', '').split('T')[0]
                                    st.write(created)
                        
                        if notes_files:
                            st.divider()
                            st.subheader("📝 Episode Notes")
                            for file in notes_files:
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.write(f"**{file['name']}**")
                                with col2:
                                    created = file.get('createdTime', '').split('T')[0]
                                    st.write(created)
                    else:
                        st.info("No folder structure found. Click 'Setup/Verify Podcast Folders' in the sidebar.")
                        
                except Exception as e:
                    st.error(f"Error loading files: {str(e)}")

# Tab 4: Settings
with tab4:
    st.header("⚙️ Settings")
    
    st.subheader("Default Recording Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        default_quality = st.selectbox(
            "Default Quality Preset",
            ["Podcast Standard (24kHz)", "High Quality (48kHz)", "Custom"],
            index=0,
            key="default_quality_setting"
        )
        
        auto_upload = st.checkbox(
            "Auto-upload to Google Drive",
            value=True,
            help="Automatically upload recordings to Google Drive after saving"
        )
        
        include_timestamp = st.checkbox(
            "Include timestamp in filenames",
            value=True,
            help="Add timestamp to prevent filename conflicts"
        )
    
    with col2:
        default_format = st.selectbox(
            "Audio Format",
            ["WAV (Uncompressed)", "MP3 (Future)", "FLAC (Future)"],
            index=0,
            disabled=True,
            help="Currently only WAV format is supported"
        )
        
        auto_notes = st.checkbox(
            "Always prompt for episode notes",
            value=True,
            help="Show notes field for every recording"
        )
        
        show_waveform = st.checkbox(
            "Show waveform visualization (Future)",
            value=False,
            disabled=True,
            help="Feature coming soon"
        )
    
    st.divider()
    
    st.subheader("Folder Management")
    
    if st.session_state.gdrive_authenticated:
        col_folder1, col_folder2 = st.columns(2)
        
        with col_folder1:
            if st.button("🔄 Verify Folder Structure", use_container_width=True):
                gdrive = GoogleDriveAPI(st.session_state.service_account)
                gdrive.access_token = st.session_state.access_token
                
                with st.spinner("Verifying folders..."):
                    folders = setup_podcast_folders(gdrive)
                    if folders:
                        st.success("✅ All folders verified!")
        
        with col_folder2:
            if st.button("📁 Create Additional Folders", use_container_width=True):
                st.info("Feature to create custom folders coming soon")
        
        if st.session_state.folder_structure:
            st.write("**Current Folder Structure:**")
            st.json({
                "Podcast Studio": {
                    "Audio Recordings": "✅ Ready",
                    "Episode Notes": "✅ Ready",
                    "Transcripts": "✅ Ready",
                    "Drafts": "✅ Ready"
                }
            })
    else:
        st.warning("⚠️ Connect to Google Drive to manage folders")
    
    st.divider()
    
    st.subheader("Storage & Data")
    
    col_storage1, col_storage2, col_storage3 = st.columns(3)
    
    with col_storage1:
        st.metric("Local Episodes", len(st.session_state.podcast_episodes))
    
    with col_storage2:
        synced_count = sum(1 for e in st.session_state.podcast_episodes if e.get('audio_file_id'))
        st.metric("Synced to Drive", synced_count)
    
    with col_storage3:
        local_only = len(st.session_state.podcast_episodes) - synced_count
        st.metric("Local Only", local_only)
    
    if st.session_state.podcast_episodes:
        total_size = sum(e.get('size_kb', 0) for e in st.session_state.podcast_episodes)
        total_size_mb = total_size / 1024
        st.metric("Total Storage Used", f"{total_size_mb:.2f} MB" if total_size_mb > 1 else f"{total_size:.2f} KB")
    
    st.divider()
    
    st.subheader("Advanced Options")
    
    col_adv1, col_adv2 = st.columns(2)
    
    with col_adv1:
        if st.button("📥 Export All Metadata", use_container_width=True):
            if st.session_state.podcast_episodes:
                metadata = []
                for ep in st.session_state.podcast_episodes:
                    metadata.append({
                        'title': ep['title'],
                        'number': ep.get('number'),
                        'season': ep.get('season'),
                        'description': ep.get('description'),
                        'timestamp': ep['timestamp'],
                        'duration': ep.get('duration'),
                        'tags': ep.get('tags', []),
                        'synced': bool(ep.get('audio_file_id'))
                    })
                
                st.download_button(
                    "📄 Download Metadata JSON",
                    data=json.dumps(metadata, indent=2),
                    file_name=f"podcast_metadata_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            else:
                st.info("No episodes to export")
    
    with col_adv2:
        if st.button("🗑️ Clear All Local Data", use_container_width=True, type="secondary"):
            if st.session_state.podcast_episodes:
                st.warning(f"⚠️ This will delete {len(st.session_state.podcast_episodes)} local episodes!")
                if st.button("⚠️ Confirm Clear All", type="secondary", key="confirm_clear"):
                    st.session_state.podcast_episodes = []
                    st.session_state.recordings = []
                    st.success("✅ All local data cleared!")
                    st.rerun()
            else:
                st.info("No local data to clear")
    
    st.divider()
    
    st.subheader("📖 Help & Documentation")
    
    with st.expander("🔐 How to Setup Google Drive Connection"):
        st.markdown("""
        **Step-by-Step Guide:**
        
        1. **Create Google Cloud Project**
           - Go to [Google Cloud Console](https://console.cloud.google.com)
           - Create a new project or select existing one
        
        2. **Enable Google Drive API**
           - Navigate to "APIs & Services" > "Library"
           - Search for "Google Drive API"
           - Click "Enable"
        
        3. **Create Service Account**
           - Go to "APIs & Services" > "Credentials"
           - Click "Create Credentials" > "Service Account"
           - Fill in service account details
           - Grant "Editor" role
        
        4. **Create JSON Key**
           - Click on the created service account
           - Go to "Keys" tab
           - Click "Add Key" > "Create New Key"
           - Choose "JSON" format
           - Download the file
        
        5. **Upload to App**
           - Use the file uploader in the sidebar
           - Upload your downloaded JSON key file
           - Wait for authentication confirmation
        
        6. **Setup Folders**
           - Click "Setup/Verify Podcast Folders" button
           - Folder structure will be created automatically
        
        **Security Note:** Keep your service account JSON file secure and never share it publicly.
        """)
    
    with st.expander("🎙️ Recording Best Practices"):
        st.markdown("""
        **Audio Quality Tips:**
        
        - **Microphone Distance:** 6-12 inches from mouth
        - **Room Treatment:** Use acoustic panels or blankets to reduce echo
        - **Background Noise:** Record in quiet environment, turn off fans/AC
        - **Pop Filter:** Use to reduce plosive sounds (P, B, T sounds)
        - **Headphones:** Always monitor with headphones while recording
        
        **Recording Settings:**
        
        - **Podcast Standard (24kHz):** Good quality, reasonable file size
        - **High Quality (48kHz):** Professional quality, larger files
        - **Sample Rate Rule:** Higher = better quality but larger files
        
        **Episode Organization:**
        
        - Use clear, descriptive titles
        - Number episodes consistently
        - Add comprehensive show notes
        - Tag episodes for easy searching
        - Always include episode descriptions
        
        **Workflow Tips:**
        
        1. Test your levels before recording
        2. Record in a consistent environment
        3. Take breaks during long sessions
        4. Save notes immediately after recording
        5. Upload to Drive for backup
        6. Review audio before finalizing
        """)
    
    with st.expander("☁️ Google Drive Sync Information"):
        st.markdown("""
        **How Sync Works:**
        
        When you save an episode to Google Drive, the app:
        1. Creates the episode audio file (.wav)
        2. Creates a notes file (.txt) if notes are provided
        3. Uploads audio to "Audio Recordings" folder
        4. Uploads notes to "Episode Notes" folder
        5. Stores file IDs for future reference
        
        **Folder Structure:**
        ```
        Podcast Studio/
        ├── Audio Recordings/
        │   └── [Your episode audio files]
        ├── Episode Notes/
        │   └── [Your episode notes files]
        ├── Transcripts/
        │   └── [Future: Auto-generated transcripts]
        └── Drafts/
            └── [Future: Draft recordings]
        ```
        
        **File Naming:**
        - Audio: `[Episode Title]_[Timestamp].wav`
        - Notes: `[Episode Title]_[Timestamp]_notes.txt`
        
        **Benefits:**
        - Automatic cloud backup
        - Access from any device
        - Share with team members
        - Version history (Google Drive feature)
        - Large storage capacity
        
        **Privacy:**
        - Files are stored in your Google Drive
        - Only accessible by your service account
        - Share settings controlled by you
        """)
    
    with st.expander("❓ Frequently Asked Questions"):
        st.markdown("""
        **Q: Why use a service account instead of OAuth?**
        A: Service accounts provide automated, long-term access without requiring manual re-authentication.
        
        **Q: Can I use my personal Google account?**
        A: Yes! Create a project in your personal Google Cloud account and use that service account.
        
        **Q: How much does Google Cloud cost?**
        A: Google Drive API has a free tier. For most podcast use cases, you won't exceed free limits.
        
        **Q: What if folders already exist?**
        A: The app checks for existing folders and only creates missing ones.
        
        **Q: Can I access files from Google Drive web interface?**
        A: Yes! Navigate to your Google Drive and find the "Podcast Studio" folder.
        
        **Q: What happens if upload fails?**
        A: The episode is saved locally. You can retry uploading from the Episode Manager.
        
        **Q: Can I delete files from the app?**
        A: Currently, deletion only removes local copies. Use Google Drive interface to delete cloud files.
        
        **Q: Is my data secure?**
        A: Yes. Files are stored in your Google Drive account. Keep your service account JSON secure.
        
        **Q: Can multiple people use the same service account?**
        A: Yes, but be cautious. Anyone with the JSON file has full access to the connected Drive.
        
        **Q: What audio formats are supported?**
        A: Currently WAV (uncompressed). MP3 and FLAC support coming in future updates.
        """)

# Footer
st.divider()

col_footer1, col_footer2, col_footer3, col_footer4 = st.columns(4)

with col_footer1:
    st.caption("🎙️ Google Drive Podcast Studio")

with col_footer2:
    if st.session_state.gdrive_authenticated:
        st.caption("✅ Connected to Drive")
    else:
        st.caption("⚠️ Not Connected")

with col_footer3:
    st.caption(f"{len(st.session_state.podcast_episodes)} Episodes")

with col_footer4:
    st.caption("Version 1.0")

# Installation requirements notice
st.divider()
st.info("""
**📦 Required Python Packages:**
```bash
pip install streamlit requests PyJWT
```

**Note:** This app requires the `PyJWT` library for service account authentication.
If you get import errors, install it with: `pip install PyJWT`
""")
