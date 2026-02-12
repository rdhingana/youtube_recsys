"""
Chat - Talk to the AI assistant
Clean UI with Voice Support
"""

import streamlit as st
import requests
import psycopg2
import os
import tempfile
from dotenv import load_dotenv

st.set_page_config(page_title="Chat | YouTube RecSys", page_icon="💬", layout="wide")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0e1117 0%, #16213e 100%);
    }
    [data-testid="stSidebarNav"] a {
        padding: 0.6rem 1rem;
        border-radius: 8px;
        color: #e0e0e0 !important;
        font-weight: 500;
    }
    [data-testid="stSidebarNav"] a:hover {
        background: rgba(99, 102, 241, 0.2);
    }
    [data-testid="stSidebarNav"] a[aria-selected="true"] {
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        color: #ffffff !important;
    }
    
    /* User message container - right aligned */
    .user-message {
        display: flex;
        justify-content: flex-end;
        margin: 8px 0;
    }
    
    /* User message content */
    .user-message [data-testid="stChatMessageContent"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        border-radius: 12px;
        padding: 12px 16px !important;
        max-width: 80%;
        margin-left: auto;
    }
    
    /* Assistant message container - left aligned */
    .assistant-message {
        display: flex;
        justify-content: flex-start;
        margin: 8px 0;
    }
    
    /* Assistant message content */
    .assistant-message [data-testid="stChatMessageContent"] {
        background: rgba(30, 30, 46, 0.6) !important;
        border-radius: 12px;
        padding: 12px 16px !important;
        max-width: 80%;
    }
</style>

<script>
// Function to style chat messages
function styleMessages() {
    const messages = document.querySelectorAll('[data-testid="stChatMessage"]');
    messages.forEach(msg => {
        // Check if this is a user message by looking for the avatar alt text
        const avatar = msg.querySelector('img');
        if (avatar) {
            if (avatar.alt === 'user') {
                msg.classList.add('user-message');
                msg.classList.remove('assistant-message');
            } else if (avatar.alt === 'assistant') {
                msg.classList.add('assistant-message');
                msg.classList.remove('user-message');
            }
        }
    });
}

// Run on page load
styleMessages();

// Run whenever the page updates
const observer = new MutationObserver(styleMessages);
observer.observe(document.body, { childList: true, subtree: true });
</script>
""", unsafe_allow_html=True)

load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://recsys:recsys_password@localhost:5432/youtube_recsys")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "tts_enabled" not in st.session_state:
    st.session_state.tts_enabled = False
if "last_response" not in st.session_state:
    st.session_state.last_response = ""
if "voice_input" not in st.session_state:
    st.session_state.voice_input = ""
if "whisper_model" not in st.session_state:
    st.session_state.whisper_model = None
if "pending_message" not in st.session_state:
    st.session_state.pending_message = None

# Sidebar
with st.sidebar:
    st.markdown("### 🎬 YouTube RecSys")
    st.caption("Video Recommendation System")
    st.markdown("---")
    
    st.markdown("#### 👤 User")
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("SELECT user_id, persona_type FROM users LIMIT 15")
        users = cur.fetchall()
        conn.close()
        
        user_options = {"Guest": None}
        user_options.update({f"{u[1] or 'User'} ({str(u[0])[:6]}...)": u[0] for u in users})
        
        selected_user = st.selectbox("Chat as", list(user_options.keys()), label_visibility="collapsed")
        user_id = user_options[selected_user]
    except:
        user_id = None
    
    st.markdown("---")
    st.markdown("#### 📊 Voice")
    st.session_state.tts_enabled = st.toggle("Read responses aloud", value=st.session_state.tts_enabled)
    
    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.session_state.last_response = ""
        st.session_state.pending_message = None
        st.rerun()


def get_tts_html(text):
    """Generate HTML/JS for text-to-speech"""
    escaped_text = text.replace("\\", "\\\\").replace("'", "\\'").replace("\n", " ").replace("\r", "").replace('"', '\\"')
    return f"""
    <script>
        const utterance = new SpeechSynthesisUtterance("{escaped_text}");
        utterance.rate = 1.0;
        speechSynthesis.speak(utterance);
    </script>
    """


@st.cache_resource
def load_whisper_model():
    """Load local Whisper model (cached)"""
    try:
        import whisper
        model = whisper.load_model("base")
        return model, None
    except ImportError:
        return None, "Whisper not installed"
    except Exception as e:
        return None, str(e)


def transcribe_with_local_whisper(audio_bytes):
    """Transcribe audio using local Whisper model"""
    try:
        import whisper
        
        if st.session_state.whisper_model is None:
            model, error = load_whisper_model()
            if error:
                return None, error
            st.session_state.whisper_model = model
        
        model = st.session_state.whisper_model
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name
        
        result = model.transcribe(temp_path, fp16=False)
        os.unlink(temp_path)
        
        return result["text"].strip(), None
        
    except ImportError:
        return None, "Install: pip install openai-whisper"
    except Exception as e:
        return None, str(e)


def transcribe_with_browser():
    """Fallback: Use browser's speech recognition"""
    return """
    <script>
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
        const recognition = new SR();
        recognition.lang = 'en-US';
        recognition.onresult = (e) => {
            const text = e.results[0][0].transcript;
            navigator.clipboard.writeText(text);
            alert('✅ "' + text + '"\\n\\nCopied! Paste in the input box.');
        };
        recognition.onerror = (e) => alert('Error: ' + e.error);
        recognition.start();
    } else {
        alert('Use Chrome or Edge for speech recognition.');
    }
    </script>
    """


def send_message_to_api(message, user_id, session_id):
    """Send message to API and get response"""
    try:
        response = requests.post(
            f"{API_URL}/chat/",
            json={
                "message": message,
                "session_id": session_id,
                "user_id": str(user_id) if user_id else None
            },
            timeout=60,
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("response", "Sorry, something went wrong."), data.get("session_id"), None
        else:
            return None, None, f"API Error: {response.status_code}"
            
    except requests.exceptions.ConnectionError:
        return None, None, "Cannot connect to API. Make sure the API is running with: `make start-api`"
    except Exception as e:
        return None, None, str(e)


# Check availability
audio_recorder_available = False
try:
    from audio_recorder_streamlit import audio_recorder
    audio_recorder_available = True
except ImportError:
    pass

whisper_available = False
try:
    import whisper
    whisper_available = True
except ImportError:
    pass


# ============ MAIN CONTENT ============

st.markdown("# 💬 Chat")
st.caption("Ask for video recommendations using text or voice")

# Quick suggestion tags (always visible at top)
if not st.session_state.messages:
    st.markdown("")
    st.markdown("**Try asking:**")
    
    cols = st.columns(4)
    
    suggestions = [
        ("👋", "Hello!"),
        ("🎯", "Recommend videos for me"),
        ("📚", "Find Python tutorials"),
        ("🎮", "Show gaming videos")
    ]
    
    for idx, (col, (icon, text)) in enumerate(zip(cols, suggestions)):
        with col:
            if st.button(f"{icon} {text}", use_container_width=True, key=f"sug_{idx}"):
                st.session_state.pending_message = text
                st.rerun()

st.markdown("---")

# Chat messages area
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# TTS controls (only show if there's a last response)
if st.session_state.last_response and st.session_state.messages:
    st.markdown("")
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("🔊 Play", use_container_width=True):
            st.components.v1.html(get_tts_html(st.session_state.last_response[:500]), height=0)
    with col2:
        if st.button("🔇 Stop", use_container_width=True):
            st.components.v1.html("<script>speechSynthesis.cancel();</script>", height=0)
    
    st.markdown("---")

# Input area - Using chat_input instead of form
col1, col2 = st.columns([6, 1])

with col1:
    # Use st.chat_input for better UX (Enter to send, no button needed)
    user_input = st.chat_input(
        placeholder="Type your message or click 🎤 to speak...",
        key="chat_input_widget"
    )
    
    if user_input and user_input.strip():
        st.session_state.pending_message = user_input.strip()

with col2:
    if audio_recorder_available and whisper_available:
        audio_bytes = audio_recorder(
            text="",
            recording_color="#6366f1",
            neutral_color="#4a4a5a",
            icon_name="microphone",
            icon_size="lg",
            pause_threshold=2.5,
            sample_rate=16000,
            key="voice_recorder"
        )
        
        if audio_bytes:
            with st.spinner("🎤 Transcribing..."):
                text, error = transcribe_with_local_whisper(audio_bytes)
                if text:
                    st.session_state.voice_input = text
                    st.rerun()
                elif error:
                    st.warning(f"⚠️ {error}")
    else:
        if st.button("🎤", use_container_width=True, help="Voice input"):
            st.components.v1.html(transcribe_with_browser(), height=0)

# Voice transcription result
if st.session_state.voice_input:
    st.success(f"🎤 **Heard:** {st.session_state.voice_input}")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Send", type="primary", use_container_width=True):
            st.session_state.pending_message = st.session_state.voice_input
            st.session_state.voice_input = ""
            st.rerun()
    with col2:
        if st.button("❌ Cancel", use_container_width=True):
            st.session_state.voice_input = ""
            st.rerun()

# Process pending message
if st.session_state.pending_message:
    message_to_send = st.session_state.pending_message
    st.session_state.pending_message = None
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": message_to_send})
    
    # Get AI response
    with st.chat_message("user"):
        st.markdown(message_to_send)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply, new_session_id, error = send_message_to_api(
                message_to_send, 
                user_id, 
                st.session_state.session_id
            )
            
            if error:
                st.error(error)
                # Still add error to messages for context
                st.session_state.messages.append({"role": "assistant", "content": f"❌ {error}"})
            elif reply:
                st.session_state.session_id = new_session_id
                st.session_state.last_response = reply
                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.markdown(reply)
                
                # Auto-play TTS if enabled
                if st.session_state.tts_enabled:
                    st.components.v1.html(get_tts_html(reply[:500]), height=0)
    
    st.rerun()

# Footer status
st.markdown("")
st.markdown("")
voice_status = "🎤 Whisper" if whisper_available else "🌐 Browser"
tts_status = "🔊 On" if st.session_state.tts_enabled else "🔇 Off"
user_status = f"👤 {selected_user}" if 'selected_user' in locals() else "👤 Guest"
st.caption(f"{user_status} • {voice_status} • TTS {tts_status}")
