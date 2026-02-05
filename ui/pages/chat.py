"""
Chat - Talk to the AI assistant
"""

import streamlit as st
import requests
import psycopg2
import os
from dotenv import load_dotenv

st.set_page_config(page_title="Chat | YouTube RecSys", page_icon="💬", layout="wide")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0e1117 0%, #1a1a2e 100%);
    }
</style>
""", unsafe_allow_html=True)

load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://recsys:recsys_password@localhost:5432/youtube_recsys")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None

# Sidebar
with st.sidebar:
    st.markdown("## 🎬 YouTube RecSys")
    st.caption("Video Recommendation System")
    st.markdown("---")
    st.markdown("### Chat Settings")
    
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("SELECT user_id, persona_type FROM users LIMIT 15")
        users = cur.fetchall()
        conn.close()
        
        user_options = {"Guest": None}
        user_options.update({f"{u[1] or 'User'} ({str(u[0])[:6]}...)": u[0] for u in users})
        
        selected_user = st.selectbox("Chat as", list(user_options.keys()))
        user_id = user_options[selected_user]
    except:
        user_id = None
    
    st.markdown("")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.rerun()

# Main content
st.markdown("# 💬 Chat")
st.caption("Ask for recommendations or search for videos")
st.markdown("---")

# Display messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/chat/",
                    json={
                        "message": prompt, 
                        "session_id": st.session_state.session_id, 
                        "user_id": str(user_id) if user_id else None
                    },
                    timeout=60,
                )
                
                if response.status_code == 200:
                    data = response.json()
                    reply = data.get("response", "Sorry, something went wrong.")
                    st.session_state.session_id = data.get("session_id")
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                else:
                    st.error(f"Error: {response.status_code}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Run: `make start-api`")
            except Exception as e:
                st.error(str(e))

# Empty state
if not st.session_state.messages:
    st.markdown("")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**💡 Try asking:**")
        st.caption("• Hello!")
        st.caption("• Recommend videos for me")
        
    with col2:
        st.markdown("**🔍 Search:**")
        st.caption("• Find Python tutorials")
        st.caption("• Search machine learning")
        
    with col3:
        st.markdown("**❓ Questions:**")
        st.caption("• What can you do?")
        st.caption("• Show me gaming videos")