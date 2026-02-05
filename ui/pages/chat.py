"""
Chat Page

Conversational interface for recommendations.
"""

import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("💬 Chat with RecSys")
st.markdown("---")

# Initialize session state
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "chat_session_id" not in st.session_state:
    st.session_state.chat_session_id = None

if "chat_user_id" not in st.session_state:
    st.session_state.chat_user_id = None

# Sidebar for user selection
with st.sidebar:
    st.subheader("Chat Settings")
    
    # Get users for personalization
    try:
        import psycopg2
        DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://recsys:recsys_password@localhost:5432/youtube_recsys")
        
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("""
            SELECT user_id, persona_type 
            FROM users 
            ORDER BY created_at DESC 
            LIMIT 20
        """)
        users = cur.fetchall()
        conn.close()
        
        user_options = {"Anonymous": None}
        user_options.update({f"{u[1]} - {u[0][:8]}...": u[0] for u in users})
        
        selected_user_label = st.selectbox(
            "Chat as user",
            options=list(user_options.keys()),
            help="Select a user for personalized recommendations"
        )
        st.session_state.chat_user_id = user_options[selected_user_label]
        
    except Exception as e:
        st.warning(f"Could not load users: {e}")
        st.session_state.chat_user_id = None
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_messages = []
        st.session_state.chat_session_id = None
        st.rerun()

# Display chat messages
for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask for recommendations or search for videos..."):
    # Add user message
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response from API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/chat/",
                    json={
                        "message": prompt,
                        "session_id": st.session_state.chat_session_id,
                        "user_id": st.session_state.chat_user_id,
                    },
                    timeout=60,
                )
                
                if response.status_code == 200:
                    data = response.json()
                    assistant_response = data.get("response", "Sorry, I couldn't process that.")
                    st.session_state.chat_session_id = data.get("session_id")
                    
                    st.markdown(assistant_response)
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": assistant_response
                    })
                else:
                    error_msg = f"Sorry, I encountered an error (Status: {response.status_code})"
                    st.error(error_msg)
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
                    
            except requests.exceptions.ConnectionError:
                error_msg = "Could not connect to the API. Make sure the server is running."
                st.error(error_msg)
                st.code("uvicorn serving.api.main:app --host 0.0.0.0 --port 8000")
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

# Example prompts
if not st.session_state.chat_messages:
    st.markdown("### 💡 Try asking:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - "Hello!"
        - "Recommend some videos for me"
        - "Find videos about machine learning"
        """)
    
    with col2:
        st.markdown("""
        - "Search for Python tutorials"
        - "What gaming videos do you have?"
        - "Show me music videos"
        """)