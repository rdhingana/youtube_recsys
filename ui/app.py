"""
YouTube RecSys - Home
"""

import streamlit as st

st.set_page_config(
    page_title="YouTube RecSys",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0e1117 0%, #1a1a2e 100%);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🎬 YouTube RecSys")
    st.caption("Video Recommendation System")

# Main
st.markdown("# 🎬 YouTube RecSys")
st.caption("Production-Grade Video Recommendation System")
st.markdown("---")

# Stats
try:
    import psycopg2
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://recsys:recsys_password@localhost:5432/youtube_recsys")
    
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    cur.execute("SELECT COUNT(*) FROM videos WHERE is_active = true")
    video_count = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM users")
    user_count = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM user_interactions")
    interaction_count = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM video_embeddings")
    embedding_count = cur.fetchone()[0]
    
    conn.close()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📹 Videos", f"{video_count:,}")
    col2.metric("👥 Users", f"{user_count:,}")
    col3.metric("🎯 Interactions", f"{interaction_count:,}")
    col4.metric("🧠 Embeddings", f"{embedding_count:,}")
    
except Exception as e:
    st.warning("Database not connected. Run `make start-db`")

st.markdown("---")

# Features
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("### 🎯 Recommendations")
    st.caption("Personalized suggestions using two-tower retrieval + deep ranking")

with col2:
    st.markdown("### 🔍 Browse")
    st.caption("Explore the catalog with filters and search")

with col3:
    st.markdown("### 💬 Chat")
    st.caption("AI assistant powered by local LLM")

with col4:
    st.markdown("### 📊 Analytics")
    st.caption("System metrics and statistics")