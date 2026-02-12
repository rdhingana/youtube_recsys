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
    
    .tech-badge {
        display: inline-block;
        background: rgba(99, 102, 241, 0.2);
        color: #a5b4fc;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 13px;
        margin: 4px 4px 4px 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🎬 YouTube RecSys")
    st.caption("Video Recommendation System")
    st.markdown("---")

# Hero
st.markdown("# 🎬 YouTube RecSys")
st.markdown("##### Production-Grade Video Recommendation System")
st.markdown("")

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
    st.warning("⚠️ Database not connected. Run `make start-db`")

st.markdown("---")

# Architecture
st.markdown("### 🏗️ How It Works")
st.markdown("")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**1️⃣ Retrieval**")
    st.caption("Two-Tower neural network with FAISS finds candidates from millions of videos in <10ms")

with col2:
    st.markdown("**2️⃣ Ranking**")
    st.caption("Deep Cross Network scores candidates using user behavior and video features")

with col3:
    st.markdown("**3️⃣ Re-ranking**")
    st.caption("Diversity optimization ensures varied, engaging recommendations")

st.markdown("---")

# Tech Stack
st.markdown("### 🛠️ Built With")

tech_items = [
    "PyTorch", "FAISS", "CLIP", "Transformers", 
    "FastAPI", "PostgreSQL", "Streamlit", "Airflow", 
    "Prometheus", "Grafana", "Docker", "Ollama"
]

badges_html = "".join([f'<span class="tech-badge">{tech}</span>' for tech in tech_items])
st.markdown(badges_html, unsafe_allow_html=True)

st.markdown("")
st.markdown("---")

# Quick Actions
st.markdown("### 🚀 Get Started")
st.markdown("")

col1, col2, col3 = st.columns(3)

with col1:
    st.page_link("pages/1_🎯_Recommendations.py", label="🎯 Get Recommendations", use_container_width=True)

with col2:
    st.page_link("pages/2_🔍_Browse.py", label="🔍 Browse Videos", use_container_width=True)

with col3:
    st.page_link("pages/3_💬_Chat.py", label="💬 Chat with AI", use_container_width=True)
