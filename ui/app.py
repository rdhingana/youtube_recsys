"""
YouTube RecSys - Streamlit UI

Main application entry point.
"""

import streamlit as st

st.set_page_config(
    page_title="YouTube RecSys",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .video-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 10px;
        margin: 5px;
        background-color: #f9f9f9;
    }
    .video-title {
        font-weight: bold;
        font-size: 14px;
        margin-bottom: 5px;
    }
    .video-channel {
        color: #666;
        font-size: 12px;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🎬 YouTube RecSys")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "🎯 Recommendations", "🔍 Browse Videos", "💬 Chat", "📊 Analytics"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "A production-grade video recommendation system with:\n"
    "- Two-tower retrieval model\n"
    "- FAISS similarity search\n"
    "- Deep ranking models\n"
    "- LLM-powered chatbot"
)

# Main content based on navigation
if page == "🏠 Home":
    st.title("🎬 YouTube Video Recommendation System")
    st.markdown("---")
    
    st.markdown("""
    Welcome to the YouTube RecSys demo! This system demonstrates a production-grade 
    video recommendation pipeline with the following components:
    
    ### 🔧 System Architecture
    
    1. **Data Pipeline** - Scrapes videos and simulates user behavior
    2. **Feature Engineering** - CLIP + Sentence Transformers for embeddings
    3. **Two-Tower Retrieval** - Fast candidate generation with FAISS
    4. **Ranking Model** - Deep Cross Network for scoring
    5. **Re-ranking** - Diversity and business rules
    6. **Chatbot** - LLM-powered conversational interface
    
    ### 🚀 Getting Started
    
    Use the sidebar to navigate:
    - **Recommendations** - Get personalized video recommendations
    - **Browse Videos** - Explore the video catalog
    - **Chat** - Talk to the AI assistant
    - **Analytics** - View system statistics
    """)
    
    # Quick stats
    st.markdown("### 📈 Quick Stats")
    
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
        col1.metric("Videos", f"{video_count:,}")
        col2.metric("Users", f"{user_count:,}")
        col3.metric("Interactions", f"{interaction_count:,}")
        col4.metric("Embeddings", f"{embedding_count:,}")
        
    except Exception as e:
        st.warning(f"Could not connect to database: {e}")
        st.info("Make sure PostgreSQL is running and the database is initialized.")

elif page == "🎯 Recommendations":
    exec(open("ui/pages/recommendations.py").read())

elif page == "🔍 Browse Videos":
    exec(open("ui/pages/browse.py").read())

elif page == "💬 Chat":
    exec(open("ui/pages/chat.py").read())

elif page == "📊 Analytics":
    exec(open("ui/pages/analytics.py").read())