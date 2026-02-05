"""
Recommendations - Get personalized video suggestions
"""

import streamlit as st
import requests
import psycopg2
import pandas as pd
import os
from dotenv import load_dotenv

st.set_page_config(page_title="Recommendations | YouTube RecSys", page_icon="🎯", layout="wide")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0e1117 0%, #1a1a2e 100%);
    }
    .video-card img {
        border-radius: 8px;
        width: 100%;
        aspect-ratio: 16/9;
        object-fit: cover;
    }
</style>
""", unsafe_allow_html=True)

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://recsys:recsys_password@localhost:5432/youtube_recsys")

with st.sidebar:
    st.markdown("## 🎬 YouTube RecSys")
    st.caption("Video Recommendation System")


def get_thumbnail(video_id: str) -> str:
    """Get YouTube thumbnail - try multiple quality levels"""
    return f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"


st.markdown("# 🎯 Recommendations")
st.caption("Get personalized video suggestions based on user preferences")
st.markdown("---")

col1, col2 = st.columns([2, 1])

selected_user_id = None

with col1:
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("""
            SELECT u.user_id, u.persona_type, COUNT(ui.interaction_id) as interactions
            FROM users u
            LEFT JOIN user_interactions ui ON u.user_id = ui.user_id
            GROUP BY u.user_id, u.persona_type
            ORDER BY interactions DESC
            LIMIT 50
        """)
        users = cur.fetchall()
        conn.close()
        
        if users:
            user_options = {f"{u[1] or 'Unknown'} ({u[2]:,} interactions)": u[0] for u in users}
            selected_user_label = st.selectbox("Select a user profile", options=list(user_options.keys()))
            selected_user_id = user_options[selected_user_label]
            
    except Exception as e:
        st.error(f"Database error: {e}")

with col2:
    num_recommendations = st.slider("Number of results", 4, 48, 20, step=4)
    exclude_watched = st.checkbox("Exclude watched videos", value=False)

st.markdown("---")

if selected_user_id:
    if st.button("🚀 Get Recommendations", type="primary", use_container_width=True):
        with st.spinner("Generating recommendations..."):
            try:
                response = requests.post(
                    f"{API_URL}/recommend",
                    json={
                        "user_id": selected_user_id, 
                        "num_recommendations": num_recommendations, 
                        "exclude_watched": exclude_watched
                    },
                    timeout=30,
                )
                
                if response.status_code == 200:
                    data = response.json()
                    recommendations = data.get("recommendations", [])
                    
                    # Performance metrics
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Retrieval", f"{data.get('retrieval_time_ms', 0):.1f}ms")
                    c2.metric("Ranking", f"{data.get('ranking_time_ms', 0):.1f}ms")
                    c3.metric("Re-ranking", f"{data.get('reranking_time_ms', 0):.1f}ms")
                    c4.metric("Total", f"{data.get('total_time_ms', 0):.1f}ms")
                    
                    st.markdown("---")
                    st.markdown(f"### 📺 {len(recommendations)} Results")
                    
                    if recommendations:
                        for row_idx in range(0, len(recommendations), 4):
                            cols = st.columns(4)
                            for col_idx, col in enumerate(cols):
                                video_idx = row_idx + col_idx
                                if video_idx < len(recommendations):
                                    video = recommendations[video_idx]
                                    with col:
                                        # Thumbnail
                                        thumb_url = get_thumbnail(video.get("video_id", ""))
                                        st.image(thumb_url, use_container_width=True)
                                        
                                        # Title
                                        title = video.get("title", "Unknown")[:60]
                                        if len(video.get("title", "")) > 60:
                                            title += "..."
                                        st.markdown(f"**{title}**")
                                        
                                        # Meta
                                        st.caption(f"📺 {video.get('channel_name', 'Unknown')}")
                                        st.caption(f"⭐ {video.get('score', 0):.3f} · 🏷️ {video.get('category_name', 'N/A')}")
                                        st.markdown("")
                    else:
                        st.info("No recommendations found.")
                else:
                    st.error(f"API Error: {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Run: `make start-api`")
            except Exception as e:
                st.error(f"Error: {e}")

# Watch history expander
if selected_user_id:
    with st.expander("📜 View Watch History"):
        try:
            conn = psycopg2.connect(DATABASE_URL)
            cur = conn.cursor()
            cur.execute("""
                SELECT v.title, v.channel_name, ui.interaction_type, ui.watch_percentage
                FROM user_interactions ui 
                JOIN videos v ON ui.video_id = v.video_id
                WHERE ui.user_id = %s 
                ORDER BY ui.created_at DESC LIMIT 20
            """, (selected_user_id,))
            history = cur.fetchall()
            conn.close()
            
            if history:
                df = pd.DataFrame(history, columns=["Title", "Channel", "Type", "Watch %"])
                df["Title"] = df["Title"].apply(lambda x: x[:40] + "..." if len(str(x)) > 40 else x)
                df["Watch %"] = df["Watch %"].apply(lambda x: f"{x*100:.0f}%" if x else "-")
                st.dataframe(df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Error: {e}")