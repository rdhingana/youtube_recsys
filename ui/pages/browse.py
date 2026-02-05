"""
Browse - Explore the video catalog
"""

import streamlit as st
import psycopg2
import os
from dotenv import load_dotenv

st.set_page_config(page_title="Browse | YouTube RecSys", page_icon="🔍", layout="wide")

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
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://recsys:recsys_password@localhost:5432/youtube_recsys")

with st.sidebar:
    st.markdown("## 🎬 YouTube RecSys")
    st.caption("Video Recommendation System")


def get_thumbnail(video_id: str) -> str:
    return f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"


def format_views(views):
    if not views:
        return "N/A"
    if views >= 1_000_000:
        return f"{views/1_000_000:.1f}M views"
    if views >= 1_000:
        return f"{views/1_000:.1f}K views"
    return f"{views} views"


def format_duration(seconds):
    if not seconds:
        return ""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


st.markdown("# 🔍 Browse Videos")
st.caption("Explore the video catalog")
st.markdown("---")

try:
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    cur.execute("SELECT DISTINCT category_name FROM videos WHERE category_name IS NOT NULL ORDER BY category_name")
    categories = ["All Categories"] + [row[0] for row in cur.fetchall()]
    
    with col1:
        selected_category = st.selectbox("Category", categories)
    with col2:
        sort_by = st.selectbox("Sort by", ["Most Views", "Newest", "Title A-Z"])
    with col3:
        search_query = st.text_input("Search", placeholder="Search videos...")
    
    st.markdown("---")
    
    # Build query
    query = """
        SELECT video_id, title, channel_name, category_name, view_count, duration_seconds 
        FROM videos WHERE is_active = true
    """
    params = []
    
    if selected_category != "All Categories":
        query += " AND category_name = %s"
        params.append(selected_category)
    
    if search_query:
        query += " AND (title ILIKE %s OR channel_name ILIKE %s)"
        params.extend([f"%{search_query}%", f"%{search_query}%"])
    
    sort_map = {
        "Most Views": "view_count DESC NULLS LAST", 
        "Newest": "published_at DESC NULLS LAST", 
        "Title A-Z": "title ASC"
    }
    query += f" ORDER BY {sort_map[sort_by]} LIMIT 60"
    
    cur.execute(query, params)
    videos = cur.fetchall()
    conn.close()
    
    st.caption(f"Showing {len(videos)} videos")
    st.markdown("")
    
    if videos:
        # Display in rows of 4
        for row_idx in range(0, len(videos), 4):
            cols = st.columns(4)
            for col_idx, col in enumerate(cols):
                video_idx = row_idx + col_idx
                if video_idx < len(videos):
                    video_id, title, channel, category, views, duration = videos[video_idx]
                    
                    with col:
                        # Thumbnail
                        st.image(get_thumbnail(video_id), use_container_width=True)
                        
                        # Title (2 lines max)
                        display_title = title[:60] + "..." if len(title) > 60 else title
                        st.markdown(f"**{display_title}**")
                        
                        # Channel
                        st.caption(f"📺 {channel or 'Unknown'}")
                        
                        # Views and duration
                        meta = format_views(views)
                        if duration:
                            meta += f" · {format_duration(duration)}"
                        st.caption(meta)
                        
                        # Category
                        if category:
                            st.caption(f"🏷️ {category}")
                        
                        st.markdown("")  # Spacer between rows
    else:
        st.info("No videos found matching your criteria.")

except Exception as e:
    st.error(f"Database error: {e}")