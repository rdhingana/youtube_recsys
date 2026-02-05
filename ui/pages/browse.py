"""
Browse Videos Page

Explore the video catalog.
"""

import streamlit as st
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://recsys:recsys_password@localhost:5432/youtube_recsys")

st.title("🔍 Browse Videos")
st.markdown("---")

# Filters
col1, col2, col3 = st.columns(3)

try:
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    # Get categories
    cur.execute("SELECT DISTINCT category_name FROM videos WHERE category_name IS NOT NULL ORDER BY category_name")
    categories = ["All"] + [row[0] for row in cur.fetchall()]
    
    with col1:
        selected_category = st.selectbox("Category", categories)
    
    with col2:
        sort_by = st.selectbox("Sort by", ["Most Views", "Newest", "Title A-Z"])
    
    with col3:
        search_query = st.text_input("Search", placeholder="Search videos...")
    
    st.markdown("---")
    
    # Build query
    query = """
        SELECT video_id, title, channel_name, category_name, 
               thumbnail_url, view_count, duration_seconds, published_at
        FROM videos
        WHERE is_active = true
    """
    params = []
    
    if selected_category != "All":
        query += " AND category_name = %s"
        params.append(selected_category)
    
    if search_query:
        query += " AND (title ILIKE %s OR channel_name ILIKE %s)"
        params.extend([f"%{search_query}%", f"%{search_query}%"])
    
    # Sort
    if sort_by == "Most Views":
        query += " ORDER BY view_count DESC NULLS LAST"
    elif sort_by == "Newest":
        query += " ORDER BY published_at DESC NULLS LAST"
    else:
        query += " ORDER BY title ASC"
    
    query += " LIMIT 50"
    
    cur.execute(query, params)
    videos = cur.fetchall()
    
    # Get total count
    count_query = "SELECT COUNT(*) FROM videos WHERE is_active = true"
    if selected_category != "All":
        count_query += f" AND category_name = '{selected_category}'"
    cur.execute(count_query)
    total_count = cur.fetchone()[0]
    
    conn.close()
    
    st.caption(f"Showing {len(videos)} of {total_count} videos")
    
    # Display videos
    if videos:
        cols = st.columns(4)
        for idx, video in enumerate(videos):
            video_id, title, channel, category, thumbnail, views, duration, published = video
            
            with cols[idx % 4]:
                with st.container():
                    # Thumbnail
                    if thumbnail:
                        st.image(thumbnail, use_container_width=True)
                    else:
                        st.image("https://via.placeholder.com/320x180?text=No+Thumbnail", use_container_width=True)
                    
                    # Title
                    display_title = title[:47] + "..." if len(title) > 50 else title
                    st.markdown(f"**{display_title}**")
                    
                    # Channel
                    st.caption(f"📺 {channel or 'Unknown'}")
                    
                    # Stats
                    if views:
                        if views >= 1000000:
                            views_str = f"{views/1000000:.1f}M"
                        elif views >= 1000:
                            views_str = f"{views/1000:.1f}K"
                        else:
                            views_str = str(views)
                        st.caption(f"👁️ {views_str} views")
                    
                    # Duration
                    if duration:
                        mins = duration // 60
                        secs = duration % 60
                        st.caption(f"⏱️ {mins}:{secs:02d}")
                    
                    # Category badge
                    if category:
                        st.caption(f"🏷️ {category}")
                    
                    st.markdown("---")
    else:
        st.info("No videos found matching your criteria.")

except Exception as e:
    st.error(f"Database error: {e}")
    st.info("Make sure PostgreSQL is running and the database is initialized.")