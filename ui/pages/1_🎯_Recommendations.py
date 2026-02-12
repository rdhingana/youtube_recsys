"""
Recommendations - Get personalized video suggestions
Features: Quick Onboarding + Real-time Feedback Loop
"""

import streamlit as st
import requests
import psycopg2
import pandas as pd
import os
import uuid
from dotenv import load_dotenv

st.set_page_config(page_title="Recommendations | YouTube RecSys", page_icon="🎯", layout="wide")

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
    
    /* Feedback buttons */
    .feedback-btn {
        padding: 4px 8px;
        border-radius: 4px;
        border: none;
        cursor: pointer;
        font-size: 14px;
    }
    
    /* Category pills */
    .category-pill {
        display: inline-block;
        padding: 8px 16px;
        margin: 4px;
        border-radius: 20px;
        background: rgba(99, 102, 241, 0.2);
        color: #a5b4fc;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .category-pill:hover {
        background: rgba(99, 102, 241, 0.4);
    }
    
    .category-pill.selected {
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://recsys:recsys_password@localhost:5432/youtube_recsys")

# Session state for feedback and onboarding
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = {}
if "liked_videos" not in st.session_state:
    st.session_state.liked_videos = []
if "disliked_videos" not in st.session_state:
    st.session_state.disliked_videos = []
if "onboarding_complete" not in st.session_state:
    st.session_state.onboarding_complete = False
if "selected_categories" not in st.session_state:
    st.session_state.selected_categories = []
if "guest_mode" not in st.session_state:
    st.session_state.guest_mode = False

with st.sidebar:
    st.markdown("### 🎬 YouTube RecSys")
    st.caption("Video Recommendation System")
    st.markdown("---")
    
    # Show feedback stats
    if st.session_state.liked_videos or st.session_state.disliked_videos:
        st.markdown("### 📊 Your Feedback")
        st.caption(f"👍 Liked: {len(st.session_state.liked_videos)}")
        st.caption(f"👎 Disliked: {len(st.session_state.disliked_videos)}")
        if st.button("🔄 Reset Feedback", use_container_width=True):
            st.session_state.feedback_given = {}
            st.session_state.liked_videos = []
            st.session_state.disliked_videos = []
            st.rerun()


def get_thumbnail(video_id: str) -> str:
    return f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"


def get_categories():
    """Get available categories from database"""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT category_name, COUNT(*) as cnt
            FROM videos 
            WHERE category_name IS NOT NULL AND is_active = true
            GROUP BY category_name
            ORDER BY cnt DESC
        """)
        categories = cur.fetchall()
        conn.close()
        return [c[0] for c in categories]
    except:
        return ["Education", "Science & Technology", "Entertainment", "Gaming", "Music"]


def get_trending_videos(categories=None, limit=20):
    """Get trending/popular videos, optionally filtered by categories"""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        if categories:
            placeholders = ','.join(['%s'] * len(categories))
            cur.execute(f"""
                SELECT video_id, title, channel_name, category_name, view_count
                FROM videos 
                WHERE is_active = true AND category_name IN ({placeholders})
                ORDER BY view_count DESC NULLS LAST
                LIMIT %s
            """, (*categories, limit))
        else:
            cur.execute("""
                SELECT video_id, title, channel_name, category_name, view_count
                FROM videos 
                WHERE is_active = true
                ORDER BY view_count DESC NULLS LAST
                LIMIT %s
            """, (limit,))
        
        videos = cur.fetchall()
        conn.close()
        
        return [
            {
                "video_id": v[0],
                "title": v[1],
                "channel_name": v[2],
                "category_name": v[3],
                "score": 0.5 + (0.5 * (i / len(videos))) if videos else 0.5
            }
            for i, v in enumerate(reversed(videos))
        ]
    except Exception as e:
        st.error(f"Error: {e}")
        return []


def record_feedback(video_id: str, feedback_type: str):
    """Record user feedback (like/dislike)"""
    if feedback_type == "like":
        if video_id not in st.session_state.liked_videos:
            st.session_state.liked_videos.append(video_id)
        if video_id in st.session_state.disliked_videos:
            st.session_state.disliked_videos.remove(video_id)
    else:
        if video_id not in st.session_state.disliked_videos:
            st.session_state.disliked_videos.append(video_id)
        if video_id in st.session_state.liked_videos:
            st.session_state.liked_videos.remove(video_id)
    
    st.session_state.feedback_given[video_id] = feedback_type


st.markdown("# 🎯 Recommendations")
st.caption("Get personalized video suggestions")
st.markdown("---")

# Mode selection tabs
tab1, tab2 = st.tabs(["👤 Existing User", "🆕 Quick Start (Guest)"])

with tab1:
    # Existing user mode
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
        num_recommendations = st.slider("Number of results", 4, 48, 20, step=4, key="user_num")
        exclude_watched = st.checkbox("Exclude watched videos", value=False)
    
    if selected_user_id:
        if st.button("🚀 Get Recommendations", type="primary", use_container_width=True, key="user_btn"):
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
                        
                        # Store in session for feedback
                        st.session_state.current_recommendations = recommendations
                        
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Retrieval", f"{data.get('retrieval_time_ms', 0):.1f}ms")
                        c2.metric("Ranking", f"{data.get('ranking_time_ms', 0):.1f}ms")
                        c3.metric("Re-ranking", f"{data.get('reranking_time_ms', 0):.1f}ms")
                        c4.metric("Total", f"{data.get('total_time_ms', 0):.1f}ms")
                        
                        st.markdown("---")
                        st.markdown(f"### 📺 {len(recommendations)} Results")
                        st.caption("👍 Like or 👎 Dislike videos to improve future recommendations")
                        
                        if recommendations:
                            for row_idx in range(0, len(recommendations), 4):
                                cols = st.columns(4)
                                for col_idx, col in enumerate(cols):
                                    video_idx = row_idx + col_idx
                                    if video_idx < len(recommendations):
                                        video = recommendations[video_idx]
                                        video_id = video.get("video_id", "")
                                        
                                        with col:
                                            st.image(get_thumbnail(video_id), use_container_width=True)
                                            title = video.get("title", "Unknown")[:55]
                                            if len(video.get("title", "")) > 55:
                                                title += "..."
                                            st.markdown(f"**{title}**")
                                            st.caption(f"📺 {video.get('channel_name', 'Unknown')}")
                                            st.caption(f"⭐ {video.get('score', 0):.3f} · 🏷️ {video.get('category_name', 'N/A')}")
                                            
                                            # Feedback buttons
                                            fb_col1, fb_col2 = st.columns(2)
                                            
                                            current_feedback = st.session_state.feedback_given.get(video_id)
                                            
                                            with fb_col1:
                                                like_label = "👍 Liked" if current_feedback == "like" else "👍"
                                                if st.button(like_label, key=f"like_{video_id}", use_container_width=True):
                                                    record_feedback(video_id, "like")
                                                    st.rerun()
                                            
                                            with fb_col2:
                                                dislike_label = "👎 Nope" if current_feedback == "dislike" else "👎"
                                                if st.button(dislike_label, key=f"dislike_{video_id}", use_container_width=True):
                                                    record_feedback(video_id, "dislike")
                                                    st.rerun()
                                            
                                            st.markdown("")
                    else:
                        st.error(f"API Error: {response.status_code}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API. Run: `make start-api`")
                except Exception as e:
                    st.error(f"Error: {e}")

with tab2:
    # Netflix-style Quick Start with clickable tags
    st.markdown("### 🚀 What do you want to watch?")
    st.markdown("Select your interests (1-5) and we'll find videos for you instantly!")
    st.markdown("")
    
    # Get available categories with icons
    categories = get_categories()
    
    # Category icons mapping
    category_icons = {
        "Education": "📚",
        "Science & Technology": "🔬",
        "Entertainment": "🎭",
        "Gaming": "🎮",
        "Music": "🎵",
        "Sports": "⚽",
        "News & Politics": "📰",
        "Comedy": "😂",
        "Film & Animation": "🎬",
        "People & Blogs": "👥",
        "Howto & Style": "💄",
        "Pets & Animals": "🐾",
        "Autos & Vehicles": "🚗",
        "Travel & Events": "✈️",
        "Nonprofits & Activism": "💚",
    }
    
    # Create pill-style buttons using columns
    st.markdown("""
    <style>
        .tag-container {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 20px;
        }
        .selected-count {
            font-size: 14px;
            color: #a5b4fc;
            margin-bottom: 10px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Display selection count
    num_selected = len(st.session_state.selected_categories)
    if num_selected > 0:
        st.markdown(f"**Selected: {num_selected}/5** {'✅' if num_selected >= 1 else ''}")
    else:
        st.markdown("**Select at least 1 category to continue**")
    
    st.markdown("")
    
    # Create tag grid (5 columns)
    num_cols = 5
    rows = [categories[i:i + num_cols] for i in range(0, len(categories), num_cols)]
    
    for row in rows:
        cols = st.columns(num_cols)
        for idx, col in enumerate(cols):
            if idx < len(row):
                cat = row[idx]
                icon = category_icons.get(cat, "🏷️")
                is_selected = cat in st.session_state.selected_categories
                
                with col:
                    # Style based on selection
                    if is_selected:
                        btn_type = "primary"
                        label = f"{icon} {cat} ✓"
                    else:
                        btn_type = "secondary"
                        label = f"{icon} {cat}"
                    
                    if st.button(
                        label, 
                        key=f"tag_{cat}", 
                        use_container_width=True,
                        type=btn_type
                    ):
                        if is_selected:
                            # Deselect
                            st.session_state.selected_categories.remove(cat)
                        else:
                            # Select (max 5)
                            if len(st.session_state.selected_categories) < 5:
                                st.session_state.selected_categories.append(cat)
                            else:
                                st.toast("Maximum 5 categories allowed!", icon="⚠️")
                        st.rerun()
    
    st.markdown("")
    
    # Auto-show recommendations when categories are selected
    if st.session_state.selected_categories:
        st.markdown("---")
        
        selected_cats = st.session_state.selected_categories
        st.session_state.guest_mode = True
        
        # Get recommendations automatically
        recommendations = get_trending_videos(selected_cats, 16)
        
        if recommendations:
            st.markdown(f"### 📺 Recommended for You")
            st.caption(f"Based on: {', '.join(selected_cats)}")
            st.markdown("")
            
            for row_idx in range(0, len(recommendations), 4):
                cols = st.columns(4)
                for col_idx, col in enumerate(cols):
                    video_idx = row_idx + col_idx
                    if video_idx < len(recommendations):
                        video = recommendations[video_idx]
                        video_id = video.get("video_id", "")
                        
                        with col:
                            st.image(get_thumbnail(video_id), use_container_width=True)
                            title = video.get("title", "Unknown")[:55]
                            if len(video.get("title", "")) > 55:
                                title += "..."
                            st.markdown(f"**{title}**")
                            st.caption(f"📺 {video.get('channel_name', 'Unknown')}")
                            st.caption(f"🏷️ {video.get('category_name', 'N/A')}")
                            
                            # Feedback buttons
                            fb_col1, fb_col2 = st.columns(2)
                            
                            current_feedback = st.session_state.feedback_given.get(video_id)
                            
                            with fb_col1:
                                like_label = "👍 Liked" if current_feedback == "like" else "👍"
                                if st.button(like_label, key=f"glike_{video_id}", use_container_width=True):
                                    record_feedback(video_id, "like")
                                    st.rerun()
                            
                            with fb_col2:
                                dislike_label = "👎 Nope" if current_feedback == "dislike" else "👎"
                                if st.button(dislike_label, key=f"gdislike_{video_id}", use_container_width=True):
                                    record_feedback(video_id, "dislike")
                                    st.rerun()
                            
                            st.markdown("")
        else:
            st.info("No videos found in selected categories.")
    else:
        # Empty state - show visual prompt
        st.markdown("---")
        st.markdown(
            """
            <div style="text-align: center; padding: 60px 20px; color: #666;">
                <p style="font-size: 48px; margin-bottom: 10px;">👆</p>
                <p style="font-size: 18px;">Click on tags above to get started</p>
                <p style="font-size: 14px; color: #888;">We'll show you personalized recommendations instantly</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# Watch history for logged in users
if selected_user_id and not st.session_state.guest_mode:
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
