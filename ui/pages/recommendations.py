"""
Recommendations Page

Get personalized video recommendations.
"""

import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("🎯 Personalized Recommendations")
st.markdown("---")

# User selection
col1, col2 = st.columns([2, 1])

with col1:
    # Get users from database
    try:
        import psycopg2
        DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://recsys:recsys_password@localhost:5432/youtube_recsys")
        
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("""
            SELECT u.user_id, u.persona_type, COUNT(ui.id) as interactions
            FROM users u
            LEFT JOIN user_interactions ui ON u.user_id = ui.user_id
            GROUP BY u.user_id, u.persona_type
            ORDER BY interactions DESC
            LIMIT 50
        """)
        users = cur.fetchall()
        conn.close()
        
        user_options = {f"{u[1]} - {u[0][:8]}... ({u[2]} interactions)": u[0] for u in users}
        
        if user_options:
            selected_user_label = st.selectbox(
                "Select a user",
                options=list(user_options.keys()),
                help="Choose a user to get personalized recommendations"
            )
            selected_user_id = user_options[selected_user_label]
        else:
            st.warning("No users found. Run the data pipeline first.")
            selected_user_id = None
            
    except Exception as e:
        st.error(f"Database error: {e}")
        selected_user_id = None

with col2:
    num_recommendations = st.slider("Number of recommendations", 5, 50, 20)
    exclude_watched = st.checkbox("Exclude watched videos", value=True)

st.markdown("---")

# Get recommendations button
if selected_user_id and st.button("🎬 Get Recommendations", type="primary"):
    with st.spinner("Fetching recommendations..."):
        try:
            response = requests.post(
                f"{API_URL}/recommend",
                json={
                    "user_id": selected_user_id,
                    "num_recommendations": num_recommendations,
                    "exclude_watched": exclude_watched,
                },
                timeout=30,
            )
            
            if response.status_code == 200:
                data = response.json()
                recommendations = data.get("recommendations", [])
                
                # Display timing info
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Retrieval", f"{data.get('retrieval_time_ms', 0):.1f}ms")
                col2.metric("Ranking", f"{data.get('ranking_time_ms', 0):.1f}ms")
                col3.metric("Re-ranking", f"{data.get('reranking_time_ms', 0):.1f}ms")
                col4.metric("Total", f"{data.get('total_time_ms', 0):.1f}ms")
                
                st.markdown("---")
                st.subheader(f"📺 {len(recommendations)} Recommendations")
                
                # Display recommendations in grid
                cols = st.columns(4)
                for idx, video in enumerate(recommendations):
                    with cols[idx % 4]:
                        with st.container():
                            # Thumbnail
                            if video.get("thumbnail_url"):
                                st.image(video["thumbnail_url"], use_container_width=True)
                            else:
                                st.image("https://via.placeholder.com/320x180?text=No+Thumbnail", use_container_width=True)
                            
                            # Title
                            title = video.get("title", "Unknown Title")
                            if len(title) > 50:
                                title = title[:47] + "..."
                            st.markdown(f"**{title}**")
                            
                            # Channel and score
                            channel = video.get("channel_name", "Unknown")
                            score = video.get("score", 0)
                            st.caption(f"📺 {channel}")
                            st.caption(f"⭐ Score: {score:.3f}")
                            
                            # Category
                            if video.get("category_name"):
                                st.caption(f"🏷️ {video['category_name']}")
                            
                            st.markdown("---")
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to API. Make sure the API server is running:")
            st.code("uvicorn serving.api.main:app --host 0.0.0.0 --port 8000")
        except Exception as e:
            st.error(f"Error: {e}")

# Show user's watch history
if selected_user_id:
    with st.expander("📜 User's Watch History"):
        try:
            conn = psycopg2.connect(DATABASE_URL)
            cur = conn.cursor()
            cur.execute("""
                SELECT v.title, v.channel_name, ui.interaction_type, 
                       ui.watch_percentage, ui.created_at
                FROM user_interactions ui
                JOIN videos v ON ui.video_id = v.video_id
                WHERE ui.user_id = %s
                ORDER BY ui.created_at DESC
                LIMIT 20
            """, (selected_user_id,))
            history = cur.fetchall()
            conn.close()
            
            if history:
                import pandas as pd
                df = pd.DataFrame(history, columns=["Title", "Channel", "Type", "Watch %", "Date"])
                df["Watch %"] = df["Watch %"].apply(lambda x: f"{x*100:.0f}%" if x else "-")
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No watch history for this user.")
                
        except Exception as e:
            st.error(f"Error loading history: {e}")