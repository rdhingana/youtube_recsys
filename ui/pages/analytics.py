"""
Analytics Page

System statistics and visualizations.
"""

import streamlit as st
import psycopg2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://recsys:recsys_password@localhost:5432/youtube_recsys")

st.title("📊 Analytics Dashboard")
st.markdown("---")

try:
    conn = psycopg2.connect(DATABASE_URL)
    
    # ==========================================
    # Key Metrics
    # ==========================================
    st.subheader("📈 Key Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    cur = conn.cursor()
    
    cur.execute("SELECT COUNT(*) FROM videos WHERE is_active = true")
    total_videos = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM users")
    total_users = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM user_interactions")
    total_interactions = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM video_embeddings")
    videos_with_emb = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM user_embeddings")
    users_with_emb = cur.fetchone()[0]
    
    col1.metric("Videos", f"{total_videos:,}")
    col2.metric("Users", f"{total_users:,}")
    col3.metric("Interactions", f"{total_interactions:,}")
    col4.metric("Video Embeddings", f"{videos_with_emb:,}")
    col5.metric("User Embeddings", f"{users_with_emb:,}")
    
    st.markdown("---")
    
    # ==========================================
    # Charts Row 1
    # ==========================================
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Videos by Category")
        cur.execute("""
            SELECT COALESCE(category_name, 'Unknown') as category, COUNT(*) as count 
            FROM videos 
            WHERE is_active = true 
            GROUP BY category_name 
            ORDER BY count DESC
            LIMIT 10
        """)
        category_data = cur.fetchall()
        
        if category_data:
            df = pd.DataFrame(category_data, columns=["Category", "Count"])
            fig = px.pie(df, values="Count", names="Category", hole=0.4)
            fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No category data available")
    
    with col2:
        st.subheader("🎯 Interactions by Type")
        cur.execute("""
            SELECT interaction_type, COUNT(*) as count 
            FROM user_interactions 
            GROUP BY interaction_type 
            ORDER BY count DESC
        """)
        interaction_data = cur.fetchall()
        
        if interaction_data:
            df = pd.DataFrame(interaction_data, columns=["Type", "Count"])
            fig = px.bar(df, x="Type", y="Count", color="Type")
            fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No interaction data available")
    
    st.markdown("---")
    
    # ==========================================
    # Charts Row 2
    # ==========================================
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👥 Users by Persona")
        cur.execute("""
            SELECT persona_type, COUNT(*) as count 
            FROM users 
            GROUP BY persona_type 
            ORDER BY count DESC
        """)
        persona_data = cur.fetchall()
        
        if persona_data:
            df = pd.DataFrame(persona_data, columns=["Persona", "Count"])
            fig = px.bar(df, x="Persona", y="Count", color="Persona")
            fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No user data available")
    
    with col2:
        st.subheader("📊 Watch Percentage Distribution")
        cur.execute("""
            SELECT 
                CASE 
                    WHEN watch_percentage < 0.25 THEN '0-25%'
                    WHEN watch_percentage < 0.50 THEN '25-50%'
                    WHEN watch_percentage < 0.75 THEN '50-75%'
                    ELSE '75-100%'
                END as bucket,
                COUNT(*) as count
            FROM user_interactions
            WHERE watch_percentage IS NOT NULL
            GROUP BY bucket
            ORDER BY bucket
        """)
        watch_data = cur.fetchall()
        
        if watch_data:
            df = pd.DataFrame(watch_data, columns=["Watch %", "Count"])
            fig = px.bar(df, x="Watch %", y="Count", color="Watch %")
            fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No watch data available")
    
    st.markdown("---")
    
    # ==========================================
    # Interactions Over Time
    # ==========================================
    st.subheader("📅 Interactions Over Time")
    
    cur.execute("""
        SELECT DATE(created_at) as date, COUNT(*) as count
        FROM user_interactions
        WHERE created_at > NOW() - INTERVAL '30 days'
        GROUP BY DATE(created_at)
        ORDER BY date
    """)
    time_data = cur.fetchall()
    
    if time_data:
        df = pd.DataFrame(time_data, columns=["Date", "Interactions"])
        fig = px.area(df, x="Date", y="Interactions")
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No time series data available")
    
    st.markdown("---")
    
    # ==========================================
    # Top Content
    # ==========================================
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔥 Most Watched Videos")
        cur.execute("""
            SELECT v.title, v.channel_name, COUNT(ui.id) as watches
            FROM videos v
            JOIN user_interactions ui ON v.video_id = ui.video_id
            WHERE ui.interaction_type = 'view'
            GROUP BY v.video_id, v.title, v.channel_name
            ORDER BY watches DESC
            LIMIT 10
        """)
        top_videos = cur.fetchall()
        
        if top_videos:
            df = pd.DataFrame(top_videos, columns=["Title", "Channel", "Watches"])
            df["Title"] = df["Title"].apply(lambda x: x[:40] + "..." if len(x) > 40 else x)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No watch data available")
    
    with col2:
        st.subheader("⭐ Most Active Users")
        cur.execute("""
            SELECT u.user_id, u.persona_type, COUNT(ui.id) as interactions
            FROM users u
            LEFT JOIN user_interactions ui ON u.user_id = ui.user_id
            GROUP BY u.user_id, u.persona_type
            ORDER BY interactions DESC
            LIMIT 10
        """)
        top_users = cur.fetchall()
        
        if top_users:
            df = pd.DataFrame(top_users, columns=["User ID", "Persona", "Interactions"])
            df["User ID"] = df["User ID"].apply(lambda x: x[:8] + "...")
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No user data available")
    
    conn.close()

except Exception as e:
    st.error(f"Database error: {e}")
    st.info("Make sure PostgreSQL is running and the database is initialized.")