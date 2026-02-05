"""
Analytics - System metrics and statistics
"""

import streamlit as st
import psycopg2
import pandas as pd
import plotly.express as px
import os
from dotenv import load_dotenv

st.set_page_config(page_title="Analytics | YouTube RecSys", page_icon="📊", layout="wide")

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

st.markdown("# 📊 Analytics")
st.caption("System metrics and content statistics")
st.markdown("---")

try:
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    cur.execute("SELECT COUNT(*) FROM videos WHERE is_active = true")
    col1.metric("📹 Videos", f"{cur.fetchone()[0]:,}")
    
    cur.execute("SELECT COUNT(*) FROM users")
    col2.metric("👥 Users", f"{cur.fetchone()[0]:,}")
    
    cur.execute("SELECT COUNT(*) FROM user_interactions")
    col3.metric("🎯 Interactions", f"{cur.fetchone()[0]:,}")
    
    cur.execute("SELECT COUNT(*) FROM video_embeddings")
    col4.metric("🧠 Video Emb.", f"{cur.fetchone()[0]:,}")
    
    cur.execute("SELECT COUNT(*) FROM user_embeddings")
    col5.metric("🧠 User Emb.", f"{cur.fetchone()[0]:,}")
    
    st.markdown("---")
    
    # Charts row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Videos by Category")
        cur.execute("""
            SELECT COALESCE(category_name, 'Unknown'), COUNT(*) 
            FROM videos WHERE is_active = true 
            GROUP BY category_name ORDER BY COUNT(*) DESC
        """)
        data = cur.fetchall()
        if data:
            df = pd.DataFrame(data, columns=["Category", "Count"])
            fig = px.pie(df, values="Count", names="Category", hole=0.4,
                        color_discrete_sequence=px.colors.sequential.Purples_r)
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), 
                            paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#ccc'))
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Interaction Types")
        cur.execute("""
            SELECT interaction_type, COUNT(*) 
            FROM user_interactions GROUP BY interaction_type ORDER BY COUNT(*) DESC
        """)
        data = cur.fetchall()
        if data:
            df = pd.DataFrame(data, columns=["Type", "Count"])
            fig = px.bar(df, x="Type", y="Count", color="Type",
                        color_discrete_sequence=px.colors.sequential.Purples_r)
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), 
                            paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#ccc'),
                            showlegend=False, xaxis_title="", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Charts row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Users by Persona")
        cur.execute("""
            SELECT COALESCE(persona_type, 'Unknown'), COUNT(*) 
            FROM users GROUP BY persona_type ORDER BY COUNT(*) DESC
        """)
        data = cur.fetchall()
        if data:
            df = pd.DataFrame(data, columns=["Persona", "Count"])
            fig = px.bar(df, x="Persona", y="Count", color="Persona",
                        color_discrete_sequence=px.colors.sequential.Tealgrn_r)
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), 
                            paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#ccc'),
                            showlegend=False, xaxis_title="", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Watch Completion")
        cur.execute("""
            SELECT 
                CASE 
                    WHEN watch_percentage < 0.25 THEN '0-25%' 
                    WHEN watch_percentage < 0.50 THEN '25-50%' 
                    WHEN watch_percentage < 0.75 THEN '50-75%' 
                    ELSE '75-100%' 
                END, COUNT(*)
            FROM user_interactions WHERE watch_percentage IS NOT NULL 
            GROUP BY 1 ORDER BY 1
        """)
        data = cur.fetchall()
        if data:
            df = pd.DataFrame(data, columns=["Range", "Count"])
            fig = px.bar(df, x="Range", y="Count", color="Range",
                        color_discrete_sequence=['#ef4444', '#f97316', '#eab308', '#22c55e'])
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), 
                            paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#ccc'),
                            showlegend=False, xaxis_title="", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Tables
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔥 Most Watched")
        cur.execute("""
            SELECT v.title, v.channel_name, COUNT(*) as views
            FROM videos v 
            JOIN user_interactions ui ON v.video_id = ui.video_id 
            WHERE ui.interaction_type = 'view' 
            GROUP BY v.video_id, v.title, v.channel_name 
            ORDER BY views DESC LIMIT 8
        """)
        data = cur.fetchall()
        if data:
            df = pd.DataFrame(data, columns=["Title", "Channel", "Views"])
            df["Title"] = df["Title"].apply(lambda x: x[:30] + "..." if len(str(x)) > 30 else x)
            st.dataframe(df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("### ⭐ Most Active Users")
        cur.execute("""
            SELECT u.persona_type, COUNT(*) as interactions
            FROM users u 
            JOIN user_interactions ui ON u.user_id = ui.user_id 
            GROUP BY u.user_id, u.persona_type 
            ORDER BY interactions DESC LIMIT 8
        """)
        data = cur.fetchall()
        if data:
            df = pd.DataFrame(data, columns=["Persona", "Interactions"])
            st.dataframe(df, use_container_width=True, hide_index=True)
    
    conn.close()

except Exception as e:
    st.error(f"Database error: {e}")