"""
Analytics - System metrics and statistics
Features: Sankey Diagram for User Journey Visualization
"""

import streamlit as st
import psycopg2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from dotenv import load_dotenv

st.set_page_config(page_title="Analytics | YouTube RecSys", page_icon="📊", layout="wide")

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
</style>
""", unsafe_allow_html=True)

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://recsys:recsys_password@localhost:5432/youtube_recsys")

with st.sidebar:
    st.markdown("### 🎬 YouTube RecSys")
    st.caption("Video Recommendation System")
    st.markdown("---")

st.markdown("# 📊 Analytics")
st.caption("System metrics, user journeys, and content statistics")
st.markdown("---")

try:
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    # Key Metrics
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
    
    # ===========================================
    # SANKEY DIAGRAM - User Journey Visualization
    # ===========================================
    st.markdown("### 🔀 User Journey Flow")
    st.caption("How users flow between content categories (Persona → Category → Engagement)")
    
    # Get flow data: Persona → Category → Interaction Type
    cur.execute("""
        WITH user_category_interactions AS (
            SELECT 
                u.persona_type,
                v.category_name,
                ui.interaction_type,
                COUNT(*) as flow_count
            FROM user_interactions ui
            JOIN users u ON ui.user_id = u.user_id
            JOIN videos v ON ui.video_id = v.video_id
            WHERE u.persona_type IS NOT NULL 
              AND v.category_name IS NOT NULL
            GROUP BY u.persona_type, v.category_name, ui.interaction_type
            HAVING COUNT(*) > 5
        )
        SELECT * FROM user_category_interactions
        ORDER BY flow_count DESC
        LIMIT 50
    """)
    flow_data = cur.fetchall()
    
    if flow_data:
        df_flow = pd.DataFrame(flow_data, columns=["Persona", "Category", "Interaction", "Count"])
        
        # Create node labels
        personas = df_flow["Persona"].unique().tolist()
        categories = df_flow["Category"].unique().tolist()
        interactions = df_flow["Interaction"].unique().tolist()
        
        # All unique labels
        all_labels = personas + categories + interactions
        
        # Create index mappings
        label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
        
        # Build Sankey links
        # First layer: Persona → Category
        source_1 = []
        target_1 = []
        value_1 = []
        
        persona_category = df_flow.groupby(["Persona", "Category"])["Count"].sum().reset_index()
        for _, row in persona_category.iterrows():
            source_1.append(label_to_idx[row["Persona"]])
            target_1.append(label_to_idx[row["Category"]])
            value_1.append(row["Count"])
        
        # Second layer: Category → Interaction
        source_2 = []
        target_2 = []
        value_2 = []
        
        category_interaction = df_flow.groupby(["Category", "Interaction"])["Count"].sum().reset_index()
        for _, row in category_interaction.iterrows():
            source_2.append(label_to_idx[row["Category"]])
            target_2.append(label_to_idx[row["Interaction"]])
            value_2.append(row["Count"])
        
        # Combine all links
        all_sources = source_1 + source_2
        all_targets = target_1 + target_2
        all_values = value_1 + value_2
        
        # Color mapping
        persona_colors = ["#6366f1", "#8b5cf6", "#a855f7", "#d946ef", "#ec4899"]
        category_colors = ["#06b6d4", "#14b8a6", "#22c55e", "#84cc16", "#eab308"]
        interaction_colors = ["#f97316", "#ef4444", "#f43f5e"]
        
        node_colors = (
            persona_colors[:len(personas)] + 
            category_colors[:len(categories)] + 
            interaction_colors[:len(interactions)]
        )
        
        # Create Sankey
        fig_sankey = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=all_labels,
                color=node_colors
            ),
            link=dict(
                source=all_sources,
                target=all_targets,
                value=all_values,
                color="rgba(150, 150, 150, 0.3)"
            )
        )])
        
        fig_sankey.update_layout(
            font=dict(size=12, color="#ccc"),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400,
            margin=dict(t=20, b=20, l=20, r=20)
        )
        
        st.plotly_chart(fig_sankey, use_container_width=True)
        
        # Insights
        col1, col2, col3 = st.columns(3)
        
        with col1:
            top_persona = persona_category.groupby("Persona")["Count"].sum().idxmax()
            st.metric("🏆 Most Active Persona", top_persona)
        
        with col2:
            top_category = persona_category.groupby("Category")["Count"].sum().idxmax()
            st.metric("🎯 Most Popular Category", top_category)
        
        with col3:
            top_interaction = category_interaction.groupby("Interaction")["Count"].sum().idxmax()
            st.metric("⚡ Top Interaction Type", top_interaction)
    else:
        st.info("Not enough data for journey visualization. Generate more interactions first.")
    
    st.markdown("---")
    
    # ===========================================
    # Original Charts
    # ===========================================
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📂 Videos by Category")
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
        st.markdown("### 🎯 Interaction Types")
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
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👥 Users by Persona")
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
        st.markdown("### 📊 Watch Completion")
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
    
    # ===========================================
    # Cross-Category Flow (Additional Insight)
    # ===========================================
    st.markdown("### 🔄 Category Co-Watching Patterns")
    st.caption("Users who watch videos in one category often watch these other categories")
    
    cur.execute("""
        WITH user_categories AS (
            SELECT 
                ui.user_id,
                v.category_name
            FROM user_interactions ui
            JOIN videos v ON ui.video_id = v.video_id
            WHERE v.category_name IS NOT NULL
        ),
        category_pairs AS (
            SELECT 
                a.category_name as cat1,
                b.category_name as cat2,
                COUNT(DISTINCT a.user_id) as shared_users
            FROM user_categories a
            JOIN user_categories b ON a.user_id = b.user_id AND a.category_name < b.category_name
            GROUP BY a.category_name, b.category_name
            HAVING COUNT(DISTINCT a.user_id) > 3
        )
        SELECT * FROM category_pairs
        ORDER BY shared_users DESC
        LIMIT 10
    """)
    
    pairs_data = cur.fetchall()
    
    if pairs_data:
        df_pairs = pd.DataFrame(pairs_data, columns=["Category 1", "Category 2", "Shared Users"])
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_pairs = px.bar(
                df_pairs, 
                x="Shared Users", 
                y=df_pairs.apply(lambda x: f"{x['Category 1']} ↔ {x['Category 2']}", axis=1),
                orientation='h',
                color="Shared Users",
                color_continuous_scale="Purples"
            )
            fig_pairs.update_layout(
                margin=dict(t=10, b=10, l=10, r=10),
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#ccc'),
                yaxis_title="",
                xaxis_title="Users watching both",
                showlegend=False,
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_pairs, use_container_width=True)
        
        with col2:
            st.markdown("**💡 Insight**")
            top_pair = df_pairs.iloc[0]
            st.caption(
                f"Users who watch **{top_pair['Category 1']}** videos "
                f"are most likely to also watch **{top_pair['Category 2']}** "
                f"({top_pair['Shared Users']} users in common)"
            )
    
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
