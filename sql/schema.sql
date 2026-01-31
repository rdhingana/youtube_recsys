-- YouTube Recommendation System Database Schema
-- Phase 1: Initial Setup

-- ============================================
-- EXTENSIONS
-- ============================================
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";  -- pgvector for embeddings (Phase 3)

-- ============================================
-- CORE TABLES
-- ============================================

-- Videos: stores all video metadata
CREATE TABLE videos (
    video_id VARCHAR(20) PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    channel_id VARCHAR(30),
    channel_name VARCHAR(255),
    category_id INTEGER,
    category_name VARCHAR(100),
    tags TEXT[],
    duration_seconds INTEGER,
    view_count BIGINT,
    like_count BIGINT,
    comment_count BIGINT,
    thumbnail_url TEXT,
    published_at TIMESTAMP WITH TIME ZONE,
    scraped_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Users: simulated users
CREATE TABLE users (
    user_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    preferred_categories INTEGER[],
    account_age_days INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- User interactions: watch history, likes, etc.
CREATE TABLE user_interactions (
    interaction_id BIGSERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    video_id VARCHAR(20) NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    interaction_type VARCHAR(20) NOT NULL,  -- 'view', 'like', 'dislike', 'share'
    watch_duration_seconds INTEGER,
    watch_percentage FLOAT,
    session_id UUID,
    device_type VARCHAR(20),
    recommendation_source VARCHAR(50),
    position_in_list INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================
-- EMBEDDING TABLES (for Phase 3)
-- ============================================

-- Video embeddings: CLIP and text embeddings
CREATE TABLE video_embeddings (
    video_id VARCHAR(20) PRIMARY KEY REFERENCES videos(video_id) ON DELETE CASCADE,
    thumbnail_embedding vector(512),      -- CLIP ViT-B/32
    title_embedding vector(384),          -- all-MiniLM-L6-v2
    description_embedding vector(384),
    combined_embedding vector(256),       -- fused embedding for retrieval
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- User embeddings: learned from watch history
CREATE TABLE user_embeddings (
    user_id UUID PRIMARY KEY REFERENCES users(user_id) ON DELETE CASCADE,
    user_embedding vector(256),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================
-- RECOMMENDATION TRACKING
-- ============================================

-- Recommendation requests: log every recommendation served
CREATE TABLE recommendation_requests (
    request_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(user_id),
    request_type VARCHAR(30),
    context_video_id VARCHAR(20),
    retrieval_time_ms INTEGER,
    ranking_time_ms INTEGER,
    reranking_time_ms INTEGER,
    total_time_ms INTEGER,
    num_candidates_retrieved INTEGER,
    num_candidates_ranked INTEGER,
    num_results_returned INTEGER,
    model_version VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Recommendation results: individual items returned
CREATE TABLE recommendation_results (
    id BIGSERIAL PRIMARY KEY,
    request_id UUID NOT NULL REFERENCES recommendation_requests(request_id) ON DELETE CASCADE,
    video_id VARCHAR(20) NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    retrieval_score FLOAT,
    ranking_score FLOAT,
    final_score FLOAT,
    was_clicked BOOLEAN DEFAULT FALSE,
    watch_percentage FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================
-- CHATBOT TABLES (for Phase 7)
-- ============================================

-- Chat sessions
CREATE TABLE chat_sessions (
    session_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(user_id),
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ended_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT TRUE
);

-- Chat messages
CREATE TABLE chat_messages (
    message_id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL,  -- 'user', 'assistant'
    content TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================
-- INDEXES
-- ============================================

-- Videos
CREATE INDEX idx_videos_category ON videos(category_id);
CREATE INDEX idx_videos_channel ON videos(channel_id);
CREATE INDEX idx_videos_published ON videos(published_at DESC);
CREATE INDEX idx_videos_active ON videos(is_active) WHERE is_active = TRUE;

-- User interactions
CREATE INDEX idx_interactions_user ON user_interactions(user_id);
CREATE INDEX idx_interactions_video ON user_interactions(video_id);
CREATE INDEX idx_interactions_user_time ON user_interactions(user_id, created_at DESC);
CREATE INDEX idx_interactions_type ON user_interactions(interaction_type);

-- Recommendations
CREATE INDEX idx_rec_requests_user ON recommendation_requests(user_id);
CREATE INDEX idx_rec_requests_time ON recommendation_requests(created_at DESC);
CREATE INDEX idx_rec_results_request ON recommendation_results(request_id);

-- Chat
CREATE INDEX idx_chat_messages_session ON chat_messages(session_id);
CREATE INDEX idx_chat_sessions_user ON chat_sessions(user_id);

-- ============================================
-- VIEWS (for Grafana)
-- ============================================

-- Daily engagement summary
CREATE VIEW v_daily_engagement AS
SELECT 
    DATE(created_at) AS date,
    COUNT(*) AS total_interactions,
    COUNT(DISTINCT user_id) AS unique_users,
    COUNT(DISTINCT video_id) AS unique_videos,
    AVG(watch_percentage) AS avg_watch_percentage,
    SUM(CASE WHEN interaction_type = 'like' THEN 1 ELSE 0 END) AS total_likes
FROM user_interactions
GROUP BY DATE(created_at);

-- Recommendation latency stats
CREATE VIEW v_recommendation_latency AS
SELECT 
    DATE(created_at) AS date,
    COUNT(*) AS total_requests,
    AVG(total_time_ms) AS avg_latency_ms,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_time_ms) AS p50_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY total_time_ms) AS p95_ms,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY total_time_ms) AS p99_ms
FROM recommendation_requests
GROUP BY DATE(created_at);

-- Category popularity
CREATE VIEW v_category_popularity AS
SELECT 
    v.category_name,
    COUNT(DISTINCT v.video_id) AS video_count,
    COUNT(ui.interaction_id) AS interaction_count,
    AVG(ui.watch_percentage) AS avg_watch_percentage
FROM videos v
LEFT JOIN user_interactions ui ON v.video_id = ui.video_id
WHERE v.category_name IS NOT NULL
GROUP BY v.category_name;

-- ============================================
-- FUNCTIONS
-- ============================================

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply triggers
CREATE TRIGGER tr_videos_updated_at 
    BEFORE UPDATE ON videos
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER tr_users_updated_at 
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER tr_video_embeddings_updated_at 
    BEFORE UPDATE ON video_embeddings
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER tr_user_embeddings_updated_at 
    BEFORE UPDATE ON user_embeddings
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();