"""
Custom Prometheus Metrics

Defines custom metrics for the recommendation system.
"""

from prometheus_client import Counter, Histogram, Gauge, Info
import time
from functools import wraps


# ============================================
# API Metrics
# ============================================

# Request counters
REQUESTS_TOTAL = Counter(
    'recsys_requests_total',
    'Total number of requests',
    ['endpoint', 'method', 'status']
)

# Recommendation metrics
RECOMMENDATIONS_SERVED = Counter(
    'recsys_recommendations_served_total',
    'Total number of recommendations served',
    ['user_type']  # 'known' or 'anonymous'
)

RECOMMENDATIONS_LATENCY = Histogram(
    'recsys_recommendations_latency_seconds',
    'Recommendation request latency',
    ['stage'],  # 'retrieval', 'ranking', 'reranking', 'total'
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# Retrieval metrics
RETRIEVAL_CANDIDATES = Histogram(
    'recsys_retrieval_candidates',
    'Number of candidates retrieved',
    buckets=[10, 50, 100, 200, 500, 1000]
)

# User feedback metrics
FEEDBACK_RECEIVED = Counter(
    'recsys_feedback_received_total',
    'Total feedback events received',
    ['interaction_type']  # 'view', 'like', 'dislike', 'share'
)

# Chat metrics
CHAT_MESSAGES = Counter(
    'recsys_chat_messages_total',
    'Total chat messages',
    ['role']  # 'user', 'assistant'
)

CHAT_LATENCY = Histogram(
    'recsys_chat_latency_seconds',
    'Chat response latency',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)


# ============================================
# System Metrics
# ============================================

# Database metrics
DB_CONNECTIONS_ACTIVE = Gauge(
    'recsys_db_connections_active',
    'Number of active database connections'
)

DB_QUERY_LATENCY = Histogram(
    'recsys_db_query_latency_seconds',
    'Database query latency',
    ['query_type'],  # 'select', 'insert', 'update'
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

# Model metrics
MODEL_INFERENCE_LATENCY = Histogram(
    'recsys_model_inference_latency_seconds',
    'Model inference latency',
    ['model'],  # 'retrieval', 'ranking', 'embedding'
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
)

# Index metrics
FAISS_INDEX_SIZE = Gauge(
    'recsys_faiss_index_size',
    'Number of vectors in FAISS index'
)

# Data metrics
TOTAL_VIDEOS = Gauge(
    'recsys_total_videos',
    'Total number of videos in database'
)

TOTAL_USERS = Gauge(
    'recsys_total_users',
    'Total number of users in database'
)

TOTAL_INTERACTIONS = Gauge(
    'recsys_total_interactions',
    'Total number of interactions in database'
)

# Service info
SERVICE_INFO = Info(
    'recsys_service',
    'Service information'
)


# ============================================
# Decorators for easy instrumentation
# ============================================

def track_latency(metric: Histogram, labels: dict = None):
    """Decorator to track function latency."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                if labels:
                    metric.labels(**labels).observe(duration)
                else:
                    metric.observe(duration)
        return wrapper
    return decorator


def track_latency_async(metric: Histogram, labels: dict = None):
    """Decorator to track async function latency."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                if labels:
                    metric.labels(**labels).observe(duration)
                else:
                    metric.observe(duration)
        return wrapper
    return decorator


# ============================================
# Helper functions
# ============================================

def record_recommendation_latency(stage: str, duration_seconds: float):
    """Record recommendation latency for a specific stage."""
    RECOMMENDATIONS_LATENCY.labels(stage=stage).observe(duration_seconds)


def record_feedback(interaction_type: str):
    """Record a feedback event."""
    FEEDBACK_RECEIVED.labels(interaction_type=interaction_type).inc()


def record_chat_message(role: str):
    """Record a chat message."""
    CHAT_MESSAGES.labels(role=role).inc()


def update_system_metrics(videos: int, users: int, interactions: int, index_size: int = 0):
    """Update system-level metrics."""
    TOTAL_VIDEOS.set(videos)
    TOTAL_USERS.set(users)
    TOTAL_INTERACTIONS.set(interactions)
    if index_size > 0:
        FAISS_INDEX_SIZE.set(index_size)


def set_service_info(version: str, environment: str = "development"):
    """Set service information."""
    SERVICE_INFO.info({
        'version': version,
        'environment': environment,
    })