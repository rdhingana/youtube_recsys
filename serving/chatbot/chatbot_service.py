"""
Chatbot Service

Conversational interface for video recommendations.
"""

import os
import sys
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import uuid

import numpy as np
import psycopg2
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from serving.chatbot.llm_client import create_llm_client, Message, LLMResponse, BaseLLMClient
from models.retrieval import SimpleRetrievalService

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://recsys:recsys_password@localhost:5432/youtube_recsys")
INDEX_PATH = os.getenv("INDEX_PATH", "models/retrieval/saved/simple_faiss_index")


# ============================================
# System Prompts
# ============================================

SYSTEM_PROMPT = """You are a helpful video recommendation assistant. Your role is to:
1. Help users discover videos they'll enjoy
2. Understand user preferences through conversation
3. Provide personalized recommendations
4. Answer questions about videos and content

When recommending videos, you have access to a video database. You can:
- Search for videos by topic
- Get personalized recommendations based on user history
- Find similar videos to ones the user liked

Keep responses concise and conversational. When showing video recommendations, 
present them in a clear, engaging way with titles and brief descriptions.

If asked about specific videos, use the information provided. Don't make up video titles or details.
"""

RECOMMENDATION_PROMPT = """Based on the conversation, the user is interested in: {interests}

Here are some relevant videos from our database:
{videos}

Please present these recommendations conversationally, highlighting why each might interest the user.
Keep it brief - mention 3-5 videos max unless asked for more."""


# ============================================
# Data Classes
# ============================================

@dataclass
class ChatSession:
    """Represents a chat session."""
    session_id: str
    user_id: Optional[str] = None
    messages: List[Message] = field(default_factory=list)
    context: Dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class VideoInfo:
    """Video information for chat context."""
    video_id: str
    title: str
    channel_name: Optional[str] = None
    category_name: Optional[str] = None
    description: Optional[str] = None


# ============================================
# Database Functions
# ============================================

def get_connection():
    return psycopg2.connect(DATABASE_URL)


def parse_embedding(emb):
    if emb is None:
        return None
    if isinstance(emb, str):
        emb = emb.strip('[]')
        emb = [float(x) for x in emb.split(',')]
    return np.array(emb, dtype=np.float32)


def search_videos_by_text(query: str, limit: int = 10) -> List[VideoInfo]:
    """Search videos by text query."""
    search_query = """
        SELECT video_id, title, channel_name, category_name, description
        FROM videos
        WHERE is_active = true
          AND (title ILIKE %s OR description ILIKE %s OR category_name ILIKE %s)
        ORDER BY view_count DESC NULLS LAST
        LIMIT %s
    """
    
    pattern = f"%{query}%"
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(search_query, (pattern, pattern, pattern, limit))
            return [
                VideoInfo(
                    video_id=row[0],
                    title=row[1],
                    channel_name=row[2],
                    category_name=row[3],
                    description=row[4][:200] if row[4] else None,
                )
                for row in cur.fetchall()
            ]


def get_user_embedding(user_id: str) -> Optional[np.ndarray]:
    """Get user embedding."""
    query = "SELECT user_embedding FROM user_embeddings WHERE user_id = %s"
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (user_id,))
            row = cur.fetchone()
            if row and row[0]:
                return parse_embedding(row[0])
    return None


def get_video_info(video_ids: List[str]) -> List[VideoInfo]:
    """Get video info by IDs."""
    if not video_ids:
        return []
    
    placeholders = ','.join(['%s'] * len(video_ids))
    query = f"""
        SELECT video_id, title, channel_name, category_name, description
        FROM videos
        WHERE video_id IN ({placeholders})
    """
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, video_ids)
            return [
                VideoInfo(
                    video_id=row[0],
                    title=row[1],
                    channel_name=row[2],
                    category_name=row[3],
                    description=row[4][:200] if row[4] else None,
                )
                for row in cur.fetchall()
            ]


def save_chat_session(session: ChatSession):
    """Save chat session to database."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Insert or update session
                cur.execute("""
                    INSERT INTO chat_sessions (session_id, user_id, started_at)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (session_id) DO NOTHING
                """, (session.session_id, session.user_id, session.created_at))
                
                # Insert messages
                for msg in session.messages[-2:]:  # Save last 2 messages (user + assistant)
                    cur.execute("""
                        INSERT INTO chat_messages (session_id, role, content, metadata)
                        VALUES (%s, %s, %s, %s)
                    """, (
                        session.session_id,
                        msg.role,
                        msg.content,
                        json.dumps(session.context) if msg.role == "assistant" else None,
                    ))
                
                conn.commit()
    except Exception as e:
        logger.error(f"Error saving chat session: {e}")


# ============================================
# Intent Detection
# ============================================

class IntentDetector:
    """Detect user intent from message."""
    
    SEARCH_PATTERNS = [
        r"search\s+(?:for\s+)?(.+)",
        r"find\s+(?:me\s+)?(?:videos?\s+)?(?:about\s+)?(.+)",
        r"looking\s+for\s+(.+)",
        r"show\s+me\s+(.+)",
        r"videos?\s+about\s+(.+)",
    ]
    
    RECOMMEND_PATTERNS = [
        r"recommend",
        r"suggestion",
        r"what\s+should\s+i\s+watch",
        r"something\s+to\s+watch",
        r"for\s+me",
    ]
    
    @classmethod
    def detect(cls, message: str) -> Tuple[str, Optional[str]]:
        """
        Detect intent and extract query.
        
        Returns:
            Tuple of (intent, query)
            intent: 'search', 'recommend', 'chat'
        """
        message_lower = message.lower()
        
        # Check for search intent
        for pattern in cls.SEARCH_PATTERNS:
            match = re.search(pattern, message_lower)
            if match:
                return "search", match.group(1).strip()
        
        # Check for recommend intent
        for pattern in cls.RECOMMEND_PATTERNS:
            if re.search(pattern, message_lower):
                return "recommend", None
        
        return "chat", None


# ============================================
# Chatbot Service
# ============================================

class ChatbotService:
    """Main chatbot service."""
    
    def __init__(
        self,
        llm_provider: str = "auto",
        retrieval_service: SimpleRetrievalService = None,
    ):
        # Initialize LLM client
        self.llm_client = create_llm_client(llm_provider)
        logger.info(f"Using LLM: {self.llm_client.__class__.__name__}")
        
        # Initialize retrieval service
        self.retrieval_service = retrieval_service
        if self.retrieval_service is None:
            try:
                index_path = Path(INDEX_PATH)
                if index_path.with_suffix('.faiss').exists():
                    self.retrieval_service = SimpleRetrievalService(embedding_dim=256)
                    self.retrieval_service.load(str(index_path))
                    logger.info("Loaded retrieval service")
            except Exception as e:
                logger.warning(f"Could not load retrieval service: {e}")
        
        # Session storage (in-memory for simplicity)
        self.sessions: Dict[str, ChatSession] = {}
    
    def get_or_create_session(
        self,
        session_id: str = None,
        user_id: str = None,
    ) -> ChatSession:
        """Get existing session or create new one."""
        if session_id and session_id in self.sessions:
            return self.sessions[session_id]
        
        session_id = session_id or str(uuid.uuid4())
        session = ChatSession(session_id=session_id, user_id=user_id)
        self.sessions[session_id] = session
        return session
    
    def _format_videos_for_prompt(self, videos: List[VideoInfo]) -> str:
        """Format video list for LLM prompt."""
        if not videos:
            return "No videos found."
        
        lines = []
        for i, video in enumerate(videos[:10], 1):
            line = f"{i}. \"{video.title}\""
            if video.channel_name:
                line += f" by {video.channel_name}"
            if video.category_name:
                line += f" [{video.category_name}]"
            lines.append(line)
        
        return "\n".join(lines)
    
    def _get_personalized_recommendations(
        self,
        user_id: str,
        k: int = 10,
    ) -> List[VideoInfo]:
        """Get personalized recommendations for user."""
        if not self.retrieval_service:
            return []
        
        user_embedding = get_user_embedding(user_id)
        if user_embedding is None:
            return []
        
        video_ids, scores = self.retrieval_service.retrieve(user_embedding, k=k)
        return get_video_info(video_ids)
    
    def _search_videos(self, query: str, k: int = 10) -> List[VideoInfo]:
        """Search for videos by query."""
        return search_videos_by_text(query, limit=k)
    
    def chat(
        self,
        message: str,
        session_id: str = None,
        user_id: str = None,
    ) -> Tuple[str, ChatSession]:
        """
        Process a chat message and return response.
        
        Args:
            message: User message
            session_id: Session ID (optional)
            user_id: User ID for personalization (optional)
            
        Returns:
            Tuple of (response_text, session)
        """
        # Get or create session
        session = self.get_or_create_session(session_id, user_id)
        
        # Add user message
        session.messages.append(Message(role="user", content=message))
        
        # Detect intent
        intent, query = IntentDetector.detect(message)
        logger.info(f"Detected intent: {intent}, query: {query}")
        
        # Get relevant videos based on intent
        videos = []
        if intent == "search" and query:
            videos = self._search_videos(query)
            session.context["last_search"] = query
        elif intent == "recommend" and user_id:
            videos = self._get_personalized_recommendations(user_id)
            session.context["last_action"] = "recommend"
        
        # Build messages for LLM
        llm_messages = [Message(role="system", content=SYSTEM_PROMPT)]
        
        # Add conversation history (last 6 messages)
        for msg in session.messages[-6:]:
            llm_messages.append(msg)
        
        # Add video context if available
        if videos:
            video_context = RECOMMENDATION_PROMPT.format(
                interests=query or "personalized recommendations",
                videos=self._format_videos_for_prompt(videos),
            )
            llm_messages.append(Message(role="system", content=video_context))
        
        # Get LLM response
        try:
            response = self.llm_client.chat(llm_messages)
            response_text = response.content
        except Exception as e:
            logger.error(f"LLM error: {e}")
            response_text = "I apologize, but I'm having trouble processing your request. Could you try again?"
        
        # Add assistant response to session
        session.messages.append(Message(role="assistant", content=response_text))
        
        # Store video IDs in context
        if videos:
            session.context["last_videos"] = [v.video_id for v in videos]
        
        # Save session
        save_chat_session(session)
        
        return response_text, session
    
    def get_session_history(self, session_id: str) -> List[Dict]:
        """Get chat history for a session."""
        if session_id not in self.sessions:
            return []
        
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self.sessions[session_id].messages
        ]


# ============================================
# Singleton instance
# ============================================

_chatbot_service: Optional[ChatbotService] = None


def get_chatbot_service() -> ChatbotService:
    """Get or create chatbot service singleton."""
    global _chatbot_service
    if _chatbot_service is None:
        _chatbot_service = ChatbotService()
    return _chatbot_service


if __name__ == "__main__":
    print("Testing Chatbot Service...")
    
    # Create service with mock LLM
    service = ChatbotService(llm_provider="mock")
    
    # Test conversation
    print("\n--- Test Conversation ---")
    
    messages = [
        "Hello!",
        "Can you recommend some videos?",
        "Search for python tutorials",
        "What about machine learning?",
    ]
    
    session_id = None
    for msg in messages:
        print(f"\nUser: {msg}")
        response, session = service.chat(msg, session_id=session_id, user_id="test-user")
        session_id = session.session_id
        print(f"Assistant: {response}")
    
    print("\n--- Session History ---")
    history = service.get_session_history(session_id)
    for entry in history:
        print(f"  {entry['role']}: {entry['content'][:50]}...")