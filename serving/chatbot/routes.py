"""
Chatbot API Routes

FastAPI routes for the chatbot interface.
"""

import logging
from typing import Optional, List
from pydantic import BaseModel, Field

from fastapi import APIRouter, HTTPException

from serving.chatbot.chatbot_service import get_chatbot_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================
# Request/Response Models
# ============================================

class ChatRequest(BaseModel):
    """Chat request model."""
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(default=None, description="Session ID for conversation continuity")
    user_id: Optional[str] = Field(default=None, description="User ID for personalized recommendations")


class ChatMessage(BaseModel):
    """Chat message model."""
    role: str
    content: str


class ChatResponse(BaseModel):
    """Chat response model."""
    response: str
    session_id: str
    history: List[ChatMessage]


class SessionHistoryResponse(BaseModel):
    """Session history response."""
    session_id: str
    messages: List[ChatMessage]
    message_count: int


# ============================================
# Router
# ============================================

router = APIRouter(prefix="/chat", tags=["Chatbot"])


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message to the chatbot.
    
    The chatbot can:
    - Recommend videos based on user preferences
    - Search for videos by topic
    - Have general conversations about content
    
    Include user_id for personalized recommendations.
    Include session_id to continue a conversation.
    """
    try:
        service = get_chatbot_service()
        
        response_text, session = service.chat(
            message=request.message,
            session_id=request.session_id,
            user_id=request.user_id,
        )
        
        history = [
            ChatMessage(role=msg.role, content=msg.content)
            for msg in session.messages[-10:]  # Last 10 messages
        ]
        
        return ChatResponse(
            response=response_text,
            session_id=session.session_id,
            history=history,
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{session_id}", response_model=SessionHistoryResponse)
async def get_history(session_id: str):
    """Get chat history for a session."""
    try:
        service = get_chatbot_service()
        history = service.get_session_history(session_id)
        
        if not history:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return SessionHistoryResponse(
            session_id=session_id,
            messages=[ChatMessage(**msg) for msg in history],
            message_count=len(history),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"History error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/history/{session_id}")
async def clear_history(session_id: str):
    """Clear chat history for a session."""
    try:
        service = get_chatbot_service()
        
        if session_id in service.sessions:
            del service.sessions[session_id]
            return {"status": "success", "message": "Session cleared"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Clear history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))