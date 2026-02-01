"""
LLM Client

Supports multiple LLM providers: OpenAI, Anthropic, Ollama, or mock.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from dataclasses import dataclass

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Chat message."""
    role: str  # 'user', 'assistant', 'system'
    content: str


@dataclass
class LLMResponse:
    """LLM response."""
    content: str
    model: str
    usage: Optional[Dict] = None


class BaseLLMClient(ABC):
    """Base class for LLM clients."""
    
    @abstractmethod
    def chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        """Send chat messages and get response."""
        pass


class OllamaClient(BaseLLMClient):
    """
    Ollama API client for local LLMs.
    
    Ollama runs open-source models locally for free.
    Install: https://ollama.ai
    
    Popular models:
    - llama3.2 (small, fast)
    - llama3.2:1b (tiny, very fast)
    - mistral (good balance)
    - phi3 (small, good quality)
    """
    
    def __init__(
        self,
        base_url: str = None,
        model: str = "llama3.2",
    ):
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model
        
        # Check if Ollama is running
        self._check_connection()
    
    def _check_connection(self):
        """Check if Ollama server is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "").split(":")[0] for m in models]
                logger.info(f"Ollama connected. Available models: {model_names}")
                
                # Check if requested model is available
                if self.model.split(":")[0] not in model_names and models:
                    logger.warning(f"Model '{self.model}' not found. Pull it with: ollama pull {self.model}")
            else:
                logger.warning(f"Ollama server returned status {response.status_code}")
        except requests.exceptions.ConnectionError:
            logger.warning(f"Could not connect to Ollama at {self.base_url}. Is Ollama running?")
        except Exception as e:
            logger.warning(f"Ollama connection check failed: {e}")
    
    def chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        """Send chat request to Ollama."""
        url = f"{self.base_url}/api/chat"
        
        payload = {
            "model": kwargs.get("model", self.model),
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
                "num_predict": kwargs.get("max_tokens", 1000),
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            
            data = response.json()
            
            return LLMResponse(
                content=data.get("message", {}).get("content", ""),
                model=data.get("model", self.model),
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                },
            )
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Could not connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running: ollama serve"
            )
        except requests.exceptions.Timeout:
            raise TimeoutError("Ollama request timed out. The model might be loading.")
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise


class OpenAIClient(BaseLLMClient):
    """OpenAI API client."""
    
    def __init__(
        self,
        api_key: str = None,
        model: str = "gpt-3.5-turbo",
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        
        if not self.api_key:
            logger.warning("OpenAI API key not set")
    
    def chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        """Send chat request to OpenAI."""
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=self.api_key)
            
            response = client.chat.completions.create(
                model=kwargs.get("model", self.model),
                messages=[{"role": m.role, "content": m.content} for m in messages],
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 1000),
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                },
            )
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise


class AnthropicClient(BaseLLMClient):
    """Anthropic API client."""
    
    def __init__(
        self,
        api_key: str = None,
        model: str = "claude-3-haiku-20240307",
    ):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        
        if not self.api_key:
            logger.warning("Anthropic API key not set")
    
    def chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        """Send chat request to Anthropic."""
        try:
            from anthropic import Anthropic
            
            client = Anthropic(api_key=self.api_key)
            
            # Extract system message if present
            system_msg = None
            chat_messages = []
            for m in messages:
                if m.role == "system":
                    system_msg = m.content
                else:
                    chat_messages.append({"role": m.role, "content": m.content})
            
            response = client.messages.create(
                model=kwargs.get("model", self.model),
                system=system_msg or "You are a helpful assistant.",
                messages=chat_messages,
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 1000),
            )
            
            return LLMResponse(
                content=response.content[0].text,
                model=response.model,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
            )
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise


class MockLLMClient(BaseLLMClient):
    """Mock LLM client for testing without API keys."""
    
    def __init__(self):
        self.model = "mock-model"
    
    def chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        """Return mock response."""
        last_message = messages[-1].content if messages else ""
        
        # Simple keyword-based responses
        if "recommend" in last_message.lower():
            content = "Based on your interests, I'd recommend checking out some technology tutorials. Would you like me to find specific videos for you?"
        elif "search" in last_message.lower() or "find" in last_message.lower():
            content = "I'll search for videos matching your request. What topic are you interested in?"
        elif "hello" in last_message.lower() or "hi" in last_message.lower():
            content = "Hello! I'm your video recommendation assistant. I can help you find videos, get personalized recommendations, or answer questions about content. What would you like to explore?"
        else:
            content = "I'm here to help you discover great videos. You can ask me for recommendations, search for specific topics, or tell me what you're in the mood to watch!"
        
        return LLMResponse(
            content=content,
            model=self.model,
            usage={"prompt_tokens": 0, "completion_tokens": 0},
        )


def check_ollama_available() -> bool:
    """Check if Ollama is running and available."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False


def create_llm_client(provider: str = "auto", **kwargs) -> BaseLLMClient:
    """
    Create LLM client based on provider.
    
    Args:
        provider: 'openai', 'anthropic', 'ollama', 'mock', or 'auto'
        **kwargs: Provider-specific arguments
        
    Returns:
        LLM client instance
    """
    if provider == "auto":
        # Priority: Ollama (free local) > OpenAI > Anthropic > Mock
        if check_ollama_available():
            provider = "ollama"
            logger.info("Auto-detected Ollama running locally")
        elif os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        elif os.getenv("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        else:
            logger.warning("No LLM provider found, using mock client")
            provider = "mock"
    
    if provider == "ollama":
        return OllamaClient(**kwargs)
    elif provider == "openai":
        return OpenAIClient(**kwargs)
    elif provider == "anthropic":
        return AnthropicClient(**kwargs)
    elif provider == "mock":
        return MockLLMClient()
    else:
        raise ValueError(f"Unknown provider: {provider}")


if __name__ == "__main__":
    print("Testing LLM Clients...")
    
    # Test Ollama if available
    if check_ollama_available():
        print("\n--- Testing Ollama Client ---")
        client = create_llm_client("ollama", model="llama3.2")
        
        messages = [
            Message(role="user", content="Hello! What can you help me with?")
        ]
        
        try:
            response = client.chat(messages)
            print(f"Response: {response.content[:200]}...")
            print(f"Model: {response.model}")
        except Exception as e:
            print(f"Ollama error: {e}")
    else:
        print("\nOllama not running. To use Ollama:")
        print("1. Install: https://ollama.ai")
        print("2. Run: ollama serve")
        print("3. Pull a model: ollama pull llama3.2")
    
    # Test mock client
    print("\n--- Testing Mock Client ---")
    client = create_llm_client("mock")
    
    messages = [
        Message(role="user", content="Hello, can you recommend some videos?")
    ]
    
    response = client.chat(messages)
    print(f"Response: {response.content}")
    print(f"Model: {response.model}")