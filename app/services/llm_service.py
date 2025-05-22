"""
LLM service for handling interactions with OpenAI's API.
"""
from typing import List, Dict, Any, Optional
import openai
from openai import AsyncOpenAI
from app.core.config import settings

class LLMService:
    """Service for interacting with OpenAI's API."""
    
    def __init__(self):
        """Initialize the OpenAI client."""
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL
    
    async def get_embeddings(
        self,
        texts: List[str],
        model: str = "text-embedding-3-small"
    ) -> List[List[float]]:
        """Get embeddings for a list of texts."""
        response = await self.client.embeddings.create(
            input=texts,
            model=model
        )
        return [item.embedding for item in response.data]
    
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        """Generate a response using the chat completion API."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            # Log the error and re-raise
            print(f"Error in generate_response: {str(e)}")
            raise
    
    async def stream_response(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ):
        """Stream the response from the LLM."""
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                **kwargs
            )
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            print(f"Error in stream_response: {str(e)}")
            raise

# Create a singleton instance
llm_service = LLMService()
