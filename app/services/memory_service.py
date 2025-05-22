"""
Memory service for managing conversation history and context.
"""
from typing import List, Dict, Any, Optional
import uuid
import json
from datetime import datetime
import redis.asyncio as redis
from app.core.config import settings
from app.services.vector_store import vector_store
from app.services.llm_service import llm_service

class MemoryService:
    """Service for managing conversation memory and context."""
    
    def __init__(self):
        """Initialize the memory service."""
        self.redis = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            decode_responses=True
        )
    
    async def store_conversation(
        self,
        conversation_id: str,
        messages: List[Dict[str, str]]
    ) -> bool:
        """Store a conversation in Redis."""
        key = f"conversation:{conversation_id}"
        await self.redis.set(key, json.dumps(messages))
        return True
    
    async def get_conversation(
        self,
        conversation_id: str
    ) -> List[Dict[str, str]]:
        """Retrieve a conversation from Redis."""
        key = f"conversation:{conversation_id}"
        data = await self.redis.get(key)
        return json.loads(data) if data else []
    
    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str
    ) -> bool:
        """Add a message to a conversation."""
        messages = await self.get_conversation(conversation_id)
        messages.append({"role": role, "content": content})
        return await self.store_conversation(conversation_id, messages)
    
    async def store_memory(
        self,
        user_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store a memory with vector embeddings."""
        # Generate embeddings for the content
        embeddings = await llm_service.get_embeddings([content])
        vector = embeddings[0]
        
        # Create a memory ID
        memory_id = str(uuid.uuid4())
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "user_id": user_id,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "type": "memory"
        })
        
        # Store in vector database
        await vector_store.upsert_embedding(
            id=memory_id,
            vector=vector,
            payload=metadata
        )
        
        return memory_id
    
    async def search_memories(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Search for relevant memories."""
        # Get query embedding
        embeddings = await llm_service.get_embeddings([query])
        query_vector = embeddings[0]
        
        # Search in vector store
        results = await vector_store.search_similar(
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            user_id=user_id
        )
        
        return results

# Create a singleton instance
memory_service = MemoryService()
