"""
Chat endpoints for the Maxa API.
"""
from fastapi import APIRouter, HTTPException, WebSocket
from typing import List, Dict, Any, Optional
import json
import uuid
from datetime import datetime

from app.core.config import settings
from app.services.llm_service import llm_service
from app.services.memory_service import memory_service

router = APIRouter()

@router.post("")
async def chat(
    message: str,
    conversation_id: Optional[str] = None,
    user_id: str = "default_user"
) -> Dict[str, Any]:
    """
    Handle a chat message and return a response.
    """
    # Generate a new conversation ID if none provided
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
    
    try:
        # Get conversation history
        messages = await memory_service.get_conversation(conversation_id)
        
        # Add user message to conversation
        messages.append({"role": "user", "content": message})
        
        # Generate response using LLM
        response = await llm_service.generate_response(
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        
        # Add assistant response to conversation
        messages.append({"role": "assistant", "content": response})
        
        # Store updated conversation
        await memory_service.store_conversation(conversation_id, messages)
        
        # Store relevant memories
        await memory_service.store_memory(
            user_id=user_id,
            content=message,
            metadata={
                "conversation_id": conversation_id,
                "type": "user_message"
            }
        )
        
        return {
            "conversation_id": conversation_id,
            "response": response,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/ws/{conversation_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    conversation_id: str,
    user_id: str = "default_user"
):
    """
    WebSocket endpoint for real-time chat.
    """
    await websocket.accept()
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            message = message_data.get("message", "")
            
            # Get conversation history
            messages = await memory_service.get_conversation(conversation_id)
            
            # Add user message to conversation
            messages.append({"role": "user", "content": message})
            
            # Stream response
            full_response = ""
            async for chunk in llm_service.stream_response(
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            ):
                full_response += chunk
                await websocket.send_text(json.dumps({
                    "type": "chunk",
                    "content": chunk
                }))
            
            # Add assistant response to conversation
            messages.append({"role": "assistant", "content": full_response})
            
            # Store updated conversation
            await memory_service.store_conversation(conversation_id, messages)
            
            # Store relevant memories
            await memory_service.store_memory(
                user_id=user_id,
                content=message,
                metadata={
                    "conversation_id": conversation_id,
                    "type": "user_message"
                }
            )
            
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
    finally:
        await websocket.close()
