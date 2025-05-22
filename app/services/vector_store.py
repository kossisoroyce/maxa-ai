"""
Qdrant vector store implementation for Maxa AI.
"""
from typing import List, Optional, Dict, Any
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
)

from app.core.config import settings

class VectorStore:
    """Wrapper around Qdrant vector database."""
    
    def __init__(self):
        """Initialize Qdrant client."""
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY or None,
            prefer_grpc=True
        )
        self.collection_name = settings.QDRANT_COLLECTION
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=1536,  # OpenAI embedding size
                    distance=Distance.COSINE
                )
            )
    
    async def upsert_embedding(
        self,
        id: str,
        vector: List[float],
        payload: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Upsert an embedding with optional payload."""
        if payload is None:
            payload = {}
            
        points = [
            PointStruct(
                id=id,
                vector=vector,
                payload=payload
            )
        ]
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        return True
    
    async def search_similar(
        self,
        query_vector: List[float],
        limit: int = 5,
        score_threshold: float = 0.7,
        **filters
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        search_filters = []
        
        # Add filters if provided
        for field, value in filters.items():
            search_filters.append(
                FieldCondition(
                    key=field,
                    match=MatchValue(value=value)
                )
            )
        
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=Filter(must=search_filters) if search_filters else None,
            limit=limit,
            score_threshold=score_threshold
        )
        
        return [
            {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload
            }
            for hit in search_result
        ]
    
    async def delete_points(self, ids: List[str]) -> bool:
        """Delete points by IDs."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=ids)
        )
        return True

# Create a singleton instance
vector_store = VectorStore()
