"""
Enhanced Qdrant-based memory store with advanced querying and management capabilities.
"""
import os
import time
import logging
from typing import List, Dict, Optional, Tuple, Any, Union
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# Configure logger
logger = logging.getLogger(__name__)

class MemoryStore:
    """
    Enhanced memory store using Qdrant with support for advanced querying,
    sharding, replication, and error handling.
    """
    
    def __init__(self, 
                 host: str = None, 
                 port: int = None, 
                 api_key: str = None,
                 collection_name: str = "maxa_memory",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the memory store.
        
        Args:
            host: Qdrant server host
            port: Qdrant server port
            api_key: Qdrant API key (for Qdrant Cloud)
            collection_name: Name of the collection to use
            embedding_model: Name of the sentence transformer model to use
        """
        self.host = host or os.getenv("QDRANT_HOST", "localhost")
        self.port = port or int(os.getenv("QDRANT_PORT", "6333"))
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.collection_name = collection_name or os.getenv("COLLECTION_NAME", "maxa_memory")
        self.embedding_model_name = embedding_model
        
        # Initialize Qdrant client
        self.client = self._init_qdrant_client()
        
        # Initialize embedder
        self.embedder = SentenceTransformer(self.embedding_model_name)
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
        
        # Ensure collection exists
        self._ensure_collection()
    
    def _init_qdrant_client(self) -> QdrantClient:
        """Initialize and return a Qdrant client with appropriate configuration."""
        try:
            if self.api_key:
                # Cloud connection with API key
                client = QdrantClient(
                    url=f"https://{self.host}",
                    api_key=self.api_key,
                    prefer_grpc=True,
                    timeout=10.0
                )
                logger.info(f"Connected to Qdrant Cloud at {self.host}")
            else:
                # Local connection
                client = QdrantClient(
                    host=self.host,
                    port=self.port,
                    prefer_grpc=True,
                    timeout=5.0
                )
                logger.info(f"Connected to local Qdrant at {self.host}:{self.port}")
            
            # Test the connection
            client.get_collections()
            return client
            
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise
    
    def _ensure_collection(self):
        """Ensure the Qdrant collection exists with proper configuration."""
        try:
            # Get or create collection with enhanced configuration
            existing = self.client.get_collections().collections
            names = [c.name for c in existing]
            
            if self.collection_name not in names:
                # Configure sharding and replication
                shard_number = int(os.getenv("QDRANT_SHARD_NUMBER", "1"))
                replication_factor = int(os.getenv("QDRANT_REPLICATION_FACTOR", "1"))
                
                # Configure payload schema for indexing
                payload_schema = {
                    "text": {"type": "text"},
                    "tags": {"type": "keyword"},
                    "importance": {"type": "float"},
                    "timestamp": {"type": "datetime"},
                    "emotional_state.mood": {"type": "keyword"},
                    "emotional_state.arousal": {"type": "float"},
                    "emotional_state.valence": {"type": "float"},
                    "goal_related": {"type": "bool"},
                    "temporal_context": {"type": "keyword"},
                    "user_sentiment": {"type": "keyword"},
                    "interaction_type": {"type": "keyword"}
                }
                
                # Create collection with enhanced configuration
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim, 
                        distance=Distance.COSINE
                    ),
                    shard_number=shard_number,
                    replication_factor=replication_factor,
                    on_disk_payload=True,
                    timeout=10.0,
                    optimizers_config={
                        "default_segment_number": 2,
                        "max_segment_size": 10000,
                        "memmap_threshold": 20000,
                        "indexing_threshold": 20000,
                    },
                    hnsw_config={
                        "m": 16,  # Number of edges per node
                        "ef_construct": 100,  # Number of neighbors to consider during index time
                        "full_scan_threshold": 10000,  # Threshold for full scan search
                    },
                    payload_schema=payload_schema
                )
                
                # Create payload indexes for faster filtering
                for field in ["tags", "importance", "timestamp", "emotional_state.mood"]:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field,
                        field_schema=payload_schema[field]
                    )
                    
                logger.info(f"Created new Qdrant collection '{self.collection_name}' with sharding and replication")
                
            else:
                logger.info(f"Using existing Qdrant collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Failed to ensure collection exists: {e}")
            raise
    
    def upsert_memory(
        self,
        text: str, 
        vector: List[float] = None,
        importance: float = None, 
        tags: List[str] = None,
        goal_context: Optional[Dict] = None,
        temporal_context: Optional[Dict] = None,
        interaction_metadata: Optional[Dict] = None,
        max_retries: int = 3,
        initial_delay: float = 0.1
    ) -> bool:
        """
        Store a memory with enhanced context and automatic consolidation.
        
        Args:
            text: The memory text content
            vector: Optional embedding vector (will be generated if not provided)
            importance: Optional importance score (0.0-1.0)
            tags: List of tags for categorization
            goal_context: Optional context from the goal system
            temporal_context: Optional temporal context
            interaction_metadata: Metadata about the interaction
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay between retries in seconds
            
        Returns:
            bool: True if successful, False otherwise
        """
        if tags is None:
            tags = []
        
        # Generate embedding if not provided
        if vector is None:
            vector = self.embedder.encode([text], convert_to_numpy=True)[0].tolist()
        
        # Calculate importance if not provided (simple heuristic based on length)
        if importance is None:
            importance = min(1.0, len(text) / 1000)  # Scale importance by text length up to 1.0
        
        # Prepare payload with all context
        payload = {
            "text": text,
            "timestamp": datetime.utcnow().isoformat(),
            "tags": tags,
            "importance": importance,
            "goal_related": bool(goal_context),
            "temporal_context": temporal_context or {},
            "interaction_metadata": interaction_metadata or {}
        }
        
        # Add goal context if available
        if goal_context:
            payload.update({
                "goal_id": goal_context.get("goal_id"),
                "goal_title": goal_context.get("title"),
                "goal_priority": goal_context.get("priority"),
                "goal_status": goal_context.get("status")
            })
        
        # Add temporal context if available
        if temporal_context:
            payload.update({
                "event_start": temporal_context.get("start_time"),
                "event_end": temporal_context.get("end_time"),
                "is_recurring": temporal_context.get("is_recurring", False),
                "time_until_event": temporal_context.get("time_until")
            })
        
        # Add interaction metadata if available
        if interaction_metadata:
            payload.update({
                "user_sentiment": interaction_metadata.get("sentiment"),
                "interaction_type": interaction_metadata.get("type"),
                "user_intent": interaction_metadata.get("intent"),
                "confidence_score": interaction_metadata.get("confidence")
            })
        
        # Create memory point with retry logic
        point = PointStruct(
            id=int(time.time() * 1000),
            vector=vector,
            payload=payload
        )
        
        # Implement retry logic with exponential backoff
        delay = initial_delay
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                # Store in Qdrant with timeout
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=[point],
                    wait=True,
                    timeout=5.0  # 5 second timeout
                )
                return True
                
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    logger.warning(
                        f"Attempt {attempt + 1} failed with error: {str(e)}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    logger.error(
                        f"Failed to upsert memory after {max_retries + 1} attempts. "
                        f"Last error: {str(last_exception)}"
                    )
                    return False
        
        return False
    
    def search_memories(
        self,
        query: str = None,
        query_vector: List[float] = None,
        top_k: int = 5,
        filter_conditions: Optional[Dict] = None,
        sort_by: str = "relevance",  # 'relevance', 'recency', 'importance'
        min_importance: float = 0.0,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        include_vectors: bool = False,
        include_metadata: bool = True,
        timeout: float = 5.0,
        max_retries: int = 2,
        initial_delay: float = 0.1
    ) -> List[Dict]:
        """
        Search memories with advanced filtering and sorting capabilities.
        
        Args:
            query: Text to search for (will be embedded if provided)
            query_vector: Optional query vector (if not provided, will be generated from query)
            top_k: Maximum number of results to return
            filter_conditions: Dictionary of filter conditions
            sort_by: How to sort results ('relevance', 'recency', 'importance')
            min_importance: Minimum importance score (0.0-1.0)
            time_range: Optional tuple of (start_datetime, end_datetime) to filter by time
            include_vectors: Whether to include vector data in results
            include_metadata: Whether to include metadata in results
            timeout: Query timeout in seconds
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay between retries in seconds
            
        Returns:
            List of memory dictionaries with requested fields
        """
        # Validate input
        if query is None and query_vector is None:
            raise ValueError("Either query or query_vector must be provided")
        
        # Initialize variables
        delay = initial_delay
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                # Generate query vector if not provided
                if query_vector is None:
                    query_vector = self.embedder.encode([query], convert_to_numpy=True)[0].tolist()
                
                # Build filter conditions
                filters = []
                
                # Add importance filter
                if min_importance > 0:
                    filters.append({"key": "importance", "range": {"gt": min_importance}})
                
                # Add time range filter
                if time_range:
                    start_time, end_time = time_range
                    filters.append({
                        "key": "timestamp",
                        "range": {
                            "gt": start_time.isoformat(),
                            "lte": end_time.isoformat()
                        }
                    })
                
                # Add custom filter conditions
                if filter_conditions:
                    for field, value in filter_conditions.items():
                        if isinstance(value, dict):
                            # Handle range queries
                            range_conditions = []
                            for op, op_value in value.items():
                                if op == "$gt":
                                    range_conditions.append({"gt": op_value})
                                elif op == "$lt":
                                    range_conditions.append({"lt": op_value})
                                elif op == "$eq":
                                    range_conditions.append({"eq": op_value})
                            if range_conditions:
                                filters.append({"key": field, "range": range_conditions[0]})
                        elif isinstance(value, list):
                            # Handle list of possible values
                            filters.append({"key": field, "match": {"any": value}})
                        else:
                            # Handle exact match
                            filters.append({"key": field, "match": {"value": value}})
                
                # Build query parameters
                query_params = {
                    "collection_name": self.collection_name,
                    "query_vector": query_vector,
                    "limit": top_k,
                    "with_payload": include_metadata,
                    "with_vectors": include_vectors,
                    "timeout": timeout,
                }
                
                # Add filters if any
                if filters:
                    query_params["query_filter"] = {"must": filters}
                
                # Execute query
                resp = self.client.search(**query_params)
                
                # Process results
                results = []
                for hit in resp:
                    result = {
                        "id": hit.id,
                        "score": hit.score,
                        "text": hit.payload.get("text", ""),
                        "payload": hit.payload if include_metadata else None,
                        "vector": hit.vector if include_vectors else None
                    }
                    results.append(result)
                
                # Sort results if needed
                if sort_by == "recency" and include_metadata:
                    results.sort(key=lambda x: x["payload"].get("timestamp", ""), reverse=True)
                elif sort_by == "importance" and include_metadata:
                    results.sort(key=lambda x: x["payload"].get("importance", 0), reverse=True)
                
                return results[:top_k]
                
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    logger.warning(
                        f"Memory query attempt {attempt + 1} failed with error: {str(e)}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    logger.error(
                        f"Failed to retrieve memories after {max_retries + 1} attempts. "
                        f"Last error: {str(last_exception)}"
                    )
                    raise
    
    def delete_memory(self, memory_id: Union[int, str]) -> bool:
        """
        Delete a memory by its ID.
        
        Args:
            memory_id: ID of the memory to delete
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[memory_id]
                )
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False
    
    def count_memories(self) -> int:
        """Return the total number of memories in the collection."""
        try:
            result = self.client.count(collection_name=self.collection_name)
            return result.count if hasattr(result, 'count') else result.result.count
        except Exception as e:
            logger.error(f"Failed to count memories: {e}")
            return 0
    
    def clear_collection(self) -> bool:
        """
        Clear all memories from the collection.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            self._ensure_collection()  # Recreate the collection
            return True
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False

# Singleton instance
memory_store = MemoryStore()
