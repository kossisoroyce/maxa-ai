#!/usr/bin/env python3
import argparse
import json
import logging
import os
import random
import re
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from queue import Queue
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import requests
from dotenv import load_dotenv
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

from anima_core import AgentCore
from memory_store import MemoryStore
from consciousness import Consciousness
from theory_of_mind import InteractionType

# Load and override environment vars from .env in project root
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize memory store
memory_store = MemoryStore()

# Initialize consciousness (which includes emotional state)
consciousness = Consciousness()
emotional_state = consciousness.state

# ... [rest of the file remains the same until get_count_result]

def get_count_result(collection_name: str) -> int:
    # count points, support both new and old CountResult interface
    cr = client.count(collection_name=collection_name)
    if hasattr(cr, "count"):
        return cr.count
    return cr.result.count

def get_memories(
    query_vec: List[float] = None,
    query_text: str = None,
    top_k: int = 5,
    filter_tags: List[str] = None,
    use_recency: bool = False,
    min_importance: float = 0.0,
    time_range: Optional[Tuple[datetime, datetime]] = None,
    include_vectors: bool = False,
    include_metadata: bool = True
) -> Union[List[str], List[Dict]]:
    """
    Retrieve memories with advanced filtering and sorting capabilities.
    
    Args:
        query_vec: Query vector for similarity search
        query_text: Text to search for (will be embedded if provided)
        top_k: Maximum number of results to return
        filter_tags: List of tags to filter by
        use_recency: Sort by recency if True
        min_importance: Minimum importance score (0.0-1.0)
        time_range: Optional tuple of (start_datetime, end_datetime)
        include_vectors: Whether to include vectors in results
        include_metadata: Whether to include metadata in results
        
    Returns:
        List of memory dictionaries with requested fields, or list of texts if include_metadata is False
    """
    # Build filter conditions
    filter_conditions = {}
    if filter_tags:
        filter_conditions["tags"] = filter_tags
    
    # Set sort order
    sort_by = "recency" if use_recency else "relevance"
    
    try:
        # Search memories using the MemoryStore
        results = memory_store.search_memories(
            query=query_text,
            query_vector=query_vec,
            top_k=top_k,
            filter_conditions=filter_conditions,
            sort_by=sort_by,
            min_importance=min_importance,
            time_range=time_range,
            include_vectors=include_vectors,
            include_metadata=include_metadata
        )
        
        # For backward compatibility, return just the text if metadata not requested
        if not include_metadata:
            return [result.get("text", "") for result in results]
        return results
        
    except Exception as e:
        logger.error(f"Error retrieving memories: {e}")
        # Return empty list on error to maintain backward compatibility
        return []

def main():
    """Main entry point for the Maxa Anima application."""
    print("Maxa Anima - Your AI Companion")
    print("Type 'exit' or press Ctrl+C to end the session\n")
    
    # Initialize the agent core
    agent = AgentCore()
    
    try:
        while True:
            # Get user input
            try:
                user_input = input("You: ")
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("\nGoodbye!")
                    break
                    
                # Process the input and get response
                response = agent.process_input(user_input)
                print(f"\nMaxa: {response['response']}\n")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nAn error occurred: {e}")
                continue
                
    except Exception as e:
        print(f"\nA critical error occurred: {e}")
    finally:
        print("\nThank you for chatting with Maxa Anima!")

if __name__ == "__main__":
    main()
