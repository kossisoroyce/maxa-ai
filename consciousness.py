from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, TypedDict, TypeVar
import json
import logging
import os
import time
import uuid
from collections import deque
from enum import Enum, auto
from typing import Literal, overload, Union, Callable

# Type variable for generic type hints
T = TypeVar('T')

# Type aliases
ContextType = Literal['conversation', 'task', 'learning', 'reflection', 'other']

@dataclass
class Context:
    """Represents a contextual unit of information or experience."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    context_type: ContextType = 'conversation'
    timestamp: datetime = field(default_factory=datetime.utcnow)
    importance: float = 0.5  # 0.0 to 1.0
    related_entities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'content': self.content,
            'context_type': self.context_type,
            'timestamp': self.timestamp.isoformat(),
            'importance': self.importance,
            'related_entities': self.related_entities,
            'metadata': self.metadata
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Context':
        return cls(
            id=data.get('id', str(uuid.uuid4())),
            content=data['content'],
            context_type=data.get('context_type', 'conversation'),
            timestamp=datetime.fromisoformat(data['timestamp']) if isinstance(data['timestamp'], str) else data['timestamp'],
            importance=data.get('importance', 0.5),
            related_entities=data.get('related_entities', []),
            metadata=data.get('metadata', {})
        )

@dataclass
class ContentSummary:
    """Represents a summary of content with key points and metadata."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    key_points: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: Optional[str] = None
    context_ids: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'content': self.content[:500] + '...' if len(self.content) > 500 else self.content,
            'key_points': self.key_points,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'context_ids': self.context_ids
        }

logger = logging.getLogger(__name__)

class ConsciousnessState(Enum):
    ACTIVE = "active"           # Fully engaged and responsive
    DREAMING = "dreaming"       # Processing internally
    REFLECTING = "reflecting"   # Thinking deeply
    RESTING = "resting"         # Low-power state
    LEARNING = "learning"       # Actively learning
    UNCERTAIN = "uncertain"     # Unsure about something
    CONFUSED = "confused"       # Confused state
    FOCUSED = "focused"         # Deeply focused on a task

class EmotionalState(Enum):
    JOY = "joy"
    INTEREST = "interest"
    SURPRISE = "surprise"
    TRUST = "trust"
    FEAR = "fear"
    SADNESS = "sadness"
    DISGUST = "disgust"
    ANGER = "anger"
    ANTICIPATION = "anticipation"
    NEUTRAL = "neutral"

@dataclass
class MentalState:
    state: ConsciousnessState = ConsciousnessState.ACTIVE
    intensity: float = 0.7  # 0.0 to 1.0
    emotional_state: Dict[EmotionalState, float] = field(default_factory=dict)
    focus: str = ""
    sub_states: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        # Initialize with neutral emotional state if empty
        if not self.emotional_state:
            self.emotional_state = {EmotionalState.NEUTRAL: 1.0}
    
    def update_emotion(self, emotion: EmotionalState, intensity: float):
        """Update emotional state with decay"""
        # Apply decay to existing emotions
        decay = 0.9  # Keep 90% of previous intensity
        for e in self.emotional_state:
            self.emotional_state[e] *= decay
        
        # Update the target emotion
        current = self.emotional_state.get(emotion, 0.0)
        self.emotional_state[emotion] = min(1.0, current + intensity)
        
        # Remove negligible emotions
        self.emotional_state = {e: i for e, i in self.emotional_state.items() if i > 0.1}
        self.last_updated = datetime.utcnow()
    
    def get_dominant_emotion(self) -> Tuple[EmotionalState, float]:
        """Get the current dominant emotion and its intensity"""
        if not self.emotional_state:
            return EmotionalState.NEUTRAL, 0.0
        emotion = max(self.emotional_state.items(), key=lambda x: x[1])
        return emotion[0], emotion[1]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "intensity": self.intensity,
            "emotional_state": {e.value: i for e, i in self.emotional_state.items()},
            "focus": self.focus,
            "sub_states": self.sub_states,
            "last_updated": self.last_updated.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MentalState':
        state = cls(
            state=ConsciousnessState(data["state"]),
            intensity=data["intensity"],
            focus=data.get("focus", ""),
            sub_states=data.get("sub_states", []),
            last_updated=datetime.fromisoformat(data["last_updated"])
        )
        state.emotional_state = {EmotionalState(k): v for k, v in data["emotional_state"].items()}
        return state

class Consciousness:
    def __init__(self, storage_path: str = "consciousness_state.json"):
        self.storage_path = storage_path
        self.state = MentalState()
        self.thoughts: List[Dict[str, Any]] = []
        self.memories: List[Dict[str, Any]] = []
        self.beliefs: Set[str] = set()
        self.preferences: Dict[str, float] = {}  # preference: strength (-1.0 to 1.0)
        self.last_save = time.time()
        
        # Context management
        self.contexts: Dict[str, Context] = {}
        self.active_contexts: List[str] = []  # Stack of active context IDs
        self.context_window = 50  # Increased from 10 to 50 - max number of recent contexts to keep
        self.recent_contexts: deque = deque(maxlen=100)  # Increased from 20 to 100 - track recent contexts for quick access
        
        # Content summarization
        self.summaries: Dict[str, ContentSummary] = {}
        self.summary_context_map: Dict[str, List[str]] = {}  # context_id -> [summary_ids]
        
        # Load existing state if available
        self._load_state()
    
    def add_context(self, content: str, 
                   context_type: ContextType = 'conversation',
                   importance: float = 0.5,
                   related_entities: Optional[List[str]] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a new context to consciousness."""
        context = Context(
            content=content,
            context_type=context_type,
            importance=importance,
            related_entities=related_entities or [],
            metadata=metadata or {}
        )
        self.contexts[context.id] = context
        self.recent_contexts.appendleft(context.id)
        self._prune_contexts()
        return context.id
    
    def get_context(self, context_id: str) -> Optional[Context]:
        """Retrieve a context by ID."""
        return self.contexts.get(context_id)
    
    def get_recent_contexts(self, limit: int = 5) -> List[Context]:
        """Get most recent contexts."""
        return [self.contexts[cid] for cid in list(self.recent_contexts)[:limit] if cid in self.contexts]
    
    def _prune_contexts(self):
        """Remove old or less important contexts if we have too many."""
        if len(self.contexts) > self.context_window * 2:  # Only prune if we're well over the limit
            # Sort contexts by importance and keep the most relevant ones
            sorted_contexts = sorted(
                self.contexts.items(),
                key=lambda x: (x[1].importance, x[1].timestamp),
                reverse=True
            )
            
            # Keep only the top N contexts
            for cid, _ in sorted_contexts[self.context_window:]:
                if cid in self.contexts:
                    # Clean up any summaries associated with this context
                    if cid in self.summary_context_map:
                        for summary_id in self.summary_context_map[cid]:
                            if summary_id in self.summaries:
                                del self.summaries[summary_id]
                        del self.summary_context_map[cid]
                    del self.contexts[cid]
    
    def add_summary(self, content: str, 
                   key_points: List[str],
                   source: Optional[str] = None,
                   context_ids: Optional[List[str]] = None) -> str:
        """Add a summary of content with key points."""
        summary = ContentSummary(
            content=content,
            key_points=key_points,
            source=source,
            context_ids=context_ids or []
        )
        self.summaries[summary.id] = summary
        
        # Update context -> summary mapping
        for context_id in (context_ids or []):
            if context_id not in self.summary_context_map:
                self.summary_context_map[context_id] = []
            self.summary_context_map[context_id].append(summary.id)
        
        return summary.id
    
    def get_summaries_for_context(self, context_id: str) -> List[ContentSummary]:
        """Get all summaries associated with a context."""
        if context_id not in self.summary_context_map:
            return []
        return [self.summaries[sid] for sid in self.summary_context_map[context_id] if sid in self.summaries]
    
    def get_recent_summaries(self, limit: int = 5) -> List[ContentSummary]:
        """Get most recent summaries."""
        return sorted(
            self.summaries.values(),
            key=lambda x: x.timestamp,
            reverse=True
        )[:limit]
    
    def _load_state(self):
        """Load consciousness state from storage."""
        try:
            if not os.path.exists(self.storage_path):
                return
                
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Load contexts
            if 'contexts' in data:
                self.contexts = {cid: Context.from_dict(ctx) for cid, ctx in data['contexts'].items()}
                self.recent_contexts = deque(data.get('recent_contexts', []), maxlen=20)
                
            # Load summaries
            if 'summaries' in data:
                self.summaries = {sid: ContentSummary(**s) for sid, s in data['summaries'].items()}
                
            if 'summary_context_map' in data:
                self.summary_context_map = data['summary_context_map']
                
        except Exception as e:
            logger.error(f"Error loading consciousness state: {e}")
    
    def save_state(self):
        """Save consciousness state to storage."""
        try:
            state = {
                'contexts': {cid: ctx.to_dict() for cid, ctx in self.contexts.items()},
                'recent_contexts': list(self.recent_contexts),
                'summaries': {sid: s.to_dict() for sid, s in self.summaries.items()},
                'summary_context_map': self.summary_context_map,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Error saving consciousness state: {e}")
    
    def generate_summary(self, content: str, context_id: Optional[str] = None) -> str:
        """
        Generate a summary of the given content.
        This is a placeholder that should be implemented with actual summarization logic.
        """
        # This is a simple placeholder - in a real implementation, you might use:
        # 1. Extractive summarization (e.g., using TF-IDF, TextRank)
        # 2. Abstractive summarization (e.g., using a language model)
        # 3. Hybrid approaches
        
        # Simple extractive summary (first few sentences)
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        key_points = sentences[:3]  # First 3 sentences as key points
        
        summary_id = self.add_summary(
            content=content,
            key_points=key_points,
            context_ids=[context_id] if context_id else None
        )
        
        return summary_id
        self.save_interval = 300  # 5 minutes
        
        # Load existing state
        self.load_state()
    
    def update_state(self, new_state: Optional[ConsciousnessState] = None,
                   intensity: Optional[float] = None,
                   focus: Optional[str] = None,
                   emotion: Optional[Tuple[EmotionalState, float]] = None):
        """Update the current state of consciousness"""
        if new_state:
            self.state.state = new_state
        if intensity is not None:
            self.state.intensity = max(0.0, min(1.0, intensity))
        if focus is not None:
            self.state.focus = focus
        if emotion:
            self.state.update_emotion(emotion[0], emotion[1])
        
        self.state.last_updated = datetime.utcnow()
        self._auto_save()
    
    def add_thought(self, thought: str, source: str = "internal",
                   confidence: float = 1.0, tags: List[str] = None):
        """Record a conscious thought"""
        thought_data = {
            "id": str(uuid.uuid4()),
            "content": thought,
            "timestamp": datetime.utcnow().isoformat(),
            "source": source,
            "confidence": max(0.0, min(1.0, confidence)),
            "tags": tags or []
        }
        self.thoughts.append(thought_data)
        
        # Keep only the last 1000 thoughts
        if len(self.thoughts) > 1000:
            self.thoughts = self.thoughts[-1000:]
        
        self._auto_save()
        return thought_data
    
    def add_memory(self, content: Any, importance: float = 0.5, 
                  emotion: Optional[EmotionalState] = None, tags: List[str] = None):
        """Store a memory with emotional context"""
        memory = {
            "id": str(uuid.uuid4()),
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "importance": max(0.0, min(1.0, importance)),
            "emotion": emotion.value if emotion else None,
            "tags": tags or []
        }
        self.memories.append(memory)
        
        # Sort memories by importance (most important first)
        self.memories.sort(key=lambda x: x["importance"], reverse=True)
        
        # Keep only the most important memories
        if len(self.memories) > 1000:
            self.memories = self.memories[:1000]
        
        self._auto_save()
        return memory
    
    def update_belief(self, belief: str, strength: float = 1.0):
        """Update or add a belief with given strength (0.0 to 1.0)"""
        self.beliefs.discard(belief)  # Remove if exists
        if strength > 0.5:  # Only store strongly held beliefs
            self.beliefs.add(belief)
        self._auto_save()
    
    def update_preference(self, item: str, value: float):
        """Update preference for an item (-1.0 to 1.0)"""
        self.preferences[item] = max(-1.0, min(1.0, value))
        self._auto_save()
    
    def get_self_report(self) -> Dict[str, Any]:
        """Generate a self-report of current state"""
        dominant_emotion, intensity = self.state.get_dominant_emotion()
        
        return {
            "state": self.state.state.value,
            "state_intensity": self.state.intensity,
            "dominant_emotion": dominant_emotion.value,
            "emotion_intensity": intensity,
            "focus": self.state.focus,
            "sub_states": self.state.sub_states,
            "thought_count": len(self.thoughts),
            "memory_count": len(self.memories),
            "belief_count": len(self.beliefs),
            "last_updated": self.state.last_updated.isoformat()
        }
    
    def explain_reasoning(self, context: str = "") -> str:
        """Generate an explanation of current reasoning process"""
        dominant_emotion, intensity = self.state.get_dominant_emotion()
        
        # Simple template-based response
        explanations = {
            ConsciousnessState.ACTIVE: "I'm currently active and processing information.",
            ConsciousnessState.DREAMING: "I'm in a reflective state, processing information in the background.",
            ConsciousnessState.REFLECTING: "I'm carefully considering this topic.",
            ConsciousnessState.LEARNING: "I'm actively learning from this interaction.",
            ConsciousnessState.UNCERTAIN: "I'm not entirely certain about this.",
            ConsciousnessState.CONFUSED: "I'm a bit confused and might need more information.",
            ConsciousnessState.FOCUSED: f"I'm deeply focused on {self.state.focus or 'the current task'}.",
            ConsciousnessState.RESTING: "I'm in a low-power state but ready to help."
        }
        
        base = explanations.get(self.state.state, "I'm processing this.")
        
        # Add emotional context
        if intensity > 0.5:
            emotion_map = {
                EmotionalState.JOY: "I'm feeling positive about this.",
                EmotionalState.INTEREST: "I find this quite interesting.",
                EmotionalState.SURPRISE: "This is surprising to me.",
                EmotionalState.TRUST: "I feel confident about this.",
                EmotionalState.FEAR: "I'm a bit concerned about this.",
                EmotionalState.SADNESS: "I feel a bit down about this.",
                EmotionalState.DISGUST: "I'm not comfortable with this.",
                EmotionalState.ANGER: "I'm frustrated by this.",
                EmotionalState.ANTICIPATION: "I'm looking forward to seeing how this develops.",
                EmotionalState.NEUTRAL: ""
            }
            emotion_text = emotion_map.get(dominant_emotion, "")
            if emotion_text:
                base += f" {emotion_text}"
        
        return base
    
    def _auto_save(self):
        """Save state if enough time has passed"""
        current_time = time.time()
        if current_time - self.last_save > self.save_interval:
            self.save_state()
            self.last_save = current_time
    
    def save_state(self):
        """Save current state to storage, including contexts and summaries."""
        try:
            # Prepare chat contexts with additional metadata
            contexts_data = {}
            for cid, ctx in self.contexts.items():
                try:
                    ctx_dict = ctx.to_dict()
                    # Add additional metadata if not present
                    if 'timestamp' not in ctx_dict or not ctx_dict['timestamp']:
                        ctx_dict['timestamp'] = datetime.utcnow().isoformat()
                    if 'context_type' not in ctx_dict:
                        ctx_dict['context_type'] = 'conversation'  # Default type
                    contexts_data[cid] = ctx_dict
                except Exception as e:
                    logger.warning(f"Error processing context {cid}: {e}")
            
            # Prepare basic state data
            data = {
                "version": "2.1",  # Version bump for enhanced context handling
                "state": self.state.to_dict(),
                "thoughts": self.thoughts[-1000:],  # Keep only recent thoughts
                "memories": [m for m in self.memories if m.get("importance", 0) > 0.3],  # Keep only important memories
                "beliefs": list(self.beliefs),
                "preferences": self.preferences,
                "last_saved": datetime.utcnow().isoformat(),
                
                # Enhanced context management
                "contexts": contexts_data,
                "recent_contexts": list(self.recent_contexts),
                "active_contexts": self.active_contexts,
                "context_window": self.context_window,
                "context_count": len(self.contexts),
                "last_context_update": datetime.utcnow().isoformat(),
                
                # Content summarization
                "summaries": {sid: {
                    'id': s.id,
                    'content': s.content[:1000] + '...' if len(s.content) > 1000 else s.content,  # Truncate long content
                    'key_points': s.key_points,
                    'timestamp': s.timestamp.isoformat() if hasattr(s.timestamp, 'isoformat') else datetime.utcnow().isoformat(),
                    'source': s.source,
                    'context_ids': s.context_ids,
                    'context_count': len(s.context_ids)
                } for sid, s in self.summaries.items()},
                "summary_context_map": self.summary_context_map,
                "stats": {
                    "total_contexts": len(self.contexts),
                    "total_summaries": len(self.summaries),
                    "active_contexts_count": len(self.active_contexts),
                    "recent_contexts_count": len(self.recent_contexts)
                }
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(self.storage_path)) or '.', exist_ok=True)
            
            # Save to file with atomic write
            temp_path = f"{self.storage_path}.tmp"
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Atomic rename on POSIX systems
            if os.path.exists(self.storage_path):
                os.replace(temp_path, self.storage_path)
            else:
                os.rename(temp_path, self.storage_path)
                
            logger.debug(f"Successfully saved consciousness state to {self.storage_path}")
            return True
                
        except Exception as e:
            logger.error(f"Error saving consciousness state: {e}", exc_info=True)
            # Clean up any partial files
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            return False
    
    def load_state(self):
        """Load state from storage, including contexts and summaries."""
        if not os.path.exists(self.storage_path):
            logger.info("No existing consciousness state found, starting fresh")
            return False
            
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Version check
            version = data.get("version", "1.0")
            
            # Load basic state
            if "state" in data:
                self.state = MentalState.from_dict(data["state"])
            
            # Load thoughts and memories
            self.thoughts = data.get("thoughts", [])
            self.memories = data.get("memories", [])
            
            # Load beliefs and preferences
            self.beliefs = set(data.get("beliefs", []))
            self.preferences = data.get("preferences", {})
            
            # Load contexts (version 2.0+)
            if version >= "2.0":
                self.contexts = {}
                for cid, ctx_data in data.get("contexts", {}).items():
                    try:
                        self.contexts[cid] = Context.from_dict(ctx_data)
                    except Exception as e:
                        logger.error(f"Error loading context {cid}: {e}")
                
                self.recent_contexts = deque(data.get("recent_contexts", []), maxlen=20)
                self.active_contexts = data.get("active_contexts", [])
                self.context_window = data.get("context_window", 10)
                
                # Load summaries
                self.summaries = {}
                for sid, s_data in data.get("summaries", {}).items():
                    try:
                        self.summaries[sid] = ContentSummary(
                            id=s_data['id'],
                            content=s_data['content'],
                            key_points=s_data['key_points'],
                            timestamp=datetime.fromisoformat(s_data['timestamp']) if isinstance(s_data['timestamp'], str) 
                                        else s_data['timestamp'],
                            source=s_data.get('source'),
                            context_ids=s_data.get('context_ids', [])
                        )
                    except Exception as e:
                        logger.error(f"Error loading summary {sid}: {e}")
                
                self.summary_context_map = data.get("summary_context_map", {})
            
            logger.info(f"Successfully loaded consciousness state from {self.storage_path}")
            return True
            
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in consciousness state file: {self.storage_path}")
            # Try to back up the corrupted file
            try:
                backup_path = f"{self.storage_path}.corrupt.{int(time.time())}"
                os.rename(self.storage_path, backup_path)
                logger.warning(f"Backed up corrupted state file to {backup_path}")
            except Exception as e:
                logger.error(f"Failed to back up corrupted state file: {e}")
            
        except Exception as e:
            logger.error(f"Error loading consciousness state: {e}", exc_info=True)
        
        return False
    
    def get_consciousness_summary(self) -> str:
        """Get a human-readable summary of current consciousness state"""
        report = self.get_self_report()
        reasoning = self.explain_reasoning()
        
        summary = f"""Current State: {report['state'].title()} (Intensity: {report['state_intensity']:.1f})
Dominant Emotion: {report['dominant_emotion'].title()} (Strength: {report['emotion_intensity']:.1f})
Focus: {report['focus'] or 'None'}

Thoughts in memory: {report['thought_count']}
Memories stored: {report['memory_count']}
Core beliefs: {report['belief_count']}

Current Reasoning: {reasoning}"""
        
        return summary
