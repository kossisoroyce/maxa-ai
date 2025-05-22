import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Union, TypedDict, Literal
import json
import logging
from enum import Enum, auto
import numpy as np
from collections import defaultdict, deque
from uuid import uuid4

class Gender(Enum):
    MALE = "male"
    FEMALE = "female"
    NON_BINARY = "non_binary"
    OTHER = "other"
    PREFER_NOT_TO_SAY = "prefer_not_to_say"

class RelationshipStatus(Enum):
    SINGLE = "single"
    IN_A_RELATIONSHIP = "in_a_relationship"
    ENGAGED = "engaged"
    MARRIED = "married"
    IN_A_CIVIL_PARTNERSHIP = "in_a_civil_partnership"
    IN_A_DOMESTIC_PARTNERSHIP = "in_a_domestic_partnership"
    IN_AN_OPEN_RELATIONSHIP = "in_an_open_relationship"
    ITS_COMPLICATED = "its_complicated"
    SEPARATED = "separated"
    DIVORCED = "divorced"
    WIDOWED = "widowed"

class EducationLevel(Enum):
    SOME_HIGH_SCHOOL = "some_high_school"
    HIGH_SCHOOL = "high_school"
    SOME_COLLEGE = "some_college"
    ASSOCIATE_DEGREE = "associate_degree"
    BACHELORS_DEGREE = "bachelors_degree"
    MASTERS_DEGREE = "masters_degree"
    PROFESSIONAL_DEGREE = "professional_degree"
    DOCTORATE = "doctorate"
    OTHER = "other"

@dataclass
class Location:
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    timezone: Optional[str] = None
    coordinates: Optional[tuple[float, float]] = None  # (latitude, longitude)

@dataclass
class ContactInfo:
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    social_media: Dict[str, str] = field(default_factory=dict)  # platform -> username

@dataclass
class HealthInfo:
    blood_type: Optional[str] = None
    allergies: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None

@dataclass
class ProfessionalInfo:
    occupation: Optional[str] = None
    employer: Optional[str] = None
    industry: Optional[str] = None
    education: List[Dict[str, Any]] = field(default_factory=list)  # List of education entries
    skills: List[str] = field(default_factory=list)
    work_history: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class PersonalPreferences:
    favorite_foods: List[str] = field(default_factory=list)
    favorite_drinks: List[str] = field(default_factory=list)
    favorite_books: List[str] = field(default_factory=list)
    favorite_movies: List[str] = field(default_factory=list)
    favorite_music: List[str] = field(default_factory=list)
    hobbies: List[str] = field(default_factory=list)
    interests: List[str] = field(default_factory=list)
    dislikes: List[str] = field(default_factory=list)

@dataclass
class LifeGoals:
    career_goals: List[str] = field(default_factory=list)
    personal_goals: List[str] = field(default_factory=list)
    travel_goals: List[str] = field(default_factory=list)
    learning_goals: List[str] = field(default_factory=list)
    financial_goals: List[str] = field(default_factory=list)
    bucket_list: List[str] = field(default_factory=list)

@dataclass
class UserProfile:
    """Comprehensive user profile information."""
    # Basic Information
    user_id: str = field(default_factory=lambda: str(uuid4()))
    first_name: Optional[str] = None
    middle_name: Optional[str] = None
    last_name: Optional[str] = None
    preferred_name: Optional[str] = None
    date_of_birth: Optional[datetime] = None
    gender: Optional[Gender] = None
    languages: List[str] = field(default_factory=list)
    nationality: Optional[str] = None
    
    # Contact and Location
    contact: ContactInfo = field(default_factory=ContactInfo)
    current_location: Location = field(default_factory=Location)
    hometown: Optional[Location] = None
    
    # Personal Details
    relationship_status: Optional[RelationshipStatus] = None
    family_members: List[Dict[str, Any]] = field(default_factory=list)
    pets: List[Dict[str, Any]] = field(default_factory=list)
    
    # Professional Information
    professional: ProfessionalInfo = field(default_factory=ProfessionalInfo)
    
    # Health Information
    health: HealthInfo = field(default_factory=HealthInfo)
    
    # Preferences
    preferences: PersonalPreferences = field(default_factory=PersonalPreferences)
    
    # Life Goals and Aspirations
    goals: LifeGoals = field(default_factory=LifeGoals)
    
    # Additional Information
    beliefs: Dict[str, Any] = field(default_factory=dict)  # Religious, political, etc.
    personality_traits: List[str] = field(default_factory=list)
    values: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict:
        """Convert the profile to a dictionary for serialization."""
        def serialize(obj):
            if isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif hasattr(obj, 'to_dict'):
                return obj.to_dict()
            elif isinstance(obj, list):
                return [serialize(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            return obj
            
        return {k: serialize(v) for k, v in asdict(self).items()}
    
    @classmethod
    def from_dict(cls, data: dict) -> 'UserProfile':
        """Create a UserProfile from a dictionary."""
        # Handle nested objects
        if 'contact' in data and isinstance(data['contact'], dict):
            data['contact'] = ContactInfo(**data['contact'])
        if 'current_location' in data and isinstance(data['current_location'], dict):
            data['current_location'] = Location(**data['current_location'])
        if 'hometown' in data and isinstance(data['hometown'], dict):
            data['hometown'] = Location(**data['hometown'])
        if 'health' in data and isinstance(data['health'], dict):
            data['health'] = HealthInfo(**data['health'])
        if 'professional' in data and isinstance(data['professional'], dict):
            data['professional'] = ProfessionalInfo(**data['professional'])
        if 'preferences' in data and isinstance(data['preferences'], dict):
            data['preferences'] = PersonalPreferences(**data['preferences'])
        if 'goals' in data and isinstance(data['goals'], dict):
            data['goals'] = LifeGoals(**data['goals'])
            
        # Handle enums
        if 'gender' in data and data['gender'] is not None:
            data['gender'] = Gender(data['gender'])
        if 'relationship_status' in data and data['relationship_status'] is not None:
            data['relationship_status'] = RelationshipStatus(data['relationship_status'])
            
        # Handle dates
        date_fields = ['date_of_birth', 'created_at', 'updated_at']
        for field in date_fields:
            if field in data and data[field] is not None and not isinstance(data[field], datetime):
                data[field] = datetime.fromisoformat(data[field])
                
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})

logger = logging.getLogger(__name__)

class UserSentiment(Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    MIXED = "mixed"

class InteractionType(Enum):
    GREETING = "greeting"
    QUESTION = "question"
    COMMAND = "command"
    FEEDBACK = "feedback"
    CHAT = "chat"
    ERROR = "error"

@dataclass
class UserPreference:
    topic: str
    preference: float  # -1.0 (dislike) to 1.0 (like)
    confidence: float  # 0.0 to 1.0
    last_updated: datetime = field(default_factory=datetime.utcnow)

@dataclass
class UserKnowledge:
    facts: Dict[str, Any]
    last_updated: datetime = field(default_factory=datetime.utcnow)

@dataclass
class Interaction:
    interaction_type: InteractionType
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    sentiment: Optional[UserSentiment] = None
    response: Optional[str] = None
    response_sentiment: Optional[UserSentiment] = None

class TheoryOfMind:
    def __init__(self, user_id: str, storage_path: str = "tom_data.json"):
        # Initialize basic attributes
        self.user_id = user_id
        self.storage_path = storage_path
        
        # Initialize collections
        self.interactions: List[Interaction] = []
        self.conversation_context = deque(maxlen=50)  # Last 50 messages
        self.preferences: Dict[str, UserPreference] = {}
        self.knowledge = UserKnowledge({})
        
        # User Profile - Comprehensive user information
        self.profile: UserProfile = UserProfile(user_id=user_id)
        
        # Social dynamics
        self.relationship_strength = 0.5  # 0.0 to 1.0
        self.trust_level = 0.5  # 0.0 to 1.0
        self.emotional_bond = 0.3  # 0.0 to 1.0
        
        # Load any existing state
        self._load_state()  # This will override defaults with saved data
        
    def get_recent_interactions(self, limit: int = 5) -> List[Interaction]:
        """
        Get the most recent interactions.
        
        Args:
            limit: Maximum number of interactions to return
            
        Returns:
            List of Interaction objects, most recent first
        """
        return sorted(
            [i for i in self.interactions if i.content or i.response],
            key=lambda x: x.timestamp,
            reverse=True
        )[:limit]
        
    def update_profile(self, **updates) -> bool:
        """
        Update the user's profile with the given fields.
        
        Args:
            **updates: Key-value pairs of profile fields to update
            
        Returns:
            bool: True if successful, False otherwise
            
        Example:
            update_profile(
                first_name="John",
                last_name="Doe",
                date_of_birth=datetime(1990, 1, 1),
                gender=Gender.MALE,
                contact={
                    'email': 'john.doe@example.com',
                    'phone': '+1234567890'
                },
                preferences={
                    'favorite_foods': ['Pizza', 'Sushi'],
                    'hobbies': ['Hiking', 'Reading']
                }
            )
        """
        try:
            # Handle nested updates
            for key, value in updates.items():
                if hasattr(self.profile, key):
                    # If it's a nested object with its own update method
                    if hasattr(getattr(self.profile, key), 'update') and isinstance(value, dict):
                        getattr(self.profile, key).update(**value)
                    else:
                        setattr(self.profile, key, value)
                # Handle direct updates to nested objects
                elif '.' in key:
                    obj_name, attr = key.split('.', 1)
                    if hasattr(self.profile, obj_name):
                        obj = getattr(self.profile, obj_name)
                        if hasattr(obj, attr):
                            setattr(obj, attr, value)
            
            # Update the timestamp
            self.profile.updated_at = datetime.utcnow()
            
            # Save the changes
            self._save_state()
            return True
            
        except Exception as e:
            logger.error(f"Error updating profile: {e}")
            return False
    
    def get_profile(self) -> UserProfile:
        """
        Get the complete user profile.
        
        Returns:
            UserProfile: The complete user profile object
        """
        return self.profile
    
    def get_profile_summary(self) -> dict:
        """
        Get a summary of the user profile.
        
        Returns:
            dict: A dictionary with key profile information
        """
        profile = self.profile
        return {
            'name': f"{profile.first_name or ''} {profile.last_name or ''}".strip() or "Not specified",
            'age': (datetime.utcnow().year - profile.date_of_birth.year) if profile.date_of_birth else None,
            'location': f"{profile.current_location.city or 'Unknown'}, {profile.current_location.country or ''}".strip(',').strip() if profile.current_location else "Unknown",
            'occupation': profile.professional.occupation or "Not specified",
            'interests': profile.preferences.interests[:5] if profile.preferences.interests else [],
            'goals': profile.goals.personal_goals[:3] if profile.goals.personal_goals else []
        }
    
    # For backward compatibility
    def set_user_name(self, name: str) -> bool:
        """
        Set the user's first name (maintained for backward compatibility).
        
        Args:
            name: The user's first name
            
        Returns:
            bool: True if successful, False otherwise
        """
        return self.update_profile(first_name=name)
    
    def get_user_name(self) -> Optional[str]:
        """
        Get the user's name (maintained for backward compatibility).
        
        Returns:
            Optional[str]: The user's first name if set, None otherwise
        """
        return self.profile.first_name or self.profile.preferred_name
    
    def save_state(self):
        """Save current state to disk"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(self.storage_path)), exist_ok=True)
            
            # Ensure profile exists
            if not hasattr(self, 'profile'):
                self.profile = UserProfile(user_id=self.user_id)
                
            # Save the state using the _save_state method which handles atomic writes
            self._save_state()
                
        except Exception as e:
            logger.error(f"Error saving Theory of Mind state: {e}")
            raise
    
    def _load_state(self):
        """Load saved state from disk"""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    
                    # Load interactions
                    if 'interactions' in data:
                        self.interactions = [
                            Interaction(
                                interaction_type=InteractionType(i.get('interaction_type', 'chat')),
                                content=i.get('content', ''),
                                timestamp=datetime.fromisoformat(i['timestamp']) if 'timestamp' in i else datetime.utcnow(),
                                sentiment=UserSentiment(i['sentiment']) if i.get('sentiment') else None,
                                response=i.get('response'),
                                response_sentiment=UserSentiment(i['response_sentiment']) if i.get('response_sentiment') else None
                            )
                            for i in data.get('interactions', [])
                        ]
                    
                    # Load social dynamics
                    self.relationship_strength = data.get('relationship_strength', 0.5)
                    self.trust_level = data.get('trust_level', 0.5)
                    self.emotional_bond = data.get('emotional_bond', 0.3)
                    
                    # Load preferences
                    self.preferences = {
                        p['topic']: UserPreference(
                            topic=p['topic'],
                            preference=p['preference'],
                            confidence=p['confidence'],
                            last_updated=datetime.fromisoformat(p['last_updated'])
                        )
                        for p in data.get('preferences', [])
                    }
                    
                    # Load user profile if it exists
                    if 'profile' in data and data['profile']:
                        try:
                            self.profile = UserProfile.from_dict(data['profile'])
                        except Exception as e:
                            logger.error(f"Error loading user profile: {e}")
                            self.profile = UserProfile(user_id=self.user_id)
                    else:
                        # If no profile exists, create a new one
                        self.profile = UserProfile(user_id=self.user_id)
                    
                    # For backward compatibility, load old user_name if profile name is not set
                    if (not self.profile.first_name and 'user_name' in data and data['user_name']):
                        self.profile.first_name = data['user_name']
                        # Save the updated profile
                        self._save_state()
                    
                    # Load knowledge
                    if 'knowledge' in data:
                        self.knowledge = UserKnowledge(
                            facts=data['knowledge'].get('facts', {}),
                            last_updated=datetime.fromisoformat(data['knowledge'].get('last_updated', datetime.utcnow().isoformat()))
                        )
                    
        except Exception as e:
            logger.error(f"Error loading ToM state: {e}")
            self.interactions = []
            self.preferences = {}
            self.knowledge = UserKnowledge({})
            self.profile = UserProfile(user_id=self.user_id)
    
    def _save_state(self):
        """Save current state to disk"""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.storage_path)), exist_ok=True)
            
            # Prepare data for JSON serialization
            data = {
                'interactions': [
                    {
                        'interaction_type': i.interaction_type.value,
                        'content': i.content,
                        'timestamp': i.timestamp.isoformat(),
                        'sentiment': i.sentiment.value if i.sentiment else None,
                        'response': i.response,
                        'response_sentiment': i.response_sentiment.value if i.response_sentiment else None
                    }
                    for i in self.interactions
                ],
                'profile': self.profile.to_dict() if hasattr(self, 'profile') else {},
                'relationship_strength': self.relationship_strength,
                'trust_level': self.trust_level,
                'emotional_bond': self.emotional_bond,
                'preferences': [
                    {
                        'topic': p.topic,
                        'preference': p.preference,
                        'confidence': p.confidence,
                        'last_updated': p.last_updated.isoformat()
                    }
                    for p in self.preferences.values()
                ],
                'knowledge': {
                    'facts': self.knowledge.facts,
                    'last_updated': self.knowledge.last_updated.isoformat()
                }
            }
            
            # Write to a temporary file first, then do an atomic rename
            temp_path = f"{self.storage_path}.tmp"
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            # Atomic rename on POSIX systems
            if os.path.exists(temp_path):
                if os.path.exists(self.storage_path):
                    os.replace(temp_path, self.storage_path)
                else:
                    os.rename(temp_path, self.storage_path)
        except Exception as e:
            logger.error(f"Error saving ToM state: {e}")
    
    def add_interaction(self, interaction: Interaction):
        """
        Add a new interaction to the history.
        
        Args:
            interaction: The interaction to add
        """
        self.interactions.append(interaction)
        self.conversation_context.append(interaction)
        self.analyze_interaction(interaction)
        
        # Update relationship metrics
        if interaction.sentiment == UserSentiment.POSITIVE:
            self.relationship_strength = min(1.0, self.relationship_strength + 0.02)
            self.trust_level = min(1.0, self.trust_level + 0.01)
        elif interaction.sentiment == UserSentiment.NEGATIVE:
            self.relationship_strength = max(0.0, self.relationship_strength - 0.03)
            self.trust_level = max(0.0, self.trust_level - 0.02)
        
        self.save_data()
    
    def clear_interactions(self):
        """Clear all interaction history"""
        # Conversation history and context - increased from 10 to 50 messages
        self.conversation_context = deque(maxlen=50)  # Last 50 messages
        self._save_state()
        
        # User modeling
        self.preferences = {}
        self.knowledge = UserKnowledge({})
        
        # Social dynamics
        self.relationship_strength = 0.5  # 0.0 to 1.0
        self.trust_level = 0.5  # 0.0 to 1.0
        self.emotional_bond = 0.3  # 0.0 to 1.0
        
        # Load existing data
        self._load_state()
    
    def update_preference(self, topic: str, delta: float, confidence: float = 0.7):
        """Update user preference for a topic"""
        if topic in self.preferences:
            pref = self.preferences[topic]
            new_pref = pref.preference + delta * confidence
            self.preferences[topic] = UserPreference(
                topic=topic,
                preference=max(-1.0, min(1.0, new_pref)),
                confidence=min(1.0, pref.confidence + confidence * 0.1)
            )
        else:
            self.preferences[topic] = UserPreference(
                topic=topic,
                preference=max(-1.0, min(1.0, delta)),
                confidence=confidence
            )
        self.save_data()
    
    def get_preference(self, topic: str) -> float:
        """Get user preference for a topic (-1.0 to 1.0)"""
        return self.preferences.get(topic, UserPreference(topic, 0.0, 0.0)).preference
    
    def add_interaction(self, interaction: Interaction):
        """
        Add a new interaction to the history.
        
        Args:
            interaction: The interaction to add
        """
        self.interactions.append(interaction)
        self.conversation_context.append(interaction)
        
        # Update relationship metrics based on sentiment
        if hasattr(interaction, 'sentiment') and interaction.sentiment:
            if interaction.sentiment == UserSentiment.POSITIVE:
                self.relationship_strength = min(1.0, self.relationship_strength + 0.02)
                self.trust_level = min(1.0, self.trust_level + 0.01)
            elif interaction.sentiment == UserSentiment.NEGATIVE:
                self.relationship_strength = max(0.0, self.relationship_strength - 0.03)
                self.trust_level = max(0.0, self.trust_level - 0.02)
        
        # Save the updated state
        self._save_state()
    
    def analyze_interaction(self, interaction: Interaction):
        """Analyze an interaction for preferences and knowledge"""
        # Simple keyword-based analysis (could be enhanced with NLP)
        content = interaction.content.lower()
        
        # Detect preferences
        like_phrases = ["i like", "i love", "i enjoy", "i prefer"]
        dislike_phrases = ["i don't like", "i hate", "i dislike", "i can't stand"]
        
        for phrase in like_phrases:
            if phrase in content:
                topic = content.split(phrase, 1)[1].strip().split()[0]  # Get next word
                self.update_preference(topic, 0.3)
        
        for phrase in dislike_phrases:
            if phrase in content:
                topic = content.split(phrase, 1)[1].strip().split()[0]
                self.update_preference(topic, -0.3)
    
    def predict_intent(self, text: str) -> str:
        """
        Predict user intent from text using a combination of pattern matching and context.
        
        Args:
            text: The input text to analyze
            
        Returns:
            str: The predicted intent type
        """
        if not text:
            return "chat"
            
        text = text.lower().strip()
        
        # Check for greetings
        if any(word in text for word in ["hi", "hello", "hey", "greetings", "hi there"]):
            return "greeting"
            
        # Check for questions
        if "?" in text or any(word in text.split() for word in ["what", "how", "why", "when", "where", "who", "which"]):
            return "question"
            
        # Check for requests
        if any(word in text for word in ["please", "could you", "would you", "can you", "will you"]):
            return "request"
            
        # Check for conversation continuation
        if self.interactions:
            last_interaction = self.interactions[-1]
            if last_interaction.interaction_type == "question" and any(word in text for word in ["yes", "no", "maybe", "i think", "probably"]):
                return "answer"
                
        # Default to chat
        return "chat"
    
    def get_conversation_summary(self, num_messages: int = 5) -> str:
        """Get a summary of recent conversation context"""
        recent = list(self.conversation_context)[-num_messages:]
        return "\n".join(f"{'User' if i % 2 == 0 else 'AI'}: {msg.content}" 
                         for i, msg in enumerate(recent))
    
    def save_data(self):
        """Save user model to storage"""
        data = {
            "user_id": self.user_id,
            "preferences": [{
                "topic": p.topic,
                "preference": p.preference,
                "confidence": p.confidence,
                "last_updated": p.last_updated.isoformat()
            } for p in self.preferences.values()],
            "knowledge": {
                "facts": self.knowledge.facts,
                "last_updated": self.knowledge.last_updated.isoformat()
            },
            "relationship_strength": self.relationship_strength,
            "trust_level": self.trust_level,
            "emotional_bond": self.emotional_bond,
            "interaction_count": len(self.interactions)
        }
        
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving theory of mind data: {e}")
    
    def load_data(self):
        """Load user model from storage"""
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                
                # Load preferences
                self.preferences = {}
                for p in data.get("preferences", []):
                    pref = UserPreference(
                        topic=p["topic"],
                        preference=p["preference"],
                        confidence=p["confidence"],
                        last_updated=datetime.fromisoformat(p["last_updated"])
                    )
                    self.preferences[p["topic"]] = pref
                
                # Load knowledge
                knowledge = data.get("knowledge", {})
                self.knowledge = UserKnowledge(
                    facts=knowledge.get("facts", {}),
                    last_updated=datetime.fromisoformat(knowledge.get("last_updated", datetime.utcnow().isoformat()))
                )
                
                # Load relationship metrics
                self.relationship_strength = data.get("relationship_strength", 0.5)
                self.trust_level = data.get("trust_level", 0.5)
                self.emotional_bond = data.get("emotional_bond", 0.3)
                
        except FileNotFoundError:
            logger.info("No existing theory of mind data found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading theory of mind data: {e}")
    
    def get_relationship_status(self) -> Dict[str, float]:
        """Get current relationship metrics"""
        return {
            "relationship_strength": self.relationship_strength,
            "trust_level": self.trust_level,
            "emotional_bond": self.emotional_bond
        }
