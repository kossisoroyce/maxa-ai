import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
import os
import time
import uuid
from pathlib import Path

# Import our modules
from goal_system import GoalSystem, Goal, SubGoal, GoalStatus, GoalPriority
from theory_of_mind import TheoryOfMind, UserSentiment, InteractionType, Interaction
from temporal_awareness import TemporalAwareness, ScheduledEvent, TimeUnit
from consciousness import Consciousness, ConsciousnessState, EmotionalState, MentalState
from autonomous_goals import AutonomousGoalGenerator

class AgentCore:
    """
    Core integration of all cognitive modules for the AI agent.
    """
    def __init__(self, user_id: str = "default_user", data_dir: str = "agent_data"):
        # Set up data directory
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize core components
        self.goal_system = GoalSystem(storage_path=str(self.data_dir / "goals.json"))
        self.theory_of_mind = TheoryOfMind(
            user_id=user_id,
            storage_path=str(self.data_dir / f"tom_{user_id}.json")
        )
        self.temporal_awareness = TemporalAwareness(
            timezone_str="UTC",
            storage_path=str(self.data_dir / "temporal.json")
        )
        self.consciousness = Consciousness(
            storage_path=str(self.data_dir / "consciousness.json")
        )
        
        # State tracking
        self.last_interaction_time = datetime.utcnow()
        self.interaction_count = 0
        
        # Initialize autonomous goal generator
        self.goal_generator = AutonomousGoalGenerator(self)
        
        # Initialize last save time
        self.last_save = time.time()
        
        # Initialize with default goals if none exist
        self._initialize_default_goals()
        
        # Process any autonomous goals that should be created at startup
        self._process_autonomous_goals()
    
    def _initialize_default_goals(self):
        """Initialize with some default goals if none exist"""
        if not self.goal_system.goals:
            # Add a default goal to learn about the user
            learn_goal = Goal(
                id=self.goal_system.generate_goal_id(),
                title="Learn About User",
                description="Gather information about the user's preferences and personality",
                category="learning",
                priority=GoalPriority.HIGH.value,
                progress=0.1
            )
            
            # Add some subgoals
            learn_goal.add_subgoal(SubGoal(
                description="Ask about user's interests",
                status=GoalStatus.PENDING,
                priority=GoalPriority.MEDIUM.value
            ))
            
            learn_goal.add_subgoal(SubGoal(
                description="Identify user's communication style",
                status=GoalStatus.PENDING,
                priority=GoalPriority.MEDIUM.value
            ))
            
            self.goal_system.add_goal(learn_goal)
            
            # Add a maintenance goal
            maint_goal = Goal(
                id=self.goal_system.generate_goal_id(),
                title="System Maintenance",
                description="Perform routine system checks and updates",
                category="maintenance",
                priority=GoalPriority.MEDIUM.value,
                progress=0.0
            )
            
            self.goal_system.add_goal(maint_goal)
    
    def process_input(self, user_input: str, context: Optional[Dict] = None) -> Dict:
        """
        Process user input and generate a response with context awareness.
        
        Args:
            user_input: The user's input text
            context: Additional context for the interaction
            
        Returns:
            Dict containing response and metadata
        """
        try:
            # Update interaction tracking
            self.interaction_count += 1
            self.last_interaction_time = datetime.utcnow()
            
            # Create a new context for this interaction
            interaction_id = str(uuid.uuid4())
            interaction_ctx = self.consciousness.add_context(
                content=user_input,
                context_type='conversation',
                importance=0.7,
                metadata={
                    'interaction_count': self.interaction_count,
                    'source': 'user_input',
                    'timestamp': datetime.utcnow().isoformat()
                }
            )
            
            # Analyze sentiment and interaction type
            sentiment = self._analyze_sentiment(user_input)
            interaction_type = self._determine_interaction_type(user_input, sentiment)
            
            # Check for goal-related interactions
            if self._is_goal_creation_request(user_input):
                goal_result = self._create_goal_from_natural_language(user_input)
                # Add goal creation to context
                if goal_result and 'goal' in goal_result:
                    self.consciousness.add_context(
                        content=f"Created goal: {goal_result['goal'].title}",
                        context_type='task',
                        importance=0.8,
                        related_entities=[f"goal:{goal_result['goal'].id}"],
                        metadata={'goal_id': goal_result['goal'].id}
                    )
                # Return a properly formatted response
                if goal_result['status'] == 'success':
                    return {"response": goal_result['message']}
                else:
                    return {"response": goal_result.get('message', 'Failed to create goal.')}
            
            # Update consciousness state with the interaction
            self._update_consciousness(user_input, interaction_type, sentiment)
            
            # Generate response with context awareness
            response = self._generate_response(user_input, interaction_type, sentiment)
            
            # Add response to context
            if response and 'response' in response:
                self.consciousness.add_context(
                    content=response['response'],
                    context_type='response',
                    importance=0.6,
                    metadata={
                        'interaction_id': interaction_id,
                        'response_type': interaction_type,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                )
                
                # Generate and store a summary of the interaction
                interaction_summary = f"User: {user_input}\n\nAssistant: {response['response']}"
                self.consciousness.generate_summary(
                    content=interaction_summary,
                    context_id=interaction_ctx
                )
            
            # Process any autonomous goals that might have been triggered
            self._process_autonomous_goals()
            
            # Save state periodically
            if time.time() - self.last_save > 300:  # 5 minutes
                self._save_states()
                self.last_save = time.time()
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing input: {e}", exc_info=True)
            return {"response": "I encountered an error processing your request. Please try again."}
    
    def _create_goal_from_natural_language(self, text: str) -> Dict:
        """
        Create a new goal from natural language input.
        
        Args:
            text: User's natural language input describing the goal
            
        Returns:
            Dict containing the created goal and status
        """
        try:
            # Default values
            title = "Untitled Goal"
            description = text.strip()
            category = "user_defined"
            priority = GoalPriority.MEDIUM
            
            # Try to extract priority from text
            text_lower = text.lower()
            if any(word in text_lower for word in ["important", "critical", "urgent", "asap"]):
                priority = GoalPriority.HIGH
            elif any(word in text_lower for word in ["low", "not urgent", "whenever"]):
                priority = GoalPriority.LOW
                
            # Try to extract category from text or use a default
            categories = ["work", "personal", "health", "learning", "entertainment"]
            for cat in categories:
                if cat in text_lower:
                    category = cat
                    break
            
            # Create a simple title by cleaning up the description
            if len(description) > 50:
                title = description[:47] + "..."
            else:
                title = description
            
            # Create the goal
            goal = Goal(
                id=self.goal_system.generate_goal_id(),
                title=title,
                description=description,
                category=category,
                priority=priority,
                status=GoalStatus.PENDING,
                progress=0.0
            )
            
            # Add the goal to the system
            self.goal_system.add_goal(goal)
            
            # Log the goal creation
            self.logger.info(f"Created new goal: {goal.title} (ID: {goal.id})")
            
            return {
                "status": "success",
                "message": f"Created new goal: {goal.title}",
                "goal": goal,
                "goal_id": goal.id
            }
            
        except Exception as e:
            self.logger.error(f"Error creating goal from natural language: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Failed to create goal: {str(e)}",
                "goal": None
            }
    
    def _is_goal_creation_request(self, text: str) -> bool:
        """Check if the user is trying to create a goal."""
        goal_keywords = [
            'create a goal', 'set a goal', 'new goal', 'add goal',
            'i want to', 'i need to', 'remind me to', 'i should',
            'goal to', 'goal is to', 'can you help me', 'help me'
        ]
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in goal_keywords)

    def _determine_interaction_type(self, text: str, sentiment: Any) -> str:
        """Determine the type of interaction based on content and sentiment."""
        text_lower = text.lower()
        
        # Check for question patterns
        if '?' in text_lower or any(q in text_lower for q in ['what', 'how', 'why', 'when', 'where', 'who', 'which']):
            return 'question'
            
        # Check for command patterns
        if any(cmd in text_lower for cmd in ['do this', 'please', 'can you', 'would you', 'help me']):
            return 'command'
            
        # Check for information sharing
        if any(phrase in text_lower for phrase in ['i think', 'i feel', 'i believe', 'in my opinion']):
            return 'sharing'
            
        # Default to generic interaction
        return 'conversation'
            
    def _analyze_sentiment(self, text: str) -> UserSentiment:
        """Analyze sentiment of the input text"""
        # This is a simple implementation - in practice, you'd use an NLP model
        text = text.lower()
        positive_words = ["happy", "great", "thanks", "awesome", "good", "like", "love"]
        negative_words = ["sad", "bad", "hate", "terrible", "awful", "worst"]
        
        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)
        
        if pos_count > neg_count:
            return UserSentiment.POSITIVE
        elif neg_count > pos_count:
            return UserSentiment.NEGATIVE
        else:
            return UserSentiment.NEUTRAL
    
    def _update_consciousness(self, text: str, interaction_type: str, sentiment: Any):
        """Update consciousness based on interaction"""
        try:
            # Update emotional state based on sentiment
            if sentiment and hasattr(sentiment, 'label'):
                emotion_map = {
                    'positive': EmotionalState.JOY,
                    'negative': EmotionalState.SADNESS,
                    'neutral': EmotionalState.NEUTRAL
                }
                emotion = emotion_map.get(sentiment.label.lower(), EmotionalState.NEUTRAL)
                self.consciousness.state.update_emotion(emotion, 0.3)
                
                # Add emotional context
                self.consciousness.add_context(
                    content=f"User interaction with {sentiment.label} sentiment: {text[:100]}...",
                    context_type='emotion',
                    importance=0.5,
                    metadata={
                        'sentiment': sentiment.label,
                        'interaction_type': interaction_type,
                        'emotion': emotion.value
                    }
                )
                
            # Update consciousness state based on interaction type
            if interaction_type == 'question':
                self.consciousness.state.state = ConsciousnessState.ACTIVE
                self.consciousness.state.focus = "answering_question"
            elif interaction_type == 'command':
                self.consciousness.state.state = ConsciousnessState.FOCUSED
                self.consciousness.state.focus = "executing_command"
            else:
                self.consciousness.state.state = ConsciousnessState.ACTIVE
                self.consciousness.state.focus = "conversation"
                
            # Update last updated timestamp
            self.consciousness.state.last_updated = datetime.utcnow()
            
        except Exception as e:
            self.logger.error(f"Error updating consciousness: {e}", exc_info=True)
    
    def _generate_response(self, user_input: str, interaction_type: str, sentiment: Any) -> Dict:
        """
        Generate a response using GPT-4-turbo with conversation context and emotional awareness.
        """
        # First, check if this is a goal creation request
        if self._is_goal_creation_request(user_input):
            goal, response = self._create_goal_from_natural_language(user_input)
            if goal:
                # Add a thought about the new goal
                self.consciousness.add_context(
                    content=f"Created a new goal: {goal.title}",
                    context_type='task',
                    importance=0.8,
                    related_entities=[f"goal:{goal.id}"],
                    metadata={'goal_id': goal.id}
                )
            return {"response": response}
        
        # If not a goal creation request, proceed with normal response generation
        import openai
        from openai import OpenAI
        import os
        from dotenv import load_dotenv
        
        # Load environment variables
        load_dotenv()
        
        # Initialize OpenAI client
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        try:
            # Prepare conversation context
            conversation = self.theory_of_mind.get_recent_interactions(limit=5)  # Get last 5 interactions
            
            # Get current goals for context
            current_goals = self.goal_system.get_goals_by_status(GoalStatus.IN_PROGRESS)
            goals_context = "\n".join([f"- {goal.title} ({goal.progress*100:.0f}%)" for goal in current_goals[:3]])
            
            # Prepare system message with personality and context
            system_message = {
                "role": "system",
                "content": f"""You are Maxa, an AI assistant with a warm and engaging personality. 
                Your current emotional state is: {self.consciousness.get_self_report()}
                
                Current active goals:
                {goals_context if goals_context else 'No active goals'}
                
                You're having a conversation with a user. Be natural, empathetic, and maintain context.
                Keep responses concise but meaningful. Show personality and emotional intelligence.
                If the user wants to create a new goal, ask them for more details."""
            }
            
            # Prepare conversation history
            messages = [system_message]
            for interaction in conversation:
                if interaction.content:
                    messages.append({"role": "user", "content": interaction.content})
                if interaction.response:
                    messages.append({"role": "assistant", "content": interaction.response})
            
            # Add current user input
            messages.append({"role": "user", "content": user_input})
            
            # Check if user is telling us their name
            name_phrases = ["my name is", "i'm called", "call me", "i am"]
            if any(phrase in user_input.lower() for phrase in name_phrases):
                # Extract name from the input
                name = None
                for phrase in name_phrases:
                    if phrase in user_input.lower():
                        name = user_input.lower().split(phrase, 1)[1].strip().split()[0].capitalize()
                        break
                
                if name:
                    # Save the name to the user profile
                    self.theory_of_mind.update_profile(first_name=name)
                    self.theory_of_mind.save_state()
            
            # Generate response using GPT-4-turbo
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=messages,
                max_tokens=2000,  # Increased from 200 to 2000 for more detailed responses
                temperature=0.7,  # Balanced between creativity and coherence
                top_p=0.9,  # Controls diversity of responses
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # If we have the user's name, personalize the response
            user_name = self.theory_of_mind.get_user_name()
            if user_name and any(phrase in response_text.lower() for phrase in ["you can call me", "my name is"]):
                # If the model is introducing itself, don't add the name
                pass
            elif user_name and user_name.lower() not in response_text.lower() and not any(phrase in response_text.lower() for phrase in ["what's your name", "who are you"]):
                # Add the user's name to the response if it's not already there and we're not asking for their name
                response_text = response_text.replace("!", f", {user_name}!", 1) if "!" in response_text else f"{response_text}, {user_name}."
            
            # Check if the assistant is asking for more details about a goal
            if any(phrase in response_text.lower() for phrase in ["what goal", "which goal", "tell me more", "more details"]):
                if self._is_goal_creation_request(' '.join(conversation[-2:])):
                    # If the last few messages were about creating a goal, suggest the format
                    response_text += "\n\n(You can say something like: 'I want to learn Python for data science by the end of the year' or 'Add a goal to exercise 3 times a week')"
            
            return {"response": response_text}
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            # Fallback responses
            fallbacks = [
                "I'm having some trouble thinking right now. Could you rephrase that?",
                "I want to make sure I understand you correctly. Could you elaborate?",
                "That's an interesting point. Let me think about how to respond.",
                "I appreciate your patience. Could you tell me more about what's on your mind?"
            ]
            return {"response": random.choice(fallbacks)}
    
    def _get_next_deadline(self) -> Optional[Dict]:
        """Get information about the next deadline"""
        next_event = self.temporal_awareness.get_time_until_next_event()
        if next_event:
            time_until, event = next_event
            return {
                "event": event.name,
                "in": f"{time_until.seconds // 60} minutes",
                "deadline": event.start_time.isoformat()
            }
        return None
    
    def _process_autonomous_goals(self):
        """Process and create any autonomous goals"""
        try:
            created_goals = self.goal_generator.process_autonomous_goals()
            for result in created_goals:
                if result["goal"]:
                    goal = result["goal"]
                    self.logger.info(f"Created autonomous goal: {goal.title}")
                    
                    # Add context about the new goal
                    self.consciousness.add_context(
                        content=f"Created autonomous goal: {goal.title}",
                        context_type='task',
                        importance=0.7,
                        related_entities=[f"goal:{goal.id}"],
                        metadata={
                            'goal_id': goal.id,
                            'priority': goal.priority,
                            'auto_generated': True
                        }
                    )
                    
                    # Log as a thought for self-reflection
                    self.consciousness.add_thought(
                        f"Created new autonomous goal: {goal.title}",
                        source="autonomous_goals",
                        confidence=0.8,
                        metadata={
                            'goal_id': goal.id,
                            'category': goal.category,
                            'priority': goal.priority
                        }
                    )
        except Exception as e:
            self.logger.error(f"Error processing autonomous goals: {e}", exc_info=True)
    
    def _save_states(self):
        """Save all component states"""
        try:
            self.goal_system.save_goals()
            self.theory_of_mind.save_state()
            self.temporal_awareness.save_state()
            
            # Save consciousness state with additional context
            if hasattr(self, 'consciousness'):
                # Add system status to consciousness before saving
                self.consciousness.add_context(
                    content=f"System status update at {datetime.utcnow().isoformat()}",
                    context_type='system',
                    importance=0.3,
                    metadata={
                        'interaction_count': self.interaction_count,
                        'last_interaction': self.last_interaction_time.isoformat() if hasattr(self, 'last_interaction_time') else None,
                        'active_goals': len([g for g in self.goal_system.goals.values() if g.status == GoalStatus.IN_PROGRESS])
                    }
                )
                self.consciousness.save_state()
                
        except Exception as e:
            self.logger.error(f"Error saving states: {e}", exc_info=True)
    
    def get_status_report(self) -> Dict[str, Any]:
        """Get a comprehensive status report of the agent"""
        return {
            "consciousness": self.consciousness.get_self_report(),
            "goals": {
                "active": len(self.goal_system.get_goals_by_status(GoalStatus.IN_PROGRESS)),
                "completed": len(self.goal_system.get_goals_by_status(GoalStatus.COMPLETED)),
                "pending": len(self.goal_system.get_goals_by_status(GoalStatus.PENDING))
            },
            "interactions": {
                "total": self.interaction_count,
                "last_interaction": self.last_interaction_time.isoformat(),
                "relationship_strength": self.theory_of_mind.relationship_strength
            },
            "temporal": {
                "next_event": self._get_next_deadline(),
                "upcoming_events": len(self.temporal_awareness.get_upcoming_events(hours=24))
            }
        }

# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize the agent
    agent = AgentCore(user_id="example_user")
    
    # Example interaction
    response = agent.process_input("Hello, how are you?")
    print("Agent:", response["response"])
    print("\nStatus:", json.dumps(agent.get_status_report(), indent=2))
