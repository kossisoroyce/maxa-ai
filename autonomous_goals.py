"""
Autonomous goal generation system for Maxa Anima.

This module handles the creation of goals by the AI system itself,
based on patterns, user interactions, and system needs.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import random
from dataclasses import dataclass, field
from enum import Enum
import json

from goal_system import Goal, GoalStatus, GoalPriority, SubGoal

logger = logging.getLogger(__name__)

class GoalGenerationTrigger(Enum):
    """Different triggers for goal generation"""
    TIME_BASED = "time_based"
    INTERACTION_PATTERN = "interaction_pattern"
    KNOWLEDGE_GAP = "knowledge_gap"
    USER_BEHAVIOR = "user_behavior"
    SYSTEM_NEED = "system_need"

@dataclass
class GoalSuggestion:
    """A suggested goal with metadata"""
    title: str
    description: str
    category: str
    priority: int = GoalPriority.MEDIUM.value
    confidence: float = 0.8  # 0.0 to 1.0
    trigger: GoalGenerationTrigger = GoalGenerationTrigger.SYSTEM_NEED
    metadata: dict = field(default_factory=dict)

class AutonomousGoalGenerator:
    """
    Generates goals autonomously based on system state and user interactions.
    
    Safety features:
    - Rate limiting
    - Confidence thresholds
    - User approval for certain goal types
    - Clear reasoning for each goal
    - Maximum number of concurrent goals
    """
    
    def __init__(self, agent):
        self.agent = agent
        self.last_goal_creation = datetime.utcnow()
        self.goals_created = 0
        self.max_goals_per_day = 5
        self.min_confidence = 0.7
        self.requires_approval_above_priority = GoalPriority.HIGH.value
        
    def can_create_goals(self) -> bool:
        """Check if we can create new goals"""
        # Rate limiting
        time_since_last = datetime.utcnow() - self.last_goal_creation
        if time_since_last < timedelta(hours=1):
            return False
            
        # Daily limit
        if self.goals_created >= self.max_goals_per_day:
            return False
            
        return True
    
    def generate_goal_suggestions(self) -> List[GoalSuggestion]:
        """Generate potential goals based on system state"""
        if not self.can_create_goals():
            return []
            
        suggestions = []
        
        # Time-based goals (e.g., weekly check-ins)
        suggestions.extend(self._generate_time_based_goals())
        
        # Interaction pattern goals
        suggestions.extend(self._generate_interaction_based_goals())
        
        # Knowledge gap goals
        suggestions.extend(self._generate_knowledge_goals())
        
        # System maintenance goals
        suggestions.extend(self._generate_system_goals())
        
        return suggestions
    
    def _generate_time_based_goals(self) -> List[GoalSuggestion]:
        """Generate goals based on time patterns"""
        suggestions = []
        now = datetime.utcnow()
        
        # Weekly check-in
        if now.weekday() == 0:  # Monday
            suggestions.append(GoalSuggestion(
                title="Weekly Check-in",
                description="Conduct weekly progress review and planning",
                category="productivity",
                priority=GoalPriority.MEDIUM.value,
                trigger=GoalGenerationTrigger.TIME_BASED,
                metadata={"frequency": "weekly"}
            ))
            
        return suggestions
    
    def _generate_interaction_based_goals(self) -> List[GoalSuggestion]:
        """Generate goals based on interaction patterns"""
        suggestions = []
        recent_interactions = self.agent.theory_of_mind.get_recent_interactions(limit=50)
        
        if not recent_interactions:
            return []
            
        # Look for patterns in recent interactions
        topics = {}
        question_patterns = [
            ("python", ["python", "programming", "code"]),
            ("data science", ["data science", "machine learning", "ml", "ai", "data analysis"]),
            ("web development", ["web", "flask", "django", "api", "backend"]),
            ("algorithms", ["algorithm", "data structure", "sorting", "searching"]),
            ("debugging", ["error", "bug", "fix", "debug", "not working"]),
        ]
        
        # Analyze content for topics and questions
        for interaction in recent_interactions:
            content = interaction.content.lower()
            
            # Check for question patterns
            if '?' in content or any(word in content for word in ["how to", "what is", "how do i", "can you explain"]):
                for topic, keywords in question_patterns:
                    if any(keyword in content for keyword in keywords):
                        topics[topic] = topics.get(topic, 0) + 1
        
        # Generate goal suggestions for frequent topics
        for topic, count in topics.items():
            if count >= 2:  # If mentioned at least twice in questions
                confidence = min(0.9, 0.5 + (count * 0.2))  # Scale confidence with count
                
                # Determine priority based on frequency
                if count >= 4:
                    priority = GoalPriority.HIGH.value
                    confidence = min(0.95, confidence + 0.1)
                else:
                    priority = GoalPriority.MEDIUM.value
                
                suggestions.append(GoalSuggestion(
                    title=f"Improve knowledge of {topic}",
                    description=f"Research and learn more about {topic} to better answer user questions",
                    category="learning",
                    priority=priority,
                    confidence=confidence,
                    trigger=GoalGenerationTrigger.INTERACTION_PATTERN,
                    metadata={
                        "topic": topic,
                        "mention_count": count,
                        "last_mentioned": datetime.utcnow().isoformat()
                    }
                ))
        
        # Look for repeated requests for help
        help_requests = [i for i in recent_interactions 
                        if any(phrase in i.content.lower() 
                             for phrase in ["help with", "can you help", "i need help", "how do i"])]
        
        if len(help_requests) >= 3:
            suggestions.append(GoalSuggestion(
                title="Create help resources",
                description="Develop a knowledge base or FAQ for common questions",
                category="productivity",
                priority=GoalPriority.MEDIUM.value,
                confidence=0.8,
                trigger=GoalGenerationTrigger.INTERACTION_PATTERN,
                metadata={
                    "help_requests_count": len(help_requests),
                    "last_request": help_requests[-1].timestamp.isoformat()
                }
            ))
        
        return suggestions
    
    def _generate_knowledge_goals(self) -> List[GoalSuggestion]:
        """Generate goals to fill knowledge gaps"""
        suggestions = []
        
        # Example: If user asks about a topic we don't know much about
        # This would be more sophisticated with actual knowledge graph integration
        
        return suggestions
    
    def _generate_system_goals(self) -> List[GoalSuggestion]:
        """Generate system maintenance and improvement goals"""
        suggestions = []
        
        # Check when we last did system maintenance
        maintenance_goals = [g for g in self.agent.goal_system.goals.values() 
                           if g.category == "maintenance" and g.status == GoalStatus.COMPLETED]
        last_maintenance = max([g.updated_at for g in maintenance_goals] + [datetime.min])
        
        if (datetime.utcnow() - last_maintenance) > timedelta(days=7):
            suggestions.append(GoalSuggestion(
                title="System Maintenance",
                description="Perform routine system checks and updates",
                category="maintenance",
                priority=GoalPriority.MEDIUM.value,
                trigger=GoalGenerationTrigger.SYSTEM_NEED,
                confidence=0.9
            ))
            
        return suggestions
    
    def create_goal_from_suggestion(self, suggestion: GoalSuggestion) -> Tuple[Optional[Goal], str]:
        """Create a goal from a suggestion with safety checks"""
        if suggestion.confidence < self.min_confidence:
            return None, "Confidence too low to create goal"
            
        # Create the goal
        goal = Goal(
            id=self.agent.goal_system.generate_goal_id(),
            title=suggestion.title,
            description=suggestion.description,
            category=suggestion.category,
            status=GoalStatus.PENDING,
            priority=suggestion.priority,
            metadata={
                "auto_generated": True,
                "trigger": suggestion.trigger.value,
                "confidence": suggestion.confidence,
                **suggestion.metadata
            }
        )
        
        # Check for conflicts
        conflicts = self.agent.goal_system.check_goal_conflicts(goal)
        if conflicts:
            conflict_titles = ", ".join([g.title for g in conflicts])
            return None, f"Conflicting goals found: {conflict_titles}"
        
        # Add to system
        self.agent.goal_system.add_goal(goal)
        self.last_goal_creation = datetime.utcnow()
        self.goals_created += 1
        
        # Log the creation
        self.agent.consciousness.add_thought(
            f"Created autonomous goal: {goal.title}",
            source="autonomous_goals",
            confidence=suggestion.confidence,
            metadata={
                "goal_id": goal.id,
                "trigger": suggestion.trigger.value,
                "priority": goal.priority
            }
        )
        
        return goal, f"Created goal: {goal.title}"
    
    def process_autonomous_goals(self) -> List[Dict]:
        """Main entry point to process and create autonomous goals"""
        if not self.can_create_goals():
            return []
            
        suggestions = self.generate_goal_suggestions()
        created_goals = []
        
        for suggestion in suggestions:
            if suggestion.priority > self.requires_approval_above_priority:
                # For high-priority goals, just log them for now
                self.agent.consciousness.add_thought(
                    f"High-priority goal suggestion needs approval: {suggestion.title}",
                    source="autonomous_goals",
                    confidence=suggestion.confidence,
                    metadata={
                        "suggestion": suggestion.__dict__,
                        "requires_approval": True
                    }
                )
                continue
                
            goal, message = self.create_goal_from_suggestion(suggestion)
            if goal:
                created_goals.append({
                    "goal": goal,
                    "message": message,
                    "requires_approval": False
                })
        
        return created_goals
