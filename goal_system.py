from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import json
import random
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class GoalStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ON_HOLD = "on_hold"
class GoalPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class SubGoal:
    description: str
    deadline: Optional[datetime] = None
    status: GoalStatus = GoalStatus.PENDING
    priority: GoalPriority = GoalPriority.MEDIUM
    progress: float = 0.0  # 0.0 to 1.0
    dependencies: List[str] = field(default_factory=list)  # IDs of dependent goals
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "status": self.status.value if hasattr(self.status, 'value') else self.status,
            "priority": self.priority.value if hasattr(self.priority, 'value') else self.priority,
            "progress": self.progress,
            "dependencies": self.dependencies,
            "created_at": self.created_at.isoformat() if hasattr(self.created_at, 'isoformat') else self.created_at,
            "updated_at": self.updated_at.isoformat() if hasattr(self.updated_at, 'isoformat') else self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SubGoal':
        # Handle status
        status = data["status"]
        if isinstance(status, str):
            try:
                status = GoalStatus(status)
            except ValueError:
                status = GoalStatus.PENDING
        elif not isinstance(status, GoalStatus):
            status = GoalStatus.PENDING
            
        # Handle priority
        priority = data["priority"]
        if isinstance(priority, int):
            try:
                priority = GoalPriority(priority)
            except ValueError:
                priority = GoalPriority.MEDIUM
        elif isinstance(priority, str):
            try:
                priority = GoalPriority[priority.upper()]
            except (KeyError, AttributeError):
                priority = GoalPriority.MEDIUM
        elif not isinstance(priority, GoalPriority):
            priority = GoalPriority.MEDIUM
            
        # Handle datetime fields
        def parse_datetime(dt_str):
            if isinstance(dt_str, str):
                try:
                    return datetime.fromisoformat(dt_str)
                except (ValueError, TypeError):
                    return datetime.utcnow()
            return dt_str if dt_str is not None else datetime.utcnow()
            
        deadline = parse_datetime(data.get("deadline"))
        created_at = parse_datetime(data.get("created_at"))
        updated_at = parse_datetime(data.get("updated_at"))
        
        return cls(
            description=data.get("description", ""),
            deadline=deadline,
            status=status,
            priority=priority,
            progress=float(data.get("progress", 0.0)),
            dependencies=list(data.get("dependencies", [])),
            created_at=created_at,
            updated_at=updated_at
        )

@dataclass
class Goal:
    id: str
    title: str
    description: str
    category: str
    status: GoalStatus = GoalStatus.PENDING
    priority: GoalPriority = GoalPriority.MEDIUM
    progress: float = 0.0
    deadline: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    subgoals: List[SubGoal] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)  # For storing additional data like auto-generation info
    
    def update_progress(self):
        """Update overall progress based on subgoals"""
        if not self.subgoals:
            return
        total = len(self.subgoals)
        completed = sum(1 for sg in self.subgoals if sg.status == GoalStatus.COMPLETED)
        self.progress = completed / total
        self.updated_at = datetime.utcnow()
    
    def add_subgoal(self, subgoal: SubGoal):
        self.subgoals.append(subgoal)
        self.update_progress()
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "category": self.category,
            "status": self.status.value if hasattr(self.status, 'value') else self.status,
            "priority": self.priority.value if hasattr(self.priority, 'value') else self.priority,
            "progress": self.progress,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "subgoals": [sg.to_dict() for sg in self.subgoals] if hasattr(self, 'subgoals') else [],
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Goal':
        # Handle status
        status = data.get("status", "pending")
        if isinstance(status, str):
            try:
                status = GoalStatus(status)
            except ValueError:
                status = GoalStatus.PENDING
        elif not isinstance(status, GoalStatus):
            status = GoalStatus.PENDING
            
        # Handle priority
        priority = data.get("priority", 2)  # Default to MEDIUM
        if isinstance(priority, int):
            try:
                priority = GoalPriority(priority)
            except ValueError:
                priority = GoalPriority.MEDIUM
        elif isinstance(priority, str):
            try:
                priority = GoalPriority[priority.upper()]
            except (KeyError, AttributeError):
                priority = GoalPriority.MEDIUM
        elif not isinstance(priority, GoalPriority):
            priority = GoalPriority.MEDIUM
            
        # Handle datetime fields
        def parse_datetime(dt_str):
            if isinstance(dt_str, str):
                try:
                    return datetime.fromisoformat(dt_str)
                except (ValueError, TypeError):
                    return datetime.utcnow()
            return dt_str if dt_str is not None else datetime.utcnow()
            
        deadline = parse_datetime(data.get("deadline"))
        created_at = parse_datetime(data.get("created_at"))
        updated_at = parse_datetime(data.get("updated_at"))
        
        # Create the goal instance
        goal = cls(
            id=str(data.get("id", "")),
            title=str(data.get("title", "")),
            description=str(data.get("description", "")),
            category=str(data.get("category", "general")),
            status=status,
            priority=priority,
            progress=float(data.get("progress", 0.0)),
            deadline=deadline,
            created_at=created_at,
            updated_at=updated_at
        )
        
        # Add subgoals if any
        goal.subgoals = [SubGoal.from_dict(sg) for sg in data.get("subgoals", [])]
        
        # Add metadata if present
        if "metadata" in data and isinstance(data["metadata"], dict):
            goal.metadata = data["metadata"]
            
        return goal

class GoalSystem:
    def __init__(self, storage_path: str = "goals.json"):
        self.storage_path = storage_path
        self.goals: Dict[str, Goal] = {}
        self.load_goals()
    
    def add_goal(self, goal: Goal):
        """Add or update a goal"""
        self.goals[goal.id] = goal
        self.save_goals()
    
    def remove_goal(self, goal_id: str) -> bool:
        """Remove a goal by ID"""
        if goal_id in self.goals:
            del self.goals[goal_id]
            self.save_goals()
            return True
        return False
    
    def get_goal(self, goal_id: str) -> Optional[Goal]:
        """Get a goal by ID"""
        return self.goals.get(goal_id)
    
    def get_goals_by_status(self, status: GoalStatus) -> List[Goal]:
        """Get all goals with a specific status"""
        return [g for g in self.goals.values() if g.status == status]
    
    def get_goals_by_priority(self, priority: GoalPriority) -> List[Goal]:
        """Get all goals with a specific priority or higher"""
        return [g for g in self.goals.values() if g.priority.value >= priority.value]
    
    def update_goal_progress(self, goal_id: str, progress: float) -> bool:
        """Update a goal's progress"""
        if goal_id in self.goals:
            self.goals[goal_id].progress = max(0.0, min(1.0, progress))
            self.goals[goal_id].updated_at = datetime.utcnow()
            self.save_goals()
            return True
        return False
    
    def complete_goal(self, goal_id: str) -> bool:
        """Mark a goal as completed"""
        if goal_id in self.goals:
            self.goals[goal_id].status = GoalStatus.COMPLETED
            self.goals[goal_id].progress = 1.0
            self.goals[goal_id].updated_at = datetime.utcnow()
            self.save_goals()
            return True
        return False
    
    def save_goals(self):
        """Save goals to storage"""
        try:
            # Prepare the data to be saved
            goals_data = []
            for goal in self.goals.values():
                try:
                    goal_dict = goal.to_dict()
                    # Ensure all values are JSON serializable
                    serializable_goal = {}
                    for key, value in goal_dict.items():
                        if hasattr(value, 'isoformat'):
                            serializable_goal[key] = value.isoformat()
                        elif hasattr(value, 'value'):  # Handle enum values
                            serializable_goal[key] = value.value if hasattr(value, 'value') else str(value)
                        elif isinstance(value, (list, dict)):
                            # Recursively process nested structures
                            serializable_goal[key] = self._make_serializable(value)
                        else:
                            serializable_goal[key] = value
                    goals_data.append(serializable_goal)
                except Exception as e:
                    logger.error(f"Error preparing goal for saving: {e}")
                    continue
            
            # Save to file
            with open(self.storage_path, 'w') as f:
                json.dump({"goals": goals_data}, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving goals: {e}")
    
    def _make_serializable(self, obj):
        """Recursively make an object JSON serializable"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, 'isoformat'):  # Handle datetime
            return obj.isoformat()
        elif hasattr(obj, 'value'):  # Handle enum values
            return obj.value if hasattr(obj, 'value') else str(obj)
        return obj
    
    def load_goals(self):
        """Load goals from storage"""
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.goals = {}
                for g in data.get("goals", []):
                    try:
                        goal = Goal.from_dict(g)
                        self.goals[goal.id] = goal
                    except Exception as e:
                        logger.error(f"Error loading goal {g.get('id', 'unknown')}: {e}")
                        continue
        except FileNotFoundError:
            logger.info("No existing goals file found, starting with empty goal system")
            self.goals = {}
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in goals file: {self.storage_path}")
            self.goals = {}
        except Exception as e:
            logger.error(f"Error loading goals: {e}")
            self.goals = {}
    
    def generate_goal_id(self) -> str:
        """Generate a unique goal ID"""
        return f"goal_{len(self.goals) + 1}_{random.randint(1000, 9999)}"
    
    def check_goal_conflicts(self, goal: Goal) -> List[Goal]:
        """Check for potential conflicts with existing goals"""
        conflicts = []
        for existing_goal in self.goals.values():
            # Check for similar goals
            if (goal.category == existing_goal.category and 
                goal.title.lower() == existing_goal.title.lower()):
                conflicts.append(existing_goal)
            
            # Check for time conflicts if deadlines are set
            if goal.deadline and existing_goal.deadline:
                time_diff = abs((goal.deadline - existing_goal.deadline).total_seconds())
                if time_diff < 3600 * 24:  # Within 24 hours
                    conflicts.append(existing_goal)
        
        return conflicts
