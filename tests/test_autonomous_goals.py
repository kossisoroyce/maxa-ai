"""
Test script for autonomous goal generation.
"""

import logging
import os
import sys
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_autonomous_goals.log')
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from anima_core import AgentCore
from goal_system import GoalStatus, Goal, GoalPriority
from theory_of_mind import TheoryOfMind, Interaction, InteractionType
from consciousness import Consciousness

TEST_DATA_DIR = "test_autonomous_goals_data"

class MockTheoryOfMind(TheoryOfMind):
    """Mock TheoryOfMind for testing"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.interactions: List[Interaction] = []
        
    def add_interaction(self, interaction: Interaction):
        self.interactions.append(interaction)
        return interaction
        
    def get_recent_interactions(self, limit: int = 10) -> List[Interaction]:
        return self.interactions[-limit:] if self.interactions else []
        
    def save_data(self):
        pass
    
    def load_data(self):
        pass


def setup_test_environment():
    """Set up test environment with clean data directory."""
    test_dir = Path(TEST_DATA_DIR)
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(exist_ok=True)
    return test_dir


def cleanup_test_environment():
    """Clean up test environment."""
    test_dir = Path(TEST_DATA_DIR)
    if test_dir.exists():
        shutil.rmtree(test_dir)

def test_autonomous_goal_creation():
    """Test that the system can create goals autonomously."""
    test_dir = setup_test_environment()
    
    try:
        # Initialize agent with test data directory
        agent = AgentCore(user_id="test_user", data_dir=str(test_dir))
        
        # Replace the theory of mind with our mock
        agent.theory_of_mind = MockTheoryOfMind(agent)
        
        # Clear existing goals for test
        agent.goal_system.goals = {}
        
        # Test interactions that should trigger goal creation
        test_interactions = [
            # Initial interaction about Python
            ("How can I learn Python for data analysis?", "user"),
            ("I'd be happy to help you learn Python for data analysis! Let's start with the basics of Python.", "assistant"),
            
            # More specific Python questions
            ("What's the difference between lists and tuples in Python?", "user"),
            ("Great question! Lists are mutable while tuples are immutable. Here's an example...", "assistant"),
            
            # Data science questions
            ("How do I use pandas to clean data?", "user"),
            ("Pandas is great for data cleaning! You can use methods like dropna(), fillna(), and apply().", "assistant"),
            
            # More advanced Python
            ("Can you explain list comprehensions?", "user"),
            ("List comprehensions are a concise way to create lists. Here's how they work...", "assistant"),
            
            # Debugging help
            ("I'm getting an error when I try to import pandas. It says 'ModuleNotFoundError: No module named 'pandas'", "user"),
            ("That means pandas isn't installed. You can install it with 'pip install pandas'.", "assistant"),
        ]
        
        # Add test interactions
        for content, role in test_interactions:
            if role == "user":
                agent.process_input(content)
        
        # Process autonomous goals
        agent._process_autonomous_goals()
        
        # Check if goals were created
        goals = agent.goal_system.get_goals_by_status(GoalStatus.IN_PROGRESS)
        assert len(goals) > 0, "No autonomous goals were created"
        
        # Check that the goals were added to consciousness
        for goal in goals:
            assert goal.id in [g.id for g in agent.goal_system.goals.values()], \
                f"Goal {goal.title} not found in goal system"
            
            # Check that context was added for the goal
            goal_contexts = [
                ctx for ctx in agent.consciousness.contexts.values()
                if f"goal:{goal.id}" in ctx.related_entities
            ]
            assert len(goal_contexts) > 0, f"No context found for goal {goal.id}"
            
            logger.info(f"Created goal: {goal.title} (ID: {goal.id})")
        
        # Test persistence
        agent._save_states()
        
        # Create a new agent that should load the state
        agent2 = AgentCore(user_id="test_user", data_dir=str(test_dir))
        
        # Check that goals were loaded
        loaded_goals = agent2.goal_system.get_goals_by_status(GoalStatus.IN_PROGRESS)
        assert len(loaded_goals) == len(goals), "Number of goals changed after reload"
        
        logger.info("✅ All autonomous goal tests passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return False
    finally:
        cleanup_test_environment()

if __name__ == "__main__":
    success = test_autonomous_goal_creation()
    if success:
        print("\n✅ Test passed: Autonomous goals were created successfully!")
        sys.exit(0)
    else:
        print("\n❌ Test failed: Check the logs for details.")
        sys.exit(1)
