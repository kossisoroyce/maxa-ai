import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from anima_core import AgentCore
from consciousness import Consciousness, Context, ContentSummary, ConsciousnessState
from theory_of_mind import Interaction, InteractionType

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_consciousness.log')
    ]
)
logger = logging.getLogger(__name__)

import pytest

@pytest.fixture
def consciousness():
    """Fixture that provides a clean consciousness instance for testing."""
    # Use a unique test file name to avoid conflicts
    test_file = "test_consciousness.json"
    if os.path.exists(test_file):
        os.remove(test_file)
    return Consciousness(test_file)

def test_context_management(consciousness):
    """Test basic context management functionality."""
    logger.info("\n=== Testing Context Management ===")
    
    # Add some contexts
    context1_id = consciousness.add_context(
        content="User is learning about Python programming",
        context_type='learning',
        importance=0.8,
        related_entities=["python", "programming"],
        metadata={"source": "user_interaction", "topic": "programming"}
    )
    
    context2_id = consciousness.add_context(
        content="User is interested in machine learning",
        context_type='learning',
        importance=0.9,
        related_entities=["machine_learning", "ai"],
        metadata={"source": "user_preferences", "topic": "ai"}
    )
    
    # Test retrieving contexts
    context1 = consciousness.get_context(context1_id)
    context2 = consciousness.get_context(context2_id)
    
    assert context1 is not None, "Failed to retrieve context1"
    assert context2 is not None, "Failed to retrieve context2"
    assert "python" in context1.related_entities, "Related entities not stored correctly"
    
    # Test recent contexts
    recent = consciousness.get_recent_contexts(1)
    assert len(recent) == 1, "Should return 1 recent context"
    assert recent[0].id == context2_id, "Most recent context should be context2"
    
    logger.info("✅ Context management tests passed")

def test_content_summarization(consciousness):
    """Test content summarization functionality."""
    logger.info("\n=== Testing Content Summarization ===")
    
    # Create a context for summarization
    context_id = consciousness.add_context(
        content="User is researching neural networks",
        context_type='learning',
        importance=0.8
    )
    
    # Add a summary
    content = """
    Neural networks are computing systems inspired by biological neural networks. 
    They consist of layers of interconnected nodes that process information. 
    Deep learning uses multiple layers to learn representations of data.
    """
    
    key_points = [
        "Neural networks are inspired by biological systems",
        "They consist of interconnected nodes in layers",
        "Deep learning uses multiple layers"
    ]
    
    summary_id = consciousness.add_summary(
        content=content,
        key_points=key_points,
        source="research_paper",
        context_ids=[context_id]
    )
    
    # Test retrieving summary
    summaries = consciousness.get_summaries_for_context(context_id)
    assert len(summaries) == 1, "Should find one summary for the context"
    assert summaries[0].id == summary_id, "Summary ID mismatch"
    assert len(summaries[0].key_points) == 3, "Should have 3 key points"
    
    # Test recent summaries
    recent = consciousness.get_recent_summaries(1)
    assert len(recent) == 1, "Should return 1 recent summary"
    
    logger.info("✅ Content summarization tests passed")

def test_consciousness_integration():
    """Test integration with AgentCore."""
    logger.info("\n=== Testing AgentCore Integration ===")
    
    # Initialize agent
    agent = AgentCore(data_dir="test_agent_data")
    
    # Test processing input with context
    response1 = agent.process_input("What is Python?")
    assert "response" in response1, "Response should contain 'response' key"
    
    # Check that context was added
    assert len(agent.consciousness.contexts) > 0, "No contexts were created"
    
    # Test generating a summary
    response2 = agent.process_input("Can you explain how neural networks work?")
    
    # Check that summary was created
    recent_summaries = agent.consciousness.get_recent_summaries(1)
    assert len(recent_summaries) > 0, "No summaries were created"
    

def test_consciousness_persistence(tmp_path):
    """Test that consciousness state persists between sessions."""
    logger.info("\n=== Testing Consciousness Persistence ===")
    
    # Create a test file in the temporary directory
    test_file = tmp_path / "test_persistence.json"
    
    try:
        # Create first instance and add context
        consciousness1 = Consciousness(str(test_file))
        
        # Add multiple contexts to ensure proper serialization
        context_ids = []
        for i in range(3):
            context_id = consciousness1.add_context(
                content=f"Test context {i} for persistence",
                context_type='test',
                importance=0.7 + (i * 0.1)
            )
            context_ids.append(context_id)
        
        # Add a summary
        summary_id = consciousness1.add_summary(
            content="This is a test summary for persistence testing",
            key_points=["Point 1", "Point 2", "Point 3"],
            source="test",
            context_ids=context_ids
        )
        
        # Save state explicitly
        consciousness1.save_state()
        
        # Verify file was created and has content
        assert test_file.exists(), "State file was not created"
        with open(test_file, 'r') as f:
            state = json.load(f)
            assert 'contexts' in state, "State file missing 'contexts' key"
            assert 'summaries' in state, "State file missing 'summaries' key"
            assert len(state['contexts']) == 3, f"Expected 3 contexts in file, got {len(state['contexts'])}"
            assert len(state['summaries']) == 1, f"Expected 1 summary in file, got {len(state['summaries'])}"
        
        # Create new instance which should load the state
        consciousness2 = Consciousness(str(test_file))
        
        # Check that state was loaded
        assert len(consciousness2.contexts) == 3, f"Expected 3 contexts, got {len(consciousness2.contexts)}"
        for cid in context_ids:
            assert cid in consciousness2.contexts, f"Context {cid} was not persisted"
        
        # Verify summary was loaded
        summaries = consciousness2.get_recent_summaries(1)
        assert len(summaries) == 1, "Summary was not loaded"
        assert summaries[0].id == summary_id, "Summary ID mismatch"
        
        logger.info("✅ Consciousness persistence tests passed")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        raise

def main():
    """Run all consciousness tests."""
    try:
        # Run tests
        test_context_management()
        test_content_summarization()
        test_consciousness_integration()
        test_consciousness_persistence()
        
        logger.info("\n✅ All consciousness tests passed successfully!")
        return True
    except AssertionError as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
