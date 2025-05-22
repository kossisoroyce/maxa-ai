"""
Configuration file for pytest.

This file contains fixtures and configuration that are shared across test modules.
"""
import os
import sys
import pytest
import tempfile
import shutil
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

@pytest.fixture(scope="session")
def temp_dir():
    """Create and return a temporary directory that will be cleaned up after tests."""
    temp_dir = tempfile.mkdtemp(prefix="maxa_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def test_data_dir():
    """Return a temporary directory for test data that will be cleaned up after each test."""
    temp_dir = tempfile.mkdtemp(prefix="maxa_test_data_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)
