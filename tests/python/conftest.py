"""
Pytest Configuration and Fixtures
Global test configuration and shared fixtures for Python tests
"""

import asyncio
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List, Generator
import json
import os
from datetime import datetime, timedelta

# Import system modules for testing
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from src.core.interfaces.detector_interfaces import IDetector, IDetectionResult
from src.core.interfaces.data_interfaces import DataSample, DataBatch
from src.core.messaging.protocol import Message, MessageType, RequestMessage, ResponseMessage


# Test Configuration
@pytest.fixture(scope="session")
def test_config():
    """Global test configuration"""
    return {
        "test_data_dir": Path(__file__).parent / "fixtures",
        "temp_dir": None,  # Will be set in temp_dir fixture
        "api_base_url": "http://localhost:8000",
        "test_timeout": 30,
        "mock_external_apis": True,
        "log_level": "DEBUG"
    }


# Temporary Directory Management
@pytest.fixture(scope="function")
def temp_dir(test_config):
    """Create temporary directory for each test"""
    temp_path = Path(tempfile.mkdtemp(prefix="ai_detector_test_"))
    test_config["temp_dir"] = temp_path
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture(scope="function")
def test_files_dir(temp_dir):
    """Create test files directory structure"""
    files_dir = temp_dir / "test_files"
    files_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    (files_dir / "data").mkdir()
    (files_dir / "models").mkdir()
    (files_dir / "cache").mkdir()
    (files_dir / "logs").mkdir()
    
    return files_dir


# Event Loop Management
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Test Data Fixtures
@pytest.fixture
def sample_texts():
    """Sample text data for testing"""
    return {
        "human_short": "Just grabbed coffee ☕ having a great morning!",
        "human_long": "I've been thinking about this for a while now, and I really think we should consider changing our approach. The current method isn't working as well as we hoped, and I've got some ideas that might help. What do you think?",
        "human_casual": "lol that's so funny 😂 can't believe it happened",
        "human_emotional": "I'm so excited about this opportunity! It's been a dream of mine for years and I finally get to do it. Thank you everyone for the support ❤️",
        
        "ai_obvious": "As an AI language model, I must emphasize that this is a complex topic that requires careful consideration of multiple factors and stakeholders.",
        "ai_formal": "It is important to note that while this approach has demonstrated considerable merit, one must carefully consider the potential implications and ramifications that may arise.",
        "ai_hedged": "This seems to suggest that perhaps there might be some potential benefits, though it's worth noting that further investigation would likely be beneficial.",
        "ai_structured": "First, we should examine the primary factors. Second, we need to consider the secondary implications. Finally, we must evaluate the long-term consequences.",
        
        "edge_case_empty": "",
        "edge_case_short": "Hi",
        "edge_case_long": "A" * 10000,  # Very long text
        "edge_case_special": "🤖🤖🤖 @@@ ### %%% ((()))",
        "edge_case_mixed": "This is NORMAL text with CAPS and numbers 123 and symbols !@# and emojis 😀🎉"
    }


@pytest.fixture
def sample_tweets():
    """Sample tweet data for testing"""
    return [
        {
            "id": "1",
            "text": "Just had the best coffee ever! Can't believe how good it was ☕",
            "author": "coffee_lover",
            "timestamp": "2024-01-01T10:00:00Z",
            "label": "human",
            "metadata": {"source": "twitter", "verified": False}
        },
        {
            "id": "2",
            "text": "It's important to note that while artificial intelligence presents numerous opportunities, one must carefully consider the ethical implications.",
            "author": "tech_analyst",
            "timestamp": "2024-01-01T11:00:00Z",
            "label": "ai",
            "metadata": {"source": "twitter", "verified": True}
        },
        {
            "id": "3",
            "text": "Working on some new features for the app. Excited to ship them soon! 🚀",
            "author": "developer",
            "timestamp": "2024-01-01T12:00:00Z",
            "label": "human",
            "metadata": {"source": "twitter", "verified": True}
        }
    ]


@pytest.fixture
def sample_data_samples(sample_tweets):
    """Convert sample tweets to DataSample objects"""
    samples = []
    for tweet in sample_tweets:
        sample = DataSample(
            id=tweet["id"],
            content=tweet["text"],
            label=tweet["label"],
            metadata=tweet["metadata"],
            timestamp=datetime.fromisoformat(tweet["timestamp"].replace('Z', '+00:00'))
        )
        samples.append(sample)
    return samples


@pytest.fixture
def sample_data_batch(sample_data_samples):
    """Create a DataBatch from sample data"""
    return DataBatch(
        samples=sample_data_samples,
        batch_id="test_batch_1",
        created_at=datetime.now(),
        metadata={"test": True, "source": "fixtures"}
    )


# Mock Objects
@pytest.fixture
def mock_detector():
    """Mock detector for testing"""
    detector = Mock(spec=IDetector)
    
    async def mock_detect(text: str, config=None):
        # Simple mock logic based on text content
        if "important to note" in text.lower() or "one must" in text.lower():
            ai_prob = 0.85
            prediction = "ai"
        else:
            ai_prob = 0.15
            prediction = "human"
        
        result = Mock(spec=IDetectionResult)
        result.get_score.return_value = Mock(
            ai_probability=ai_prob,
            prediction=prediction,
            confidence=0.8
        )
        result.get_evidence.return_value = []
        result.get_metadata.return_value = {"model": "mock", "version": "1.0.0"}
        result.get_processing_time.return_value = 0.1
        result.to_dict.return_value = {
            "prediction": prediction,
            "ai_probability": ai_prob,
            "confidence": 0.8,
            "processing_time": 0.1
        }
        
        return result
    
    detector.detect = AsyncMock(side_effect=mock_detect)
    detector.detect_batch = AsyncMock()
    detector.get_supported_languages.return_value = ["en"]
    detector.get_detection_method.return_value = "mock"
    detector.initialize = AsyncMock(return_value=True)
    detector.is_initialized.return_value = True
    
    return detector


@pytest.fixture
def mock_api_client():
    """Mock API client for testing"""
    client = Mock()
    
    async def mock_request(method, endpoint, **kwargs):
        # Mock different responses based on endpoint
        if endpoint.endswith("/detect"):
            return Mock(
                status_code=200,
                body={
                    "success": True,
                    "result": {
                        "prediction": "human",
                        "ai_probability": 0.15,
                        "confidence": 0.85
                    }
                }
            )
        elif endpoint.endswith("/health"):
            return Mock(
                status_code=200,
                body={"status": "healthy", "version": "1.0.0"}
            )
        else:
            return Mock(
                status_code=404,
                body={"error": "Not found"}
            )
    
    client.request = AsyncMock(side_effect=mock_request)
    client.get = AsyncMock(side_effect=lambda ep, **kw: mock_request("GET", ep, **kw))
    client.post = AsyncMock(side_effect=lambda ep, **kw: mock_request("POST", ep, **kw))
    client.initialize = AsyncMock(return_value=True)
    client.close = AsyncMock()
    
    return client


@pytest.fixture
def mock_message_bus():
    """Mock message bus for testing"""
    bus = Mock()
    
    messages_sent = []
    
    async def mock_send(message):
        messages_sent.append(message)
        return True
    
    async def mock_request(message_type, data, timeout=5.0):
        # Mock response based on message type
        if message_type == "detect_text":
            return {
                "success": True,
                "result": {
                    "prediction": "human",
                    "confidence": 0.85
                }
            }
        return {"success": True, "data": "mock_response"}
    
    bus.send = AsyncMock(side_effect=mock_send)
    bus.request = AsyncMock(side_effect=mock_request)
    bus.subscribe = Mock(return_value="sub_123")
    bus.unsubscribe = Mock(return_value=True)
    bus.get_messages_sent = lambda: messages_sent
    bus.initialize = AsyncMock(return_value=True)
    
    return bus


# Database and Storage Mocks
@pytest.fixture
def mock_database():
    """Mock database for testing"""
    db = Mock()
    
    # In-memory storage for testing
    tables = {
        "detections": [],
        "samples": [],
        "models": [],
        "settings": {}
    }
    
    async def mock_insert(table, data):
        if table in tables:
            item = {**data, "id": len(tables[table]) + 1}
            tables[table].append(item)
            return item["id"]
        return None
    
    async def mock_select(table, conditions=None):
        if table in tables:
            items = tables[table]
            if conditions:
                # Simple filtering for tests
                filtered = []
                for item in items:
                    match = True
                    for key, value in conditions.items():
                        if item.get(key) != value:
                            match = False
                            break
                    if match:
                        filtered.append(item)
                return filtered
            return items
        return []
    
    async def mock_update(table, item_id, data):
        if table in tables:
            for i, item in enumerate(tables[table]):
                if item.get("id") == item_id:
                    tables[table][i] = {**item, **data}
                    return True
        return False
    
    async def mock_delete(table, item_id):
        if table in tables:
            tables[table] = [item for item in tables[table] if item.get("id") != item_id]
            return True
        return False
    
    db.insert = AsyncMock(side_effect=mock_insert)
    db.select = AsyncMock(side_effect=mock_select)
    db.update = AsyncMock(side_effect=mock_update)
    db.delete = AsyncMock(side_effect=mock_delete)
    db.connect = AsyncMock(return_value=True)
    db.disconnect = AsyncMock()
    db.get_tables = lambda: tables
    
    return db


# Environment and Configuration
@pytest.fixture
def test_env_vars():
    """Set test environment variables"""
    original_env = dict(os.environ)
    
    test_vars = {
        "TESTING": "true",
        "DEBUG": "true",
        "LOG_LEVEL": "DEBUG",
        "API_BASE_URL": "http://localhost:8000",
        "DATABASE_URL": "sqlite:///:memory:",
        "CACHE_ENABLED": "false",
        "RATE_LIMIT_ENABLED": "false"
    }
    
    os.environ.update(test_vars)
    
    yield test_vars
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


# Performance Testing
@pytest.fixture
def performance_monitor():
    """Monitor performance during tests"""
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.measurements = []
        
        def start(self):
            self.start_time = datetime.now()
        
        def stop(self):
            self.end_time = datetime.now()
            if self.start_time:
                duration = (self.end_time - self.start_time).total_seconds()
                self.measurements.append(duration)
                return duration
            return 0
        
        def get_average(self):
            if self.measurements:
                return sum(self.measurements) / len(self.measurements)
            return 0
        
        def get_max(self):
            return max(self.measurements) if self.measurements else 0
        
        def reset(self):
            self.measurements = []
    
    return PerformanceMonitor()


# Markers and Parameterization
@pytest.fixture
def detection_test_cases():
    """Test cases for detection testing"""
    return [
        {
            "name": "human_casual",
            "text": "lol that's so funny 😂",
            "expected": "human",
            "min_confidence": 0.7
        },
        {
            "name": "ai_formal",
            "text": "It's important to note that this matter requires careful consideration.",
            "expected": "ai",
            "min_confidence": 0.7
        },
        {
            "name": "edge_case_short",
            "text": "Hi",
            "expected": "human",
            "min_confidence": 0.5
        }
    ]


# Cleanup and Utilities
@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Automatic cleanup after each test"""
    yield
    
    # Clear any global state, caches, etc.
    # This runs after each test automatically


def pytest_configure(config):
    """Pytest configuration hook"""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "external: marks tests that require external services"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    # Automatically mark slow tests
    for item in items:
        if "slow" in item.nodeid or "performance" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        if "external" in item.nodeid or "api" in item.nodeid:
            item.add_marker(pytest.mark.external)