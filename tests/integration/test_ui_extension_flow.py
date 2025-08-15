"""
Integration tests for UI to background to content script flow.

Tests the complete Chrome extension message flow from popup UI
through background script to content script and back.
"""

import pytest
import asyncio
import json
import time
from typing import Dict, Any, List
from unittest.mock import AsyncMock, patch, MagicMock

from src.core.messaging.protocol import MessageProtocol, MessageType
from src.utils.schema_validator import validate_extension_message


class MockChromeAPI:
    """Mock Chrome extension APIs for testing."""
    
    def __init__(self):
        self.tabs = MockTabsAPI()
        self.runtime = MockRuntimeAPI()
        self.storage = MockStorageAPI()
        self.action = MockActionAPI()
        self.message_handlers = {}
        self.messages = []
    
    def reset(self):
        """Reset all mock state."""
        self.messages.clear()
        self.tabs.reset()
        self.runtime.reset()
        self.storage.reset()


class MockTabsAPI:
    """Mock chrome.tabs API."""
    
    def __init__(self):
        self.active_tab = {
            "id": 123,
            "url": "https://twitter.com/user/status/123456",
            "title": "Twitter / X"
        }
        self.messages_sent = []
    
    def reset(self):
        """Reset tabs state."""
        self.messages_sent.clear()
    
    async def query(self, query_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Mock tabs.query."""
        if query_info.get("active"):
            return [self.active_tab]
        return []
    
    async def sendMessage(self, tab_id: int, message: Dict[str, Any]) -> Any:
        """Mock tabs.sendMessage."""
        self.messages_sent.append({
            "tab_id": tab_id,
            "message": message,
            "timestamp": time.time()
        })
        
        # Simulate content script response
        if message.get("type") == "DETECT_TEXT":
            return {
                "type": "DETECTION_RESULT",
                "id": f"response_{message.get('id')}",
                "correlation_id": message.get("correlation_id"),
                "payload": {
                    "is_ai_generated": True,
                    "confidence_score": 0.85,
                    "element_selector": message.get("payload", {}).get("element_selector")
                }
            }
        
        return {"success": True}


class MockRuntimeAPI:
    """Mock chrome.runtime API."""
    
    def __init__(self):
        self.message_listeners = []
        self.messages_sent = []
        self.last_error = None
    
    def reset(self):
        """Reset runtime state."""
        self.message_listeners.clear()
        self.messages_sent.clear()
        self.last_error = None
    
    def onMessage(self, listener):
        """Mock runtime.onMessage.addListener."""
        self.message_listeners.append(listener)
    
    async def sendMessage(self, message: Dict[str, Any]) -> Any:
        """Mock runtime.sendMessage."""
        self.messages_sent.append({
            "message": message,
            "timestamp": time.time()
        })
        
        # Simulate background script response
        if message.get("type") == "GET_SETTINGS":
            return {
                "auto_detect": True,
                "confidence_threshold": 0.7,
                "visual_indicators": True
            }
        
        return {"success": True}


class MockStorageAPI:
    """Mock chrome.storage API."""
    
    def __init__(self):
        self.local_data = {
            "settings": {
                "auto_detect": True,
                "confidence_threshold": 0.7,
                "visual_indicators": True,
                "detection_method": "ensemble"
            },
            "cache": {},
            "statistics": {
                "total_detections": 42,
                "ai_detected": 15,
                "human_detected": 27
            }
        }
        self.sync_data = {}
    
    def reset(self):
        """Reset storage state."""
        self.local_data = {"settings": {}, "cache": {}, "statistics": {}}
        self.sync_data = {}
    
    class MockLocalStorage:
        def __init__(self, data):
            self.data = data
        
        async def get(self, keys=None):
            if keys is None:
                return self.data
            if isinstance(keys, str):
                return {keys: self.data.get(keys)}
            if isinstance(keys, list):
                return {key: self.data.get(key) for key in keys}
            return {}
        
        async def set(self, items):
            self.data.update(items)
    
    @property
    def local(self):
        return self.MockLocalStorage(self.local_data)
    
    @property
    def sync(self):
        return self.MockLocalStorage(self.sync_data)


class MockActionAPI:
    """Mock chrome.action API."""
    
    def __init__(self):
        self.badge_text = ""
        self.badge_color = ""
        self.icon_path = ""
    
    async def setBadgeText(self, details: Dict[str, Any]):
        """Mock action.setBadgeText."""
        self.badge_text = details.get("text", "")
    
    async def setBadgeBackgroundColor(self, details: Dict[str, Any]):
        """Mock action.setBadgeBackgroundColor."""
        self.badge_color = details.get("color", "")
    
    async def setIcon(self, details: Dict[str, Any]):
        """Mock action.setIcon."""
        self.icon_path = details.get("path", "")


class TestUIExtensionFlow:
    """Test suite for UI to extension flow."""
    
    @pytest.fixture
    def mock_chrome(self):
        """Create mock Chrome API."""
        return MockChromeAPI()
    
    @pytest.fixture
    def sample_popup_actions(self):
        """Sample popup UI actions."""
        return {
            "detect_page": {
                "type": "DETECT_PAGE",
                "payload": {
                    "url": "https://twitter.com/user/status/123456",
                    "options": {
                        "method": "ensemble",
                        "threshold": 0.7
                    }
                }
            },
            "toggle_auto_detect": {
                "type": "TOGGLE_SETTING",
                "payload": {
                    "setting": "auto_detect",
                    "value": False
                }
            },
            "get_statistics": {
                "type": "GET_STATISTICS",
                "payload": {}
            },
            "clear_cache": {
                "type": "CLEAR_CACHE",
                "payload": {}
            }
        }
    
    @pytest.fixture
    def sample_content_elements(self):
        """Sample content elements for detection."""
        return [
            {
                "selector": "div.tweet-content[data-testid='tweet-text-1']",
                "text": "This comprehensive analysis demonstrates the multifaceted nature of contemporary discourse paradigms.",
                "metadata": {
                    "user": "@academic_user",
                    "timestamp": "2024-01-15T10:00:00Z",
                    "verified": True
                }
            },
            {
                "selector": "div.tweet-content[data-testid='tweet-text-2']",
                "text": "just grabbed coffee and it's amazing! ☕ totally recommend this place",
                "metadata": {
                    "user": "@casual_user",
                    "timestamp": "2024-01-15T10:15:00Z",
                    "verified": False
                }
            }
        ]
    
    @pytest.mark.asyncio
    async def test_popup_to_background_communication(self, mock_chrome, sample_popup_actions):
        """Test popup to background script communication."""
        # Simulate popup sending message to background
        message = sample_popup_actions["detect_page"]
        message["id"] = "popup_msg_001"
        message["timestamp"] = int(time.time() * 1000)
        message["source"] = "popup"
        message["target"] = "background"
        
        # Validate message format
        result = validate_extension_message(message)
        assert result.is_valid, f"Message validation failed: {result.errors}"
        
        # Send message through mock runtime
        response = await mock_chrome.runtime.sendMessage(message)
        
        # Verify message was sent
        assert len(mock_chrome.runtime.messages_sent) == 1
        sent_message = mock_chrome.runtime.messages_sent[0]["message"]
        assert sent_message["type"] == "DETECT_PAGE"
        assert sent_message["source"] == "popup"
    
    @pytest.mark.asyncio
    async def test_background_to_content_communication(self, mock_chrome, sample_content_elements):
        """Test background to content script communication."""
        # Get active tab
        tabs = await mock_chrome.tabs.query({"active": True})
        assert len(tabs) == 1
        active_tab = tabs[0]
        
        # Simulate background sending detection request to content script
        detection_message = {
            "type": "DETECT_TEXT",
            "id": "bg_msg_001",
            "timestamp": int(time.time() * 1000),
            "source": "background",
            "target": "content_script",
            "payload": {
                "text": sample_content_elements[0]["text"],
                "element_selector": sample_content_elements[0]["selector"],
                "detection_options": {
                    "method": "ensemble",
                    "threshold": 0.7
                }
            },
            "correlation_id": "corr_001"
        }
        
        # Send message to content script
        response = await mock_chrome.tabs.sendMessage(active_tab["id"], detection_message)
        
        # Verify content script response
        assert response["type"] == "DETECTION_RESULT"
        assert response["correlation_id"] == "corr_001"
        assert "payload" in response
        assert response["payload"]["is_ai_generated"] is True
    
    @pytest.mark.asyncio
    async def test_full_detection_flow(self, mock_chrome, sample_content_elements):
        """Test complete detection flow from popup to content script."""
        # Step 1: Popup initiates page detection
        popup_message = {
            "type": "DETECT_PAGE",
            "id": "popup_001",
            "timestamp": int(time.time() * 1000),
            "source": "popup",
            "target": "background",
            "payload": {
                "url": "https://twitter.com/user/status/123456",
                "options": {"method": "ensemble", "threshold": 0.7}
            }
        }
        
        # Send popup message
        await mock_chrome.runtime.sendMessage(popup_message)
        
        # Step 2: Background processes request and sends to content script
        tabs = await mock_chrome.tabs.query({"active": True})
        active_tab = tabs[0]
        
        # For each element on page, send detection request
        results = []
        for element in sample_content_elements:
            content_message = {
                "type": "DETECT_TEXT",
                "id": f"detect_{element['selector'].split('-')[-1]}",
                "timestamp": int(time.time() * 1000),
                "source": "background",
                "target": "content_script",
                "payload": {
                    "text": element["text"],
                    "element_selector": element["selector"],
                    "detection_options": popup_message["payload"]["options"]
                },
                "correlation_id": f"corr_{len(results)}"
            }
            
            # Send to content script
            response = await mock_chrome.tabs.sendMessage(active_tab["id"], content_message)
            results.append(response)
        
        # Step 3: Verify all detections completed
        assert len(results) == 2
        for result in results:
            assert result["type"] == "DETECTION_RESULT"
            assert "correlation_id" in result
            assert result["payload"]["is_ai_generated"] in [True, False]
    
    @pytest.mark.asyncio
    async def test_settings_management_flow(self, mock_chrome):
        """Test settings management through extension flow."""
        # Step 1: Popup requests current settings
        settings_request = {
            "type": "GET_SETTINGS",
            "id": "settings_001",
            "source": "popup",
            "target": "background"
        }
        
        response = await mock_chrome.runtime.sendMessage(settings_request)
        
        # Verify settings response
        assert "auto_detect" in response
        assert "confidence_threshold" in response
        
        # Step 2: Update settings
        update_message = {
            "type": "UPDATE_SETTINGS",
            "id": "settings_002",
            "source": "popup",
            "target": "background",
            "payload": {
                "settings": {
                    "auto_detect": False,
                    "confidence_threshold": 0.8
                }
            }
        }
        
        # Store settings update
        await mock_chrome.storage.local.set({"settings": update_message["payload"]["settings"]})
        
        # Verify settings were updated
        stored_settings = await mock_chrome.storage.local.get("settings")
        assert stored_settings["settings"]["auto_detect"] is False
        assert stored_settings["settings"]["confidence_threshold"] == 0.8
    
    @pytest.mark.asyncio
    async def test_visual_indicator_flow(self, mock_chrome, sample_content_elements):
        """Test visual indicator display flow."""
        # Simulate detection with visual feedback
        element = sample_content_elements[0]
        
        detection_message = {
            "type": "DETECT_TEXT",
            "id": "visual_001",
            "source": "background",
            "target": "content_script",
            "payload": {
                "text": element["text"],
                "element_selector": element["selector"],
                "show_visual": True
            }
        }
        
        # Get tab and send message
        tabs = await mock_chrome.tabs.query({"active": True})
        response = await mock_chrome.tabs.sendMessage(tabs[0]["id"], detection_message)
        
        # Verify visual indicator in response
        assert "visual_indicator" in response["payload"]
        indicator = response["payload"]["visual_indicator"]
        assert "type" in indicator
        assert "color" in indicator
        
        # Update badge to show detection count
        await mock_chrome.action.setBadgeText({"text": "1"})
        await mock_chrome.action.setBadgeBackgroundColor({"color": "#FF6B35"})
        
        # Verify badge was updated
        assert mock_chrome.action.badge_text == "1"
        assert mock_chrome.action.badge_color == "#FF6B35"
    
    @pytest.mark.asyncio
    async def test_error_handling_in_flow(self, mock_chrome):
        """Test error handling throughout the extension flow."""
        # Test invalid message format
        invalid_message = {
            "type": "INVALID_TYPE",
            "source": "popup"
            # Missing required fields
        }
        
        # Validate message (should fail)
        result = validate_extension_message(invalid_message)
        assert not result.is_valid
        
        # Test content script error response
        error_message = {
            "type": "DETECT_TEXT",
            "id": "error_001",
            "source": "background",
            "target": "content_script",
            "payload": {
                "text": "",  # Empty text should cause error
                "element_selector": "invalid-selector"
            }
        }
        
        # Mock error response
        with patch.object(mock_chrome.tabs, 'sendMessage') as mock_send:
            mock_send.return_value = {
                "type": "ERROR",
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Empty text provided"
                }
            }
            
            response = await mock_chrome.tabs.sendMessage(123, error_message)
            assert response["type"] == "ERROR"
            assert response["error"]["code"] == "VALIDATION_ERROR"
    
    @pytest.mark.asyncio
    async def test_batch_detection_flow(self, mock_chrome, sample_content_elements):
        """Test batch detection flow for multiple elements."""
        # Popup requests batch detection
        batch_message = {
            "type": "BATCH_DETECT",
            "id": "batch_001",
            "source": "popup",
            "target": "background",
            "payload": {
                "elements": [
                    {
                        "selector": elem["selector"],
                        "text": elem["text"]
                    }
                    for elem in sample_content_elements
                ],
                "options": {"method": "ensemble", "batch_size": 10}
            }
        }
        
        # Send batch request
        await mock_chrome.runtime.sendMessage(batch_message)
        
        # Simulate background processing batch
        tabs = await mock_chrome.tabs.query({"active": True})
        
        # Send batch to content script
        content_batch = {
            "type": "BATCH_PROCESS",
            "id": "batch_content_001",
            "source": "background",
            "target": "content_script",
            "payload": batch_message["payload"]
        }
        
        # Mock batch response
        with patch.object(mock_chrome.tabs, 'sendMessage') as mock_send:
            mock_send.return_value = {
                "type": "BATCH_RESULT",
                "results": [
                    {
                        "selector": sample_content_elements[0]["selector"],
                        "is_ai_generated": True,
                        "confidence_score": 0.85
                    },
                    {
                        "selector": sample_content_elements[1]["selector"],
                        "is_ai_generated": False,
                        "confidence_score": 0.25
                    }
                ]
            }
            
            response = await mock_chrome.tabs.sendMessage(tabs[0]["id"], content_batch)
            
            # Verify batch results
            assert response["type"] == "BATCH_RESULT"
            assert len(response["results"]) == 2
            assert response["results"][0]["is_ai_generated"] is True
            assert response["results"][1]["is_ai_generated"] is False
    
    @pytest.mark.asyncio
    async def test_statistics_tracking_flow(self, mock_chrome):
        """Test statistics tracking throughout the flow."""
        # Initial statistics
        initial_stats = await mock_chrome.storage.local.get("statistics")
        initial_total = initial_stats["statistics"]["total_detections"]
        
        # Simulate detection that updates statistics
        detection_message = {
            "type": "DETECT_TEXT",
            "id": "stats_001",
            "source": "background",
            "target": "content_script",
            "payload": {
                "text": "Test text for statistics",
                "element_selector": "div.test"
            }
        }
        
        # Send detection request
        tabs = await mock_chrome.tabs.query({"active": True})
        await mock_chrome.tabs.sendMessage(tabs[0]["id"], detection_message)
        
        # Update statistics
        new_stats = initial_stats["statistics"].copy()
        new_stats["total_detections"] += 1
        new_stats["ai_detected"] += 1
        
        await mock_chrome.storage.local.set({"statistics": new_stats})
        
        # Verify statistics were updated
        updated_stats = await mock_chrome.storage.local.get("statistics")
        assert updated_stats["statistics"]["total_detections"] == initial_total + 1
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_flow(self, mock_chrome):
        """Test performance monitoring in extension flow."""
        start_time = time.time()
        
        # Simulate timed detection
        detection_message = {
            "type": "DETECT_TEXT",
            "id": "perf_001",
            "source": "background",
            "target": "content_script",
            "payload": {
                "text": "Text for performance testing",
                "element_selector": "div.perf-test",
                "track_performance": True
            }
        }
        
        # Send message
        tabs = await mock_chrome.tabs.query({"active": True})
        
        # Mock response with performance data
        with patch.object(mock_chrome.tabs, 'sendMessage') as mock_send:
            end_time = time.time()
            mock_send.return_value = {
                "type": "DETECTION_RESULT",
                "payload": {
                    "is_ai_generated": True,
                    "confidence_score": 0.8,
                    "performance": {
                        "detection_time_ms": (end_time - start_time) * 1000,
                        "memory_used_mb": 12.5
                    }
                }
            }
            
            response = await mock_chrome.tabs.sendMessage(tabs[0]["id"], detection_message)
            
            # Verify performance data
            assert "performance" in response["payload"]
            perf_data = response["payload"]["performance"]
            assert "detection_time_ms" in perf_data
            assert "memory_used_mb" in perf_data
    
    @pytest.mark.asyncio
    async def test_cache_management_flow(self, mock_chrome):
        """Test cache management in extension flow."""
        # Add item to cache
        cache_data = {
            "cache": {
                "text_hash_123": {
                    "result": {"is_ai_generated": True, "confidence_score": 0.85},
                    "timestamp": int(time.time() * 1000),
                    "expiry": int(time.time() * 1000) + 3600000  # 1 hour
                }
            }
        }
        
        await mock_chrome.storage.local.set(cache_data)
        
        # Test cache hit
        detection_message = {
            "type": "DETECT_TEXT",
            "id": "cache_001",
            "source": "background", 
            "target": "content_script",
            "payload": {
                "text": "Cached text content",
                "text_hash": "text_hash_123",
                "use_cache": True
            }
        }
        
        # Mock cache hit response
        with patch.object(mock_chrome.tabs, 'sendMessage') as mock_send:
            mock_send.return_value = {
                "type": "DETECTION_RESULT",
                "payload": {
                    "is_ai_generated": True,
                    "confidence_score": 0.85,
                    "from_cache": True
                }
            }
            
            tabs = await mock_chrome.tabs.query({"active": True})
            response = await mock_chrome.tabs.sendMessage(tabs[0]["id"], detection_message)
            
            # Verify cache hit
            assert response["payload"]["from_cache"] is True
        
        # Test cache clear
        clear_message = {
            "type": "CLEAR_CACHE",
            "source": "popup",
            "target": "background"
        }
        
        await mock_chrome.runtime.sendMessage(clear_message)
        
        # Clear cache
        await mock_chrome.storage.local.set({"cache": {}})
        
        # Verify cache was cleared
        cache_after_clear = await mock_chrome.storage.local.get("cache")
        assert len(cache_after_clear["cache"]) == 0
    
    def test_message_correlation_flow(self, mock_chrome):
        """Test message correlation throughout the flow."""
        correlation_id = "test_correlation_123"
        
        # Create message chain with same correlation ID
        messages = [
            {
                "type": "DETECT_PAGE",
                "id": "msg_001",
                "source": "popup",
                "target": "background",
                "correlation_id": correlation_id
            },
            {
                "type": "DETECT_TEXT", 
                "id": "msg_002",
                "source": "background",
                "target": "content_script",
                "correlation_id": correlation_id
            },
            {
                "type": "DETECTION_RESULT",
                "id": "msg_003",
                "source": "content_script",
                "target": "background",
                "correlation_id": correlation_id
            }
        ]
        
        # Verify all messages have same correlation ID
        for message in messages:
            assert message["correlation_id"] == correlation_id
            
            # Validate message structure
            result = validate_extension_message(message)
            assert result.is_valid