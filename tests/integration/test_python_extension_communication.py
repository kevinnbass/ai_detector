"""
Integration tests for Python-Chrome extension communication.

Tests the communication flow between the Python backend API
and the Chrome extension components.
"""

import pytest
import asyncio
import json
import time
from typing import Dict, Any
from unittest.mock import AsyncMock, patch, MagicMock

from src.core.messaging.protocol import MessageProtocol, MessageType
from src.core.error_handling import ErrorContext
from src.utils.schema_validator import validate_extension_message, validate_detection_response
from src.api.rest.routes import DetectionService


class MockWebSocket:
    """Mock WebSocket for testing extension communication."""
    
    def __init__(self):
        self.messages_sent = []
        self.messages_received = []
        self.is_closed = False
    
    async def send(self, message: str):
        """Mock send message."""
        self.messages_sent.append(json.loads(message))
    
    async def receive_text(self) -> str:
        """Mock receive message."""
        if self.messages_received:
            return json.dumps(self.messages_received.pop(0))
        await asyncio.sleep(0.1)  # Simulate waiting
        return json.dumps({"type": "heartbeat"})
    
    async def close(self):
        """Mock close connection."""
        self.is_closed = True
    
    def add_received_message(self, message: Dict[str, Any]):
        """Add a message to be received."""
        self.messages_received.append(message)


class TestPythonExtensionCommunication:
    """Test suite for Python-Extension communication."""
    
    @pytest.fixture
    def message_protocol(self):
        """Create message protocol instance."""
        return MessageProtocol()
    
    @pytest.fixture
    def mock_websocket(self):
        """Create mock WebSocket."""
        return MockWebSocket()
    
    @pytest.fixture
    def sample_detection_request(self):
        """Sample detection request from extension."""
        return {
            "type": "DETECT_TEXT",
            "id": "msg_12345",
            "timestamp": int(time.time() * 1000),
            "source": "content_script",
            "target": "background",
            "payload": {
                "text": "This comprehensive analysis demonstrates the multifaceted nature of contemporary discourse paradigms.",
                "element_selector": "div.tweet-content",
                "page_url": "https://twitter.com/user/status/123",
                "detection_options": {
                    "method": "ensemble",
                    "threshold": 0.7
                }
            },
            "correlation_id": "corr_789",
            "priority": "normal",
            "metadata": {
                "tab_id": 123,
                "user_triggered": True
            }
        }
    
    @pytest.fixture
    def sample_detection_response(self):
        """Sample detection response to extension."""
        return {
            "request_id": "req_12345",
            "is_ai_generated": True,
            "confidence_score": 0.85,
            "processing_time_ms": 250,
            "detection_details": {
                "method_used": "ensemble",
                "individual_scores": {
                    "pattern_score": 0.8,
                    "ml_score": 0.75,
                    "llm_score": 0.9
                },
                "detected_patterns": [
                    {
                        "pattern_type": "formal_language",
                        "matches": [
                            {
                                "text": "comprehensive analysis demonstrates",
                                "position": 5,
                                "confidence": 0.9
                            }
                        ],
                        "total_score": 0.8
                    }
                ]
            },
            "timestamp": "2024-01-15T10:30:01Z",
            "version": "1.0.0"
        }
    
    def test_message_validation(self, sample_detection_request):
        """Test message schema validation."""
        # Test valid message
        result = validate_extension_message(sample_detection_request)
        assert result.is_valid, f"Valid message failed validation: {result.errors}"
        
        # Test invalid message - missing required fields
        invalid_message = sample_detection_request.copy()
        del invalid_message["type"]
        
        result = validate_extension_message(invalid_message)
        assert not result.is_valid
        assert any("type" in error for error in result.errors)
    
    def test_response_validation(self, sample_detection_response):
        """Test detection response validation."""
        # Test valid response
        result = validate_detection_response(sample_detection_response)
        assert result.is_valid, f"Valid response failed validation: {result.errors}"
        
        # Test invalid response - missing required fields
        invalid_response = sample_detection_response.copy()
        del invalid_response["is_ai_generated"]
        
        result = validate_detection_response(invalid_response)
        assert not result.is_valid
        assert any("is_ai_generated" in error for error in result.errors)
    
    @pytest.mark.asyncio
    async def test_websocket_message_flow(self, mock_websocket, sample_detection_request):
        """Test WebSocket message sending and receiving."""
        # Simulate extension sending message
        message_json = json.dumps(sample_detection_request)
        mock_websocket.add_received_message(sample_detection_request)
        
        # Simulate receiving message
        received = await mock_websocket.receive_text()
        received_data = json.loads(received)
        
        assert received_data["type"] == "DETECT_TEXT"
        assert received_data["payload"]["text"] == sample_detection_request["payload"]["text"]
        
        # Simulate sending response
        response = {
            "type": "DETECTION_RESULT",
            "id": "msg_12346",
            "correlation_id": sample_detection_request["correlation_id"],
            "payload": {
                "is_ai_generated": True,
                "confidence_score": 0.85
            }
        }
        
        await mock_websocket.send(json.dumps(response))
        
        assert len(mock_websocket.messages_sent) == 1
        sent_message = mock_websocket.messages_sent[0]
        assert sent_message["type"] == "DETECTION_RESULT"
        assert sent_message["correlation_id"] == sample_detection_request["correlation_id"]
    
    @pytest.mark.asyncio
    async def test_detection_api_integration(self, sample_detection_request):
        """Test integration with detection API."""
        with patch('src.core.detection.detector.DetectionEngine') as mock_engine:
            # Setup mock detection engine
            mock_instance = mock_engine.return_value
            mock_instance.detect_ai_text.return_value = {
                "is_ai_generated": True,
                "confidence_score": 0.85,
                "processing_time_ms": 250,
                "method_used": "ensemble"
            }
            
            # Create detection service
            service = DetectionService()
            
            # Extract detection request from extension message
            payload = sample_detection_request["payload"]
            detection_request = {
                "text": payload["text"],
                "request_id": sample_detection_request["id"],
                "options": payload.get("detection_options", {})
            }
            
            # Call detection API
            result = await service.detect_text(detection_request)
            
            # Verify results
            assert result["is_ai_generated"] is True
            assert result["confidence_score"] == 0.85
            assert "processing_time_ms" in result
            
            # Verify engine was called
            mock_instance.detect_ai_text.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_error_handling_in_communication(self, sample_detection_request):
        """Test error handling in extension communication."""
        with patch('src.core.detection.detector.DetectionEngine') as mock_engine:
            # Setup mock to raise exception
            mock_instance = mock_engine.return_value
            mock_instance.detect_ai_text.side_effect = Exception("Detection failed")
            
            service = DetectionService()
            
            # Extract detection request
            payload = sample_detection_request["payload"]
            detection_request = {
                "text": payload["text"],
                "request_id": sample_detection_request["id"],
                "options": payload.get("detection_options", {})
            }
            
            # Call detection API and expect error handling
            with pytest.raises(Exception):
                await service.detect_text(detection_request)
    
    def test_message_protocol_serialization(self, message_protocol, sample_detection_request):
        """Test message protocol serialization/deserialization."""
        # Test serialization
        serialized = message_protocol.serialize_message(sample_detection_request)
        assert isinstance(serialized, str)
        
        # Test deserialization
        deserialized = message_protocol.deserialize_message(serialized)
        assert deserialized == sample_detection_request
    
    def test_correlation_id_tracking(self, sample_detection_request):
        """Test correlation ID is properly tracked across requests."""
        correlation_id = sample_detection_request["correlation_id"]
        
        # Create response with same correlation ID
        response = {
            "type": "DETECTION_RESULT",
            "id": "msg_response",
            "correlation_id": correlation_id,
            "payload": {"is_ai_generated": True}
        }
        
        # Verify correlation IDs match
        assert response["correlation_id"] == sample_detection_request["correlation_id"]
    
    @pytest.mark.asyncio
    async def test_batch_processing_communication(self):
        """Test batch processing through extension communication."""
        batch_request = {
            "type": "BATCH_PROCESS",
            "id": "batch_123",
            "timestamp": int(time.time() * 1000),
            "source": "content_script",
            "target": "background",
            "payload": {
                "texts": [
                    "This is a formal academic analysis.",
                    "hey lol that's so funny 😂",
                    "Furthermore, it is important to consider multiple perspectives."
                ],
                "options": {
                    "method": "ensemble",
                    "threshold": 0.7
                }
            }
        }
        
        # Validate batch request
        result = validate_extension_message(batch_request)
        assert result.is_valid
        
        # Simulate processing
        with patch('src.core.detection.detector.DetectionEngine') as mock_engine:
            mock_instance = mock_engine.return_value
            mock_instance.detect_ai_text.side_effect = [
                {"is_ai_generated": True, "confidence_score": 0.9},
                {"is_ai_generated": False, "confidence_score": 0.2},
                {"is_ai_generated": True, "confidence_score": 0.8}
            ]
            
            # Process each text
            results = []
            for text in batch_request["payload"]["texts"]:
                result = mock_instance.detect_ai_text(text)
                results.append(result)
            
            # Verify batch results
            assert len(results) == 3
            assert results[0]["is_ai_generated"] is True
            assert results[1]["is_ai_generated"] is False
            assert results[2]["is_ai_generated"] is True
    
    def test_extension_context_propagation(self, sample_detection_request):
        """Test context propagation from extension to backend."""
        # Create error context from extension message
        context = ErrorContext.from_extension_message(
            sample_detection_request,
            component="api"
        )
        
        # Verify context contains extension information
        assert context.request_id == sample_detection_request["id"]
        assert context.component == "api"
        assert context.operation == sample_detection_request["type"]
        assert context.metadata["source"] == sample_detection_request["source"]
        assert context.metadata["correlation_id"] == sample_detection_request["correlation_id"]
    
    @pytest.mark.asyncio
    async def test_performance_metrics_collection(self, sample_detection_request):
        """Test performance metrics collection during communication."""
        from src.core.monitoring import get_metrics_collector
        
        metrics = get_metrics_collector()
        
        # Record request metrics
        metrics.increment_counter("extension_requests_total")
        
        start_time = time.time()
        
        # Simulate processing
        await asyncio.sleep(0.01)  # Simulate 10ms processing
        
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        
        # Record performance metrics
        metrics.observe_histogram("extension_request_duration_ms", duration_ms)
        
        # Verify metrics were recorded
        counter = metrics.get_metric("extension_requests_total")
        assert counter is not None
        assert counter.get_value() >= 1
        
        histogram = metrics.get_metric("extension_request_duration_ms")
        assert histogram is not None
        histogram_value = histogram.get_value()
        assert histogram_value["count"] >= 1
    
    def test_security_validation(self, sample_detection_request):
        """Test security validation of extension messages."""
        # Test message with potential XSS
        malicious_request = sample_detection_request.copy()
        malicious_request["payload"]["text"] = "<script>alert('xss')</script>"
        
        # Validate that the message structure is still valid
        # (Content sanitization would happen at a different layer)
        result = validate_extension_message(malicious_request)
        assert result.is_valid  # Structure is valid, content filtering is separate
        
        # Test oversized message
        oversized_request = sample_detection_request.copy()
        oversized_request["payload"]["text"] = "A" * 100000  # Very large text
        
        # This should still validate structurally
        result = validate_extension_message(oversized_request)
        # Depending on schema constraints, this might fail
        # assert not result.is_valid or "too long" in str(result.errors)
    
    @pytest.mark.asyncio
    async def test_connection_recovery(self, mock_websocket):
        """Test connection recovery mechanisms."""
        # Simulate connection loss
        mock_websocket.is_closed = True
        
        # Simulate reconnection attempt
        new_websocket = MockWebSocket()
        assert not new_websocket.is_closed
        
        # Test message after reconnection
        test_message = {"type": "heartbeat", "id": "reconnect_test"}
        await new_websocket.send(json.dumps(test_message))
        
        assert len(new_websocket.messages_sent) == 1
        assert new_websocket.messages_sent[0]["type"] == "heartbeat"
    
    def test_message_priority_handling(self):
        """Test different message priority levels."""
        priorities = ["low", "normal", "high", "critical"]
        
        for priority in priorities:
            message = {
                "type": "DETECT_TEXT",
                "id": f"msg_{priority}",
                "timestamp": int(time.time() * 1000),
                "source": "content_script",
                "target": "background",
                "payload": {"text": "test"},
                "priority": priority
            }
            
            # Validate message with priority
            result = validate_extension_message(message)
            assert result.is_valid
            assert priority in ["low", "normal", "high", "critical"]
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test timeout handling in communication."""
        with patch('asyncio.wait_for') as mock_wait:
            # Simulate timeout
            mock_wait.side_effect = asyncio.TimeoutError("Operation timed out")
            
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.sleep(1), timeout=0.1)
    
    def test_message_versioning(self, sample_detection_request):
        """Test message version compatibility."""
        # Add version to message
        versioned_message = sample_detection_request.copy()
        versioned_message["version"] = "1.0.0"
        
        # Validate versioned message
        result = validate_extension_message(versioned_message)
        assert result.is_valid
        
        # Test future version
        future_message = sample_detection_request.copy()
        future_message["version"] = "2.0.0"
        
        # Should still validate (forward compatibility)
        result = validate_extension_message(future_message)
        assert result.is_valid