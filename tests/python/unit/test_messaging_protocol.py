"""
Unit Tests for Messaging Protocol System
Comprehensive tests for message protocol and communication
"""

import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import uuid

from src.core.messaging.protocol import (
    Message, MessageType, MessagePriority, MessageStatus, MessageHeaders,
    RequestMessage, ResponseMessage, EventMessage, NotificationMessage,
    CommandMessage, StatusMessage, ErrorMessage, MessageProtocol
)
from src.core.messaging.exceptions import InvalidMessageError


class TestMessageHeaders:
    """Test suite for MessageHeaders"""
    
    @pytest.mark.unit
    def test_headers_creation(self):
        """Test message headers creation"""
        headers = MessageHeaders()
        
        assert headers.message_id is not None
        assert isinstance(headers.timestamp, datetime)
        assert headers.priority == MessagePriority.NORMAL
        assert headers.status == MessageStatus.CREATED
        assert headers.retry_count == 0
        assert headers.max_retries == 3
    
    @pytest.mark.unit
    def test_headers_custom_values(self):
        """Test headers with custom values"""
        custom_id = str(uuid.uuid4())
        custom_time = datetime.now()
        
        headers = MessageHeaders(
            message_id=custom_id,
            correlation_id="test-correlation",
            source="test-service",
            destination="target-service",
            priority=MessagePriority.HIGH,
            timestamp=custom_time
        )
        
        assert headers.message_id == custom_id
        assert headers.correlation_id == "test-correlation"
        assert headers.source == "test-service"
        assert headers.destination == "target-service"
        assert headers.priority == MessagePriority.HIGH
        assert headers.timestamp == custom_time
    
    @pytest.mark.unit
    def test_headers_serialization(self):
        """Test headers serialization/deserialization"""
        headers = MessageHeaders(
            correlation_id="test-123",
            priority=MessagePriority.HIGH,
            custom={"key1": "value1", "key2": 42}
        )
        
        # Test to_dict
        headers_dict = headers.to_dict()
        assert isinstance(headers_dict, dict)
        assert headers_dict["correlation_id"] == "test-123"
        assert headers_dict["priority"] == MessagePriority.HIGH.value
        assert headers_dict["custom"]["key1"] == "value1"
        
        # Test from_dict
        restored_headers = MessageHeaders.from_dict(headers_dict)
        assert restored_headers.correlation_id == headers.correlation_id
        assert restored_headers.priority == headers.priority
        assert restored_headers.custom == headers.custom
    
    @pytest.mark.unit
    def test_headers_expiry(self):
        """Test message expiry functionality"""
        headers = MessageHeaders()
        
        # Not expired by default
        assert not headers.is_expired()
        
        # Set expiry in the past
        headers.expires_at = datetime.now() - timedelta(seconds=1)
        assert headers.is_expired()
        
        # Set expiry in the future
        headers.expires_at = datetime.now() + timedelta(hours=1)
        assert not headers.is_expired()
    
    @pytest.mark.unit
    def test_headers_age(self):
        """Test message age calculation"""
        past_time = datetime.now() - timedelta(seconds=30)
        headers = MessageHeaders(timestamp=past_time)
        
        age = headers.get_age()
        assert isinstance(age, timedelta)
        assert age.total_seconds() >= 30


class TestMessage:
    """Test suite for base Message class"""
    
    @pytest.mark.unit
    def test_message_creation(self):
        """Test basic message creation"""
        message = Message(
            type=MessageType.REQUEST,
            subject="test_message",
            payload={"key": "value"}
        )
        
        assert message.type == MessageType.REQUEST
        assert message.subject == "test_message"
        assert message.payload == {"key": "value"}
        assert isinstance(message.headers, MessageHeaders)
    
    @pytest.mark.unit
    def test_message_serialization(self):
        """Test message serialization"""
        message = Message(
            type=MessageType.EVENT,
            subject="user_action",
            payload={"action": "click", "element": "button"}
        )
        
        # Test to_dict
        message_dict = message.to_dict()
        assert isinstance(message_dict, dict)
        assert message_dict["type"] == "event"
        assert message_dict["subject"] == "user_action"
        assert message_dict["payload"]["action"] == "click"
        
        # Test from_dict
        restored_message = Message.from_dict(message_dict)
        assert restored_message.type == message.type
        assert restored_message.subject == message.subject
        assert restored_message.payload == message.payload
    
    @pytest.mark.unit
    def test_message_validation(self):
        """Test message validation"""
        # Valid message
        valid_message = Message(
            type=MessageType.REQUEST,
            subject="valid_subject",
            payload={"data": "test"}
        )
        
        is_valid, errors = valid_message.validate()
        assert is_valid
        assert len(errors) == 0
        
        # Invalid message - empty subject
        invalid_message = Message(
            type=MessageType.REQUEST,
            subject="",
            payload={"data": "test"}
        )
        
        is_valid, errors = invalid_message.validate()
        assert not is_valid
        assert len(errors) > 0
        assert any("subject" in error.lower() for error in errors)
    
    @pytest.mark.unit
    def test_message_custom_headers(self):
        """Test message custom headers"""
        message = Message(
            type=MessageType.NOTIFICATION,
            subject="test_notification"
        )
        
        # Add custom header
        message.add_custom_header("priority_level", "urgent")
        message.add_custom_header("source_system", "monitoring")
        
        # Get custom headers
        assert message.get_custom_header("priority_level") == "urgent"
        assert message.get_custom_header("source_system") == "monitoring"
        assert message.get_custom_header("nonexistent", "default") == "default"
    
    @pytest.mark.unit
    def test_message_expiry(self):
        """Test message expiry setting"""
        message = Message(
            type=MessageType.REQUEST,
            subject="timed_request"
        )
        
        # Set expiry
        duration = timedelta(minutes=5)
        message.set_expiry(duration)
        
        assert message.headers.expires_at is not None
        assert message.headers.expires_at > datetime.now()
        assert not message.headers.is_expired()
    
    @pytest.mark.unit
    def test_message_cloning(self):
        """Test message cloning"""
        original = Message(
            type=MessageType.COMMAND,
            subject="original_command",
            payload={"command": "start", "params": {"timeout": 30}}
        )
        original.add_custom_header("trace_id", "abc123")
        
        # Clone with overrides
        clone = original.clone(
            subject="cloned_command",
            payload={"command": "stop"}
        )
        
        assert clone.subject == "cloned_command"
        assert clone.payload["command"] == "stop"
        assert clone.type == original.type  # Unchanged
        assert clone.headers.message_id != original.headers.message_id  # New ID


class TestSpecializedMessages:
    """Test suite for specialized message types"""
    
    @pytest.mark.unit
    def test_request_message(self):
        """Test RequestMessage functionality"""
        request = RequestMessage(
            subject="data_request",
            payload={"query": "SELECT * FROM users"},
            timeout=30.0
        )
        
        assert request.type == MessageType.REQUEST
        assert request.subject == "data_request"
        assert request.headers.correlation_id is not None
        assert request.headers.expires_at is not None
    
    @pytest.mark.unit
    def test_response_message(self):
        """Test ResponseMessage functionality"""
        # Create original request
        request = RequestMessage(
            subject="test_request",
            payload={"action": "get_data"}
        )
        
        # Create response
        response = ResponseMessage(
            subject="test_response",
            payload={"data": [1, 2, 3], "count": 3},
            request_message=request,
            success=True
        )
        
        assert response.type == MessageType.RESPONSE
        assert response.headers.correlation_id == request.headers.correlation_id
        assert response.get_custom_header("success") is True
    
    @pytest.mark.unit
    def test_event_message(self):
        """Test EventMessage functionality"""
        event = EventMessage(
            subject="user_registered",
            payload={"user_id": 123, "email": "test@example.com"},
            event_data={
                "timestamp": "2024-01-01T12:00:00Z",
                "source": "registration_service"
            }
        )
        
        assert event.type == MessageType.EVENT
        assert event.get_custom_header("event_timestamp") == "2024-01-01T12:00:00Z"
        assert event.get_custom_header("event_source") == "registration_service"
    
    @pytest.mark.unit
    def test_notification_message(self):
        """Test NotificationMessage functionality"""
        notification = NotificationMessage(
            subject="system_alert",
            payload={"message": "High CPU usage detected", "cpu_usage": 95.5},
            severity="warning",
            category="system"
        )
        
        assert notification.type == MessageType.NOTIFICATION
        assert notification.get_custom_header("severity") == "warning"
        assert notification.get_custom_header("category") == "system"
    
    @pytest.mark.unit
    def test_command_message(self):
        """Test CommandMessage functionality"""
        command = CommandMessage(
            subject="system_command",
            payload={"target": "database", "action": "backup"},
            command="backup_database",
            parameters={"include_indexes": True, "compression": "gzip"}
        )
        
        assert command.type == MessageType.COMMAND
        assert command.get_custom_header("command") == "backup_database"
        assert command.get_custom_header("parameters")["include_indexes"] is True
    
    @pytest.mark.unit
    def test_status_message(self):
        """Test StatusMessage functionality"""
        status = StatusMessage(
            subject="service_status",
            payload={"uptime": 3600, "connections": 42},
            component="api_server",
            status="healthy",
            metrics={"response_time": 150, "throughput": 1000}
        )
        
        assert status.type == MessageType.STATUS
        assert status.get_custom_header("component") == "api_server"
        assert status.get_custom_header("status") == "healthy"
        assert status.get_custom_header("metrics")["response_time"] == 150
    
    @pytest.mark.unit
    def test_error_message(self):
        """Test ErrorMessage functionality"""
        error = ErrorMessage(
            subject="processing_error",
            payload={"error": "Failed to process request", "request_id": "req_123"},
            error_code="PROCESSING_FAILED",
            error_details={
                "stack_trace": "Error at line 42",
                "context": {"user_id": 456}
            }
        )
        
        assert error.type == MessageType.ERROR
        assert error.get_custom_header("error_code") == "PROCESSING_FAILED"
        assert error.get_custom_header("error_details")["context"]["user_id"] == 456


class TestMessageProtocol:
    """Test suite for MessageProtocol utilities"""
    
    @pytest.mark.unit
    def test_create_message(self):
        """Test message creation through protocol"""
        # Test creating different message types
        request = MessageProtocol.create_message(
            MessageType.REQUEST,
            "test_request",
            payload={"data": "test"}
        )
        assert isinstance(request, RequestMessage)
        assert request.type == MessageType.REQUEST
        
        # Test with string type
        event = MessageProtocol.create_message(
            "event",
            "test_event",
            payload={"event_data": "test"}
        )
        assert isinstance(event, EventMessage)
        assert event.type == MessageType.EVENT
    
    @pytest.mark.unit
    def test_validate_message(self):
        """Test message validation through protocol"""
        # Valid message
        valid_message = Message(
            type=MessageType.REQUEST,
            subject="valid_message",
            payload={"data": "test"}
        )
        
        # Should not raise exception
        MessageProtocol.validate_message(valid_message)
        
        # Invalid message
        invalid_message = Message(
            type=MessageType.REQUEST,
            subject="",  # Empty subject
            payload={"data": "test"}
        )
        
        with pytest.raises(InvalidMessageError):
            MessageProtocol.validate_message(invalid_message)
    
    @pytest.mark.unit
    def test_create_response(self):
        """Test response creation through protocol"""
        # Create request
        request = RequestMessage(
            subject="test_request",
            payload={"query": "test"}
        )
        
        # Create response
        response = MessageProtocol.create_response(
            request,
            payload={"result": "success"},
            success=True
        )
        
        assert isinstance(response, ResponseMessage)
        assert response.headers.correlation_id == request.headers.correlation_id
        assert response.get_custom_header("success") is True
    
    @pytest.mark.unit
    def test_create_error_response(self):
        """Test error response creation"""
        request = RequestMessage(
            subject="failing_request",
            payload={"action": "fail"}
        )
        
        error = Exception("Something went wrong")
        error_response = MessageProtocol.create_error_response(
            request,
            error,
            error_code="REQUEST_FAILED"
        )
        
        assert isinstance(error_response, ErrorMessage)
        assert error_response.get_custom_header("error_code") == "REQUEST_FAILED"
        assert "failing_request" in error_response.subject
    
    @pytest.mark.unit
    def test_message_statistics(self):
        """Test message statistics calculation"""
        messages = [
            Message(MessageType.REQUEST, "req1"),
            Message(MessageType.RESPONSE, "resp1"),
            Message(MessageType.EVENT, "event1"),
            Message(MessageType.REQUEST, "req2")
        ]
        
        # Set different priorities
        messages[0].headers.priority = MessagePriority.HIGH
        messages[1].headers.priority = MessagePriority.NORMAL
        messages[2].headers.priority = MessagePriority.LOW
        messages[3].headers.priority = MessagePriority.HIGH
        
        # Set different statuses
        messages[0].headers.status = MessageStatus.COMPLETED
        messages[1].headers.status = MessageStatus.PROCESSING
        messages[2].headers.status = MessageStatus.FAILED
        messages[3].headers.status = MessageStatus.COMPLETED
        
        stats = MessageProtocol.get_message_stats(messages)
        
        assert stats["total_messages"] == 4
        assert stats["by_type"]["request"] == 2
        assert stats["by_type"]["response"] == 1
        assert stats["by_type"]["event"] == 1
        assert stats["by_priority"]["HIGH"] == 2
        assert stats["by_priority"]["NORMAL"] == 1
        assert stats["by_priority"]["LOW"] == 1
        assert stats["by_status"]["completed"] == 2
        assert stats["by_status"]["processing"] == 1
        assert stats["by_status"]["failed"] == 1
    
    @pytest.mark.unit
    def test_empty_message_statistics(self):
        """Test statistics with empty message list"""
        stats = MessageProtocol.get_message_stats([])
        
        assert stats["total_messages"] == 0
        assert stats["by_type"] == {}
        assert stats["by_priority"] == {}
        assert stats["by_status"] == {}
        assert stats["average_age"] == 0
        assert stats["expired_count"] == 0


class TestMessageExceptions:
    """Test suite for message-related exceptions"""
    
    @pytest.mark.unit
    def test_invalid_message_error(self):
        """Test InvalidMessageError exception"""
        message_data = {"type": "invalid", "subject": ""}
        validation_errors = ["Subject cannot be empty", "Invalid type"]
        
        error = InvalidMessageError(
            "Message validation failed",
            message_data=message_data,
            validation_errors=validation_errors
        )
        
        assert str(error) == "Message validation failed"
        assert error.message_data == message_data
        assert error.validation_errors == validation_errors
        
        # Test serialization
        error_dict = error.to_dict()
        assert error_dict["error_type"] == "InvalidMessageError"
        assert error_dict["message_data"] == message_data
        assert error_dict["validation_errors"] == validation_errors


@pytest.mark.integration
class TestMessageProtocolIntegration:
    """Integration tests for message protocol"""
    
    @pytest.mark.integration
    async def test_request_response_flow(self):
        """Test complete request-response message flow"""
        # Create request
        request = RequestMessage(
            subject="calculate_sum",
            payload={"numbers": [1, 2, 3, 4, 5]},
            timeout=10.0
        )
        
        # Validate request
        MessageProtocol.validate_message(request)
        
        # Simulate processing
        numbers = request.payload["numbers"]
        result = sum(numbers)
        
        # Create response
        response = MessageProtocol.create_response(
            request,
            payload={"sum": result, "count": len(numbers)},
            success=True
        )
        
        # Validate response
        MessageProtocol.validate_message(response)
        
        # Verify correlation
        assert response.headers.correlation_id == request.headers.correlation_id
        assert response.payload["sum"] == 15
        assert response.payload["count"] == 5
    
    @pytest.mark.integration
    async def test_event_notification_flow(self):
        """Test event and notification message flow"""
        # Create event
        event = EventMessage(
            subject="user_login",
            payload={
                "user_id": 123,
                "ip_address": "192.168.1.100",
                "user_agent": "Mozilla/5.0..."
            },
            event_data={
                "timestamp": datetime.now().isoformat(),
                "source": "auth_service"
            }
        )
        
        # Create related notification
        notification = NotificationMessage(
            subject="security_alert",
            payload={
                "message": "New login from unknown location",
                "user_id": 123,
                "risk_score": 0.75
            },
            severity="warning",
            category="security"
        )
        
        # Both should be valid
        MessageProtocol.validate_message(event)
        MessageProtocol.validate_message(notification)
        
        # Should have appropriate types
        assert event.type == MessageType.EVENT
        assert notification.type == MessageType.NOTIFICATION
        assert notification.get_custom_header("severity") == "warning"
    
    @pytest.mark.integration
    async def test_command_status_flow(self):
        """Test command and status message flow"""
        # Create command
        command = CommandMessage(
            subject="system_maintenance",
            payload={"action": "restart_service", "service": "api_server"},
            command="restart",
            parameters={"graceful": True, "timeout": 30}
        )
        
        # Create status updates
        status_starting = StatusMessage(
            subject="maintenance_status",
            payload={"phase": "starting", "progress": 0},
            component="api_server",
            status="maintenance"
        )
        
        status_complete = StatusMessage(
            subject="maintenance_status", 
            payload={"phase": "complete", "progress": 100},
            component="api_server",
            status="healthy",
            metrics={"restart_time": 25.5, "connections": 0}
        )
        
        # All should be valid
        for msg in [command, status_starting, status_complete]:
            MessageProtocol.validate_message(msg)
        
        # Verify flow
        assert command.type == MessageType.COMMAND
        assert status_starting.get_custom_header("status") == "maintenance"
        assert status_complete.get_custom_header("status") == "healthy"