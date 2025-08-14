"""
Core Message Protocol
Defines message types and standardized communication protocol
"""

import uuid
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union, Type
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod

from .exceptions import InvalidMessageError


class MessageType(Enum):
    """Standard message types"""
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    NOTIFICATION = "notification"
    COMMAND = "command"
    STATUS = "status"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


class MessagePriority(Enum):
    """Message priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


class MessageStatus(Enum):
    """Message processing status"""
    CREATED = "created"
    SENT = "sent"
    DELIVERED = "delivered"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class MessageHeaders:
    """Standard message headers"""
    
    # Core identification
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: Optional[str] = None
    conversation_id: Optional[str] = None
    
    # Routing information
    source: Optional[str] = None
    destination: Optional[str] = None
    reply_to: Optional[str] = None
    
    # Message metadata
    timestamp: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    priority: MessagePriority = MessagePriority.NORMAL
    
    # Processing information
    status: MessageStatus = MessageStatus.CREATED
    retry_count: int = 0
    max_retries: int = 3
    
    # Content information
    content_type: str = "application/json"
    content_encoding: Optional[str] = None
    content_length: Optional[int] = None
    
    # Custom headers
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert headers to dictionary"""
        data = asdict(self)
        
        # Convert datetime objects to ISO strings
        if self.timestamp:
            data["timestamp"] = self.timestamp.isoformat()
        if self.expires_at:
            data["expires_at"] = self.expires_at.isoformat()
        
        # Convert enums to values
        data["priority"] = self.priority.value
        data["status"] = self.status.value
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MessageHeaders':
        """Create headers from dictionary"""
        # Convert string timestamps back to datetime
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        
        if "expires_at" in data and isinstance(data["expires_at"], str):
            data["expires_at"] = datetime.fromisoformat(data["expires_at"])
        
        # Convert enum values back to enums
        if "priority" in data and isinstance(data["priority"], (int, str)):
            if isinstance(data["priority"], str):
                data["priority"] = MessagePriority[data["priority"]]
            else:
                data["priority"] = MessagePriority(data["priority"])
        
        if "status" in data and isinstance(data["status"], str):
            data["status"] = MessageStatus[data["status"]]
        
        # Handle missing custom dict
        if "custom" not in data:
            data["custom"] = {}
        
        return cls(**data)
    
    def is_expired(self) -> bool:
        """Check if message has expired"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def get_age(self) -> timedelta:
        """Get message age"""
        return datetime.now() - self.timestamp


@dataclass
class Message:
    """Base message class"""
    
    # Message classification
    type: MessageType
    subject: str  # Message topic/subject
    
    # Headers and metadata
    headers: MessageHeaders = field(default_factory=MessageHeaders)
    
    # Message content
    payload: Any = None
    
    # Validation schema (optional)
    schema: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Post-initialization processing"""
        # Set content length if payload exists
        if self.payload is not None and self.headers.content_length is None:
            try:
                if isinstance(self.payload, (dict, list)):
                    self.headers.content_length = len(json.dumps(self.payload))
                else:
                    self.headers.content_length = len(str(self.payload))
            except:
                pass  # Unable to calculate length
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary"""
        return {
            "type": self.type.value,
            "subject": self.subject,
            "headers": self.headers.to_dict(),
            "payload": self.payload,
            "schema": self.schema
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary"""
        # Convert message type
        if isinstance(data["type"], str):
            data["type"] = MessageType[data["type"].upper()]
        
        # Convert headers
        if "headers" in data and isinstance(data["headers"], dict):
            data["headers"] = MessageHeaders.from_dict(data["headers"])
        
        return cls(**data)
    
    def validate(self) -> tuple[bool, List[str]]:
        """Validate message content"""
        errors = []
        
        # Basic validation
        if not self.subject or not self.subject.strip():
            errors.append("Message subject cannot be empty")
        
        if self.headers.is_expired():
            errors.append("Message has expired")
        
        if self.headers.retry_count > self.headers.max_retries:
            errors.append("Message has exceeded maximum retry attempts")
        
        # Schema validation (if provided)
        if self.schema and self.payload:
            # Placeholder for JSON schema validation
            # In practice, would use jsonschema library
            pass
        
        return len(errors) == 0, errors
    
    def clone(self, **overrides) -> 'Message':
        """Create a copy of the message with optional overrides"""
        data = self.to_dict()
        data.update(overrides)
        return self.from_dict(data)
    
    def set_expiry(self, duration: timedelta):
        """Set message expiry"""
        self.headers.expires_at = datetime.now() + duration
    
    def add_custom_header(self, key: str, value: Any):
        """Add custom header"""
        self.headers.custom[key] = value
    
    def get_custom_header(self, key: str, default: Any = None) -> Any:
        """Get custom header value"""
        return self.headers.custom.get(key, default)


class RequestMessage(Message):
    """Request message with response expectations"""
    
    def __init__(self, subject: str, payload: Any = None,
                 timeout: Optional[float] = None, **kwargs):
        super().__init__(MessageType.REQUEST, subject, payload=payload, **kwargs)
        
        if timeout:
            self.set_expiry(timedelta(seconds=timeout))
        
        # Ensure we have correlation ID for tracking response
        if not self.headers.correlation_id:
            self.headers.correlation_id = self.headers.message_id


class ResponseMessage(Message):
    """Response message linked to a request"""
    
    def __init__(self, subject: str, payload: Any = None,
                 request_message: Optional[RequestMessage] = None,
                 success: bool = True, **kwargs):
        super().__init__(MessageType.RESPONSE, subject, payload=payload, **kwargs)
        
        if request_message:
            self.headers.correlation_id = request_message.headers.correlation_id
            self.headers.destination = request_message.headers.source
        
        self.add_custom_header("success", success)


class EventMessage(Message):
    """Event message for publishing events"""
    
    def __init__(self, subject: str, payload: Any = None,
                 event_data: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(MessageType.EVENT, subject, payload=payload, **kwargs)
        
        if event_data:
            for key, value in event_data.items():
                self.add_custom_header(f"event_{key}", value)


class NotificationMessage(Message):
    """Notification message for alerts and notifications"""
    
    def __init__(self, subject: str, payload: Any = None,
                 severity: str = "info", category: str = "general", **kwargs):
        super().__init__(MessageType.NOTIFICATION, subject, payload=payload, **kwargs)
        
        self.add_custom_header("severity", severity)
        self.add_custom_header("category", category)


class CommandMessage(Message):
    """Command message for system commands"""
    
    def __init__(self, subject: str, payload: Any = None,
                 command: str = None, parameters: Optional[Dict[str, Any]] = None,
                 **kwargs):
        super().__init__(MessageType.COMMAND, subject, payload=payload, **kwargs)
        
        if command:
            self.add_custom_header("command", command)
        
        if parameters:
            self.add_custom_header("parameters", parameters)


class StatusMessage(Message):
    """Status message for system status updates"""
    
    def __init__(self, subject: str, payload: Any = None,
                 component: str = None, status: str = "ok",
                 metrics: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(MessageType.STATUS, subject, payload=payload, **kwargs)
        
        if component:
            self.add_custom_header("component", component)
        
        self.add_custom_header("status", status)
        
        if metrics:
            self.add_custom_header("metrics", metrics)


class ErrorMessage(Message):
    """Error message for error reporting"""
    
    def __init__(self, subject: str, payload: Any = None,
                 error_code: str = None, error_details: Optional[Dict[str, Any]] = None,
                 **kwargs):
        super().__init__(MessageType.ERROR, subject, payload=payload, **kwargs)
        
        if error_code:
            self.add_custom_header("error_code", error_code)
        
        if error_details:
            self.add_custom_header("error_details", error_details)


class MessageProtocol:
    """Message protocol handler and utilities"""
    
    @staticmethod
    def create_message(message_type: Union[MessageType, str], 
                      subject: str, payload: Any = None,
                      **kwargs) -> Message:
        """Create message of appropriate type"""
        
        if isinstance(message_type, str):
            message_type = MessageType[message_type.upper()]
        
        message_classes = {
            MessageType.REQUEST: RequestMessage,
            MessageType.RESPONSE: ResponseMessage,
            MessageType.EVENT: EventMessage,
            MessageType.NOTIFICATION: NotificationMessage,
            MessageType.COMMAND: CommandMessage,
            MessageType.STATUS: StatusMessage,
            MessageType.ERROR: ErrorMessage
        }
        
        message_class = message_classes.get(message_type, Message)
        return message_class(subject, payload=payload, **kwargs)
    
    @staticmethod
    def validate_message(message: Message) -> None:
        """Validate message and raise exception if invalid"""
        is_valid, errors = message.validate()
        
        if not is_valid:
            raise InvalidMessageError(
                "Message validation failed",
                message_data=message.to_dict(),
                validation_errors=errors
            )
    
    @staticmethod
    def create_response(request: RequestMessage, payload: Any = None,
                       success: bool = True, subject: str = None) -> ResponseMessage:
        """Create response for a request message"""
        
        if subject is None:
            subject = f"response_{request.subject}"
        
        return ResponseMessage(
            subject=subject,
            payload=payload,
            request_message=request,
            success=success
        )
    
    @staticmethod
    def create_error_response(request: RequestMessage, error: Exception,
                             error_code: str = None) -> ErrorMessage:
        """Create error response for a failed request"""
        
        error_details = {
            "original_request": request.subject,
            "error_message": str(error),
            "error_type": type(error).__name__
        }
        
        return ErrorMessage(
            subject=f"error_{request.subject}",
            payload=error_details,
            error_code=error_code,
            error_details=error_details
        )
    
    @staticmethod
    def get_message_stats(messages: List[Message]) -> Dict[str, Any]:
        """Get statistics for a collection of messages"""
        if not messages:
            return {
                "total_messages": 0,
                "by_type": {},
                "by_priority": {},
                "by_status": {},
                "average_age": 0,
                "expired_count": 0
            }
        
        # Count by type
        by_type = {}
        for msg in messages:
            msg_type = msg.type.value
            by_type[msg_type] = by_type.get(msg_type, 0) + 1
        
        # Count by priority
        by_priority = {}
        for msg in messages:
            priority = msg.headers.priority.name
            by_priority[priority] = by_priority.get(priority, 0) + 1
        
        # Count by status
        by_status = {}
        for msg in messages:
            status = msg.headers.status.value
            by_status[status] = by_status.get(status, 0) + 1
        
        # Calculate average age
        total_age = sum(msg.headers.get_age().total_seconds() for msg in messages)
        average_age = total_age / len(messages)
        
        # Count expired messages
        expired_count = sum(1 for msg in messages if msg.headers.is_expired())
        
        return {
            "total_messages": len(messages),
            "by_type": by_type,
            "by_priority": by_priority,
            "by_status": by_status,
            "average_age": average_age,
            "expired_count": expired_count
        }