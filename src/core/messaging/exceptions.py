"""
Message Protocol Exception Classes
Comprehensive error handling for messaging system
"""

from typing import Dict, Any, Optional
from datetime import datetime


class MessageProtocolError(Exception):
    """Base exception for message protocol errors"""
    
    def __init__(self, message: str, 
                 error_code: Optional[str] = None,
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary"""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "context": self.context,
            "timestamp": self.timestamp.isoformat()
        }


class InvalidMessageError(MessageProtocolError):
    """Exception for invalid message format or content"""
    
    def __init__(self, message: str = "Invalid message format",
                 message_data: Optional[Dict[str, Any]] = None,
                 validation_errors: Optional[list] = None,
                 **kwargs):
        super().__init__(message, error_code="INVALID_MESSAGE", **kwargs)
        self.message_data = message_data
        self.validation_errors = validation_errors or []
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "message_data": self.message_data,
            "validation_errors": self.validation_errors
        })
        return data


class SerializationError(MessageProtocolError):
    """Exception for message serialization/deserialization errors"""
    
    def __init__(self, message: str = "Message serialization failed",
                 serialization_format: Optional[str] = None,
                 original_error: Optional[Exception] = None,
                 **kwargs):
        super().__init__(message, error_code="SERIALIZATION_ERROR", **kwargs)
        self.serialization_format = serialization_format
        self.original_error = original_error
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "serialization_format": self.serialization_format,
            "original_error": str(self.original_error) if self.original_error else None
        })
        return data


class RoutingError(MessageProtocolError):
    """Exception for message routing errors"""
    
    def __init__(self, message: str = "Message routing failed",
                 destination: Optional[str] = None,
                 routing_table: Optional[Dict[str, Any]] = None,
                 **kwargs):
        super().__init__(message, error_code="ROUTING_ERROR", **kwargs)
        self.destination = destination
        self.routing_table = routing_table
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "destination": self.destination,
            "routing_table": self.routing_table
        })
        return data


class HandlerNotFoundError(MessageProtocolError):
    """Exception when no handler is found for a message"""
    
    def __init__(self, message: str = "No handler found for message",
                 message_type: Optional[str] = None,
                 available_handlers: Optional[list] = None,
                 **kwargs):
        super().__init__(message, error_code="HANDLER_NOT_FOUND", **kwargs)
        self.message_type = message_type
        self.available_handlers = available_handlers or []
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "message_type": self.message_type,
            "available_handlers": self.available_handlers
        })
        return data


class TimeoutError(MessageProtocolError):
    """Exception for message timeout errors"""
    
    def __init__(self, message: str = "Message timeout",
                 timeout_duration: Optional[float] = None,
                 message_id: Optional[str] = None,
                 **kwargs):
        super().__init__(message, error_code="TIMEOUT", **kwargs)
        self.timeout_duration = timeout_duration
        self.message_id = message_id
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "timeout_duration": self.timeout_duration,
            "message_id": self.message_id
        })
        return data


class TransportError(MessageProtocolError):
    """Exception for message transport errors"""
    
    def __init__(self, message: str = "Message transport error",
                 transport_type: Optional[str] = None,
                 connection_info: Optional[Dict[str, Any]] = None,
                 **kwargs):
        super().__init__(message, error_code="TRANSPORT_ERROR", **kwargs)
        self.transport_type = transport_type
        self.connection_info = connection_info
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "transport_type": self.transport_type,
            "connection_info": self.connection_info
        })
        return data


class DuplicateMessageError(MessageProtocolError):
    """Exception for duplicate message handling"""
    
    def __init__(self, message: str = "Duplicate message detected",
                 original_message_id: Optional[str] = None,
                 duplicate_message_id: Optional[str] = None,
                 **kwargs):
        super().__init__(message, error_code="DUPLICATE_MESSAGE", **kwargs)
        self.original_message_id = original_message_id
        self.duplicate_message_id = duplicate_message_id
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "original_message_id": self.original_message_id,
            "duplicate_message_id": self.duplicate_message_id
        })
        return data


class MessageBusError(MessageProtocolError):
    """Exception for message bus errors"""
    
    def __init__(self, message: str = "Message bus error",
                 bus_state: Optional[str] = None,
                 operation: Optional[str] = None,
                 **kwargs):
        super().__init__(message, error_code="MESSAGE_BUS_ERROR", **kwargs)
        self.bus_state = bus_state
        self.operation = operation
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "bus_state": self.bus_state,
            "operation": self.operation
        })
        return data


class SecurityError(MessageProtocolError):
    """Exception for message security errors"""
    
    def __init__(self, message: str = "Message security violation",
                 security_policy: Optional[str] = None,
                 violation_type: Optional[str] = None,
                 **kwargs):
        super().__init__(message, error_code="SECURITY_ERROR", **kwargs)
        self.security_policy = security_policy
        self.violation_type = violation_type
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "security_policy": self.security_policy,
            "violation_type": self.violation_type
        })
        return data


class ConfigurationError(MessageProtocolError):
    """Exception for message system configuration errors"""
    
    def __init__(self, message: str = "Message system configuration error",
                 config_section: Optional[str] = None,
                 invalid_values: Optional[Dict[str, Any]] = None,
                 **kwargs):
        super().__init__(message, error_code="CONFIGURATION_ERROR", **kwargs)
        self.config_section = config_section
        self.invalid_values = invalid_values
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "config_section": self.config_section,
            "invalid_values": self.invalid_values
        })
        return data