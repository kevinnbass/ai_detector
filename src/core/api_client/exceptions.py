"""
API Client Exception Classes
Comprehensive error handling for API operations
"""

from typing import Dict, Any, Optional
from datetime import datetime


class APIClientError(Exception):
    """Base exception for API client errors"""
    
    def __init__(self, message: str, 
                 status_code: Optional[int] = None,
                 response_data: Optional[Dict[str, Any]] = None,
                 request_id: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response_data = response_data
        self.request_id = request_id
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary"""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "status_code": self.status_code,
            "response_data": self.response_data,
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat()
        }


class RateLimitError(APIClientError):
    """Exception for rate limit exceeded errors"""
    
    def __init__(self, message: str = "Rate limit exceeded",
                 retry_after: Optional[int] = None,
                 limit: Optional[int] = None,
                 remaining: Optional[int] = None,
                 reset_time: Optional[datetime] = None,
                 **kwargs):
        super().__init__(message, status_code=429, **kwargs)
        self.retry_after = retry_after
        self.limit = limit
        self.remaining = remaining
        self.reset_time = reset_time
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "retry_after": self.retry_after,
            "limit": self.limit,
            "remaining": self.remaining,
            "reset_time": self.reset_time.isoformat() if self.reset_time else None
        })
        return data


class TimeoutError(APIClientError):
    """Exception for request timeout errors"""
    
    def __init__(self, message: str = "Request timeout",
                 timeout_duration: Optional[float] = None,
                 **kwargs):
        super().__init__(message, status_code=408, **kwargs)
        self.timeout_duration = timeout_duration
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["timeout_duration"] = self.timeout_duration
        return data


class AuthenticationError(APIClientError):
    """Exception for authentication errors"""
    
    def __init__(self, message: str = "Authentication failed",
                 auth_type: Optional[str] = None,
                 **kwargs):
        super().__init__(message, status_code=401, **kwargs)
        self.auth_type = auth_type
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["auth_type"] = self.auth_type
        return data


class ValidationError(APIClientError):
    """Exception for request validation errors"""
    
    def __init__(self, message: str = "Request validation failed",
                 validation_errors: Optional[list] = None,
                 **kwargs):
        super().__init__(message, status_code=422, **kwargs)
        self.validation_errors = validation_errors or []
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["validation_errors"] = self.validation_errors
        return data


class NetworkError(APIClientError):
    """Exception for network-related errors"""
    
    def __init__(self, message: str = "Network error",
                 connection_error: Optional[Exception] = None,
                 **kwargs):
        super().__init__(message, **kwargs)
        self.connection_error = connection_error
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["connection_error"] = str(self.connection_error) if self.connection_error else None
        return data


class ServerError(APIClientError):
    """Exception for server errors (5xx status codes)"""
    
    def __init__(self, message: str = "Server error",
                 server_message: Optional[str] = None,
                 **kwargs):
        super().__init__(message, **kwargs)
        self.server_message = server_message
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["server_message"] = self.server_message
        return data


class ConfigurationError(APIClientError):
    """Exception for configuration errors"""
    
    def __init__(self, message: str = "Configuration error",
                 config_key: Optional[str] = None,
                 **kwargs):
        super().__init__(message, **kwargs)
        self.config_key = config_key
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["config_key"] = self.config_key
        return data


class QueueFullError(APIClientError):
    """Exception for queue capacity exceeded"""
    
    def __init__(self, message: str = "Request queue is full",
                 queue_size: Optional[int] = None,
                 max_size: Optional[int] = None,
                 **kwargs):
        super().__init__(message, **kwargs)
        self.queue_size = queue_size
        self.max_size = max_size
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "queue_size": self.queue_size,
            "max_size": self.max_size
        })
        return data


class RetryExhaustedError(APIClientError):
    """Exception for retry attempts exhausted"""
    
    def __init__(self, message: str = "Retry attempts exhausted",
                 max_retries: Optional[int] = None,
                 last_error: Optional[Exception] = None,
                 **kwargs):
        super().__init__(message, **kwargs)
        self.max_retries = max_retries
        self.last_error = last_error
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "max_retries": self.max_retries,
            "last_error": str(self.last_error) if self.last_error else None
        })
        return data


def create_error_from_response(response: Any, request_id: Optional[str] = None) -> APIClientError:
    """Create appropriate error from HTTP response"""
    
    if hasattr(response, 'status_code'):
        status_code = response.status_code
        
        try:
            response_data = response.json() if hasattr(response, 'json') else None
        except:
            response_data = response.text if hasattr(response, 'text') else None
        
        # Determine error type based on status code
        if status_code == 400:
            return ValidationError(
                message="Bad Request",
                status_code=status_code,
                response_data=response_data,
                request_id=request_id
            )
        elif status_code == 401:
            return AuthenticationError(
                message="Unauthorized",
                status_code=status_code,
                response_data=response_data,
                request_id=request_id
            )
        elif status_code == 403:
            return AuthenticationError(
                message="Forbidden",
                status_code=status_code,
                response_data=response_data,
                request_id=request_id
            )
        elif status_code == 404:
            return APIClientError(
                message="Not Found",
                status_code=status_code,
                response_data=response_data,
                request_id=request_id
            )
        elif status_code == 408:
            return TimeoutError(
                message="Request Timeout",
                status_code=status_code,
                response_data=response_data,
                request_id=request_id
            )
        elif status_code == 422:
            validation_errors = []
            if response_data and isinstance(response_data, dict):
                validation_errors = response_data.get("errors", [])
            
            return ValidationError(
                message="Validation Error",
                status_code=status_code,
                response_data=response_data,
                validation_errors=validation_errors,
                request_id=request_id
            )
        elif status_code == 429:
            retry_after = None
            if hasattr(response, 'headers'):
                retry_after = response.headers.get('Retry-After')
                if retry_after:
                    try:
                        retry_after = int(retry_after)
                    except ValueError:
                        retry_after = None
            
            return RateLimitError(
                message="Rate Limit Exceeded",
                status_code=status_code,
                response_data=response_data,
                retry_after=retry_after,
                request_id=request_id
            )
        elif 500 <= status_code < 600:
            server_message = None
            if response_data and isinstance(response_data, dict):
                server_message = response_data.get("message") or response_data.get("error")
            
            return ServerError(
                message=f"Server Error ({status_code})",
                status_code=status_code,
                response_data=response_data,
                server_message=server_message,
                request_id=request_id
            )
        else:
            return APIClientError(
                message=f"HTTP Error ({status_code})",
                status_code=status_code,
                response_data=response_data,
                request_id=request_id
            )
    
    # Generic error for responses without status codes
    return APIClientError(
        message="Unknown API error",
        request_id=request_id
    )