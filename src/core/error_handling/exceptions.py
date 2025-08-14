"""
Custom exceptions for the AI Detector system.

Provides a hierarchy of exceptions for different error categories,
enabling precise error handling and recovery strategies.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
import traceback


class AIDetectorException(Exception):
    """
    Base exception for all AI Detector errors.
    
    Provides structured error information including error codes,
    context, and recovery suggestions.
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        recovery_suggestion: Optional[str] = None,
        inner_exception: Optional[Exception] = None
    ):
        """
        Initialize the base exception.
        
        Args:
            message: Human-readable error message
            error_code: Unique error code for identification
            details: Additional error details
            recovery_suggestion: Suggested recovery action
            inner_exception: Original exception if wrapping
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.recovery_suggestion = recovery_suggestion
        self.inner_exception = inner_exception
        self.timestamp = datetime.utcnow()
        self.traceback = traceback.format_exc() if inner_exception else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary format."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "recovery_suggestion": self.recovery_suggestion,
            "timestamp": self.timestamp.isoformat(),
            "traceback": self.traceback
        }


class ValidationError(AIDetectorException):
    """
    Raised when data validation fails.
    
    Used for schema validation, input validation, and format checking.
    """
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        expected_format: Optional[str] = None,
        actual_value: Any = None,
        **kwargs
    ):
        """
        Initialize validation error.
        
        Args:
            message: Error message
            field: Field that failed validation
            expected_format: Expected data format
            actual_value: Actual value that failed
        """
        details = {
            "field": field,
            "expected_format": expected_format,
            "actual_value": str(actual_value)[:100] if actual_value else None
        }
        
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details={**details, **kwargs.get("details", {})},
            recovery_suggestion=kwargs.get(
                "recovery_suggestion",
                f"Please ensure {field or 'the data'} meets the expected format"
            )
        )


class DetectionError(AIDetectorException):
    """
    Raised when AI detection fails.
    
    Covers pattern matching failures, ML model errors, and LLM issues.
    """
    
    def __init__(
        self,
        message: str,
        detection_method: Optional[str] = None,
        input_text: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize detection error.
        
        Args:
            message: Error message
            detection_method: Method that failed (pattern/ml/llm)
            input_text: Text that caused the error
        """
        details = {
            "detection_method": detection_method,
            "input_text_preview": input_text[:200] if input_text else None,
            "input_text_length": len(input_text) if input_text else 0
        }
        
        super().__init__(
            message=message,
            error_code="DETECTION_ERROR",
            details={**details, **kwargs.get("details", {})},
            recovery_suggestion=kwargs.get(
                "recovery_suggestion",
                "Try using a different detection method or simplify the input"
            )
        )


class APIError(AIDetectorException):
    """
    Raised for API-related errors.
    
    Includes HTTP errors, network issues, and API validation failures.
    """
    
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        endpoint: Optional[str] = None,
        request_data: Optional[Dict[str, Any]] = None,
        response_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize API error.
        
        Args:
            message: Error message
            status_code: HTTP status code
            endpoint: API endpoint that failed
            request_data: Request that caused the error
            response_data: Response from the API
        """
        details = {
            "status_code": status_code,
            "endpoint": endpoint,
            "request_preview": str(request_data)[:200] if request_data else None,
            "response": response_data
        }
        
        # Determine recovery suggestion based on status code
        if status_code == 429:
            recovery = "Rate limit exceeded. Please wait before retrying."
        elif status_code == 401:
            recovery = "Authentication failed. Check API credentials."
        elif status_code == 503:
            recovery = "Service temporarily unavailable. Try again later."
        elif status_code and 500 <= status_code < 600:
            recovery = "Server error. Contact support if the issue persists."
        else:
            recovery = kwargs.get("recovery_suggestion", "Check API endpoint and request format")
        
        super().__init__(
            message=message,
            error_code=f"API_ERROR_{status_code}" if status_code else "API_ERROR",
            details={**details, **kwargs.get("details", {})},
            recovery_suggestion=recovery
        )


class ConfigurationError(AIDetectorException):
    """
    Raised when configuration is invalid or missing.
    
    Used for settings validation and environment setup issues.
    """
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_file: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize configuration error.
        
        Args:
            message: Error message
            config_key: Configuration key that's problematic
            config_file: Configuration file path
        """
        details = {
            "config_key": config_key,
            "config_file": config_file
        }
        
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            details={**details, **kwargs.get("details", {})},
            recovery_suggestion=kwargs.get(
                "recovery_suggestion",
                f"Check configuration for {config_key or 'required settings'}"
            )
        )


class ServiceError(AIDetectorException):
    """
    Raised when a service fails or is unavailable.
    
    Covers database, cache, and external service failures.
    """
    
    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize service error.
        
        Args:
            message: Error message
            service_name: Name of the failing service
            operation: Operation that failed
        """
        details = {
            "service_name": service_name,
            "operation": operation
        }
        
        super().__init__(
            message=message,
            error_code="SERVICE_ERROR",
            details={**details, **kwargs.get("details", {})},
            recovery_suggestion=kwargs.get(
                "recovery_suggestion",
                f"Check if {service_name or 'the service'} is running and accessible"
            )
        )


class IntegrationError(AIDetectorException):
    """
    Raised when component integration fails.
    
    Used for inter-component communication and data flow issues.
    """
    
    def __init__(
        self,
        message: str,
        source_component: Optional[str] = None,
        target_component: Optional[str] = None,
        integration_point: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize integration error.
        
        Args:
            message: Error message
            source_component: Component initiating the integration
            target_component: Component being integrated with
            integration_point: Specific integration point that failed
        """
        details = {
            "source_component": source_component,
            "target_component": target_component,
            "integration_point": integration_point
        }
        
        super().__init__(
            message=message,
            error_code="INTEGRATION_ERROR",
            details={**details, **kwargs.get("details", {})},
            recovery_suggestion=kwargs.get(
                "recovery_suggestion",
                f"Check integration between {source_component} and {target_component}"
            )
        )


class PerformanceError(AIDetectorException):
    """
    Raised when performance thresholds are exceeded.
    
    Used for timeout, memory, and throughput issues.
    """
    
    def __init__(
        self,
        message: str,
        metric_name: Optional[str] = None,
        actual_value: Optional[float] = None,
        threshold_value: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize performance error.
        
        Args:
            message: Error message
            metric_name: Performance metric that failed
            actual_value: Actual metric value
            threshold_value: Expected threshold
        """
        details = {
            "metric_name": metric_name,
            "actual_value": actual_value,
            "threshold_value": threshold_value,
            "exceeded_by": actual_value - threshold_value if actual_value and threshold_value else None
        }
        
        super().__init__(
            message=message,
            error_code="PERFORMANCE_ERROR",
            details={**details, **kwargs.get("details", {})},
            recovery_suggestion=kwargs.get(
                "recovery_suggestion",
                "Consider optimizing the operation or adjusting performance thresholds"
            )
        )


class SecurityError(AIDetectorException):
    """
    Raised for security-related issues.
    
    Used for authentication, authorization, and data security errors.
    """
    
    def __init__(
        self,
        message: str,
        security_type: Optional[str] = None,
        user_context: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize security error.
        
        Args:
            message: Error message
            security_type: Type of security issue
            user_context: User or context information (sanitized)
        """
        # Be careful not to expose sensitive information
        details = {
            "security_type": security_type,
            "user_context": user_context  # Should be sanitized
        }
        
        super().__init__(
            message=message,
            error_code="SECURITY_ERROR",
            details={**details, **kwargs.get("details", {})},
            recovery_suggestion=kwargs.get(
                "recovery_suggestion",
                "Verify credentials and permissions"
            )
        )


class ResourceError(AIDetectorException):
    """
    Raised when resource limits are exceeded.
    
    Used for quota, rate limiting, and resource exhaustion.
    """
    
    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        limit: Optional[int] = None,
        current_usage: Optional[int] = None,
        reset_time: Optional[datetime] = None,
        **kwargs
    ):
        """
        Initialize resource error.
        
        Args:
            message: Error message
            resource_type: Type of resource exhausted
            limit: Resource limit
            current_usage: Current usage level
            reset_time: When the resource resets
        """
        details = {
            "resource_type": resource_type,
            "limit": limit,
            "current_usage": current_usage,
            "reset_time": reset_time.isoformat() if reset_time else None
        }
        
        super().__init__(
            message=message,
            error_code="RESOURCE_ERROR",
            details={**details, **kwargs.get("details", {})},
            recovery_suggestion=kwargs.get(
                "recovery_suggestion",
                f"Wait until {reset_time} or upgrade your plan" if reset_time else "Reduce resource usage"
            )
        )


class TimeoutError(PerformanceError):
    """
    Raised when an operation times out.
    
    Specialized performance error for timeout scenarios.
    """
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize timeout error.
        
        Args:
            message: Error message
            operation: Operation that timed out
            timeout_ms: Timeout duration in milliseconds
        """
        super().__init__(
            message=message,
            metric_name="timeout",
            actual_value=timeout_ms,
            threshold_value=timeout_ms,
            **kwargs
        )
        self.error_code = "TIMEOUT_ERROR"
        self.details["operation"] = operation


class RetryableError(AIDetectorException):
    """
    Base class for errors that can be retried.
    
    Provides retry metadata and strategies.
    """
    
    def __init__(
        self,
        message: str,
        max_retries: int = 3,
        retry_after: Optional[int] = None,
        backoff_strategy: str = "exponential",
        **kwargs
    ):
        """
        Initialize retryable error.
        
        Args:
            message: Error message
            max_retries: Maximum retry attempts
            retry_after: Seconds to wait before retry
            backoff_strategy: Retry backoff strategy
        """
        super().__init__(message, **kwargs)
        self.max_retries = max_retries
        self.retry_after = retry_after
        self.backoff_strategy = backoff_strategy
        self.details.update({
            "retryable": True,
            "max_retries": max_retries,
            "retry_after": retry_after,
            "backoff_strategy": backoff_strategy
        })