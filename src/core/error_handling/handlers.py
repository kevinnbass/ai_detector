"""
Error handlers for managing exceptions at system boundaries.

Provides centralized error handling, logging, and recovery mechanisms
for all components in the AI Detector system.
"""

import logging
import json
from typing import Optional, Dict, Any, Callable, Type, List, Union
from datetime import datetime
from functools import wraps
import traceback

from .exceptions import (
    AIDetectorException,
    ValidationError,
    APIError,
    RetryableError,
    TimeoutError
)
from .context import ErrorContext
from .recovery import RecoveryManager


logger = logging.getLogger(__name__)


class ErrorHandler:
    """
    Central error handler for the AI Detector system.
    
    Manages error logging, transformation, and recovery strategies
    across all system components.
    """
    
    def __init__(
        self,
        recovery_manager: Optional['RecoveryManager'] = None,
        enable_detailed_logging: bool = True,
        enable_error_aggregation: bool = True
    ):
        """
        Initialize the error handler.
        
        Args:
            recovery_manager: Manager for error recovery strategies
            enable_detailed_logging: Whether to log detailed error info
            enable_error_aggregation: Whether to aggregate similar errors
        """
        self.recovery_manager = recovery_manager or RecoveryManager()
        self.enable_detailed_logging = enable_detailed_logging
        self.enable_error_aggregation = enable_error_aggregation
        self.error_history: List[Dict[str, Any]] = []
        self.error_counts: Dict[str, int] = {}
        
        # Error transformation mappings
        self.error_transformers: Dict[Type[Exception], Callable] = {}
        
        # Register default transformers
        self._register_default_transformers()
    
    def _register_default_transformers(self):
        """Register default error transformers."""
        self.register_transformer(ValueError, self._transform_value_error)
        self.register_transformer(KeyError, self._transform_key_error)
        self.register_transformer(TypeError, self._transform_type_error)
        self.register_transformer(IOError, self._transform_io_error)
        self.register_transformer(ConnectionError, self._transform_connection_error)
    
    def register_transformer(
        self,
        exception_type: Type[Exception],
        transformer: Callable[[Exception], AIDetectorException]
    ):
        """
        Register a custom error transformer.
        
        Args:
            exception_type: Type of exception to transform
            transformer: Function to transform the exception
        """
        self.error_transformers[exception_type] = transformer
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[ErrorContext] = None,
        reraise: bool = True,
        attempt_recovery: bool = True
    ) -> Optional[Any]:
        """
        Handle an error with logging and optional recovery.
        
        Args:
            error: The exception to handle
            context: Additional error context
            reraise: Whether to reraise the error after handling
            attempt_recovery: Whether to attempt recovery
            
        Returns:
            Recovery result if successful, None otherwise
            
        Raises:
            The original or transformed error if reraise=True
        """
        # Transform standard exceptions to AIDetectorException
        if not isinstance(error, AIDetectorException):
            error = self._transform_error(error)
        
        # Add context to error
        if context:
            error.details.update(context.to_dict())
        
        # Log the error
        self._log_error(error)
        
        # Track error statistics
        self._track_error(error)
        
        # Attempt recovery if requested
        recovery_result = None
        if attempt_recovery and self.recovery_manager:
            recovery_result = self.recovery_manager.attempt_recovery(error, context)
            
            if recovery_result is not None:
                logger.info(f"Successfully recovered from {error.error_code}")
                return recovery_result
        
        # Reraise if requested
        if reraise:
            raise error
        
        return recovery_result
    
    def _transform_error(self, error: Exception) -> AIDetectorException:
        """
        Transform a standard exception to AIDetectorException.
        
        Args:
            error: Standard exception to transform
            
        Returns:
            Transformed AIDetectorException
        """
        # Check for registered transformer
        for exc_type, transformer in self.error_transformers.items():
            if isinstance(error, exc_type):
                return transformer(error)
        
        # Default transformation
        return AIDetectorException(
            message=str(error),
            error_code=error.__class__.__name__,
            inner_exception=error,
            details={
                "original_type": error.__class__.__name__,
                "original_message": str(error)
            }
        )
    
    def _transform_value_error(self, error: ValueError) -> ValidationError:
        """Transform ValueError to ValidationError."""
        return ValidationError(
            message=str(error),
            recovery_suggestion="Check input values and formats"
        )
    
    def _transform_key_error(self, error: KeyError) -> ValidationError:
        """Transform KeyError to ValidationError."""
        return ValidationError(
            message=f"Missing required field: {error}",
            field=str(error),
            recovery_suggestion=f"Ensure field '{error}' is provided"
        )
    
    def _transform_type_error(self, error: TypeError) -> ValidationError:
        """Transform TypeError to ValidationError."""
        return ValidationError(
            message=str(error),
            recovery_suggestion="Check data types match expected formats"
        )
    
    def _transform_io_error(self, error: IOError) -> AIDetectorException:
        """Transform IOError to appropriate exception."""
        return AIDetectorException(
            message=f"I/O operation failed: {error}",
            error_code="IO_ERROR",
            recovery_suggestion="Check file permissions and paths"
        )
    
    def _transform_connection_error(self, error: ConnectionError) -> APIError:
        """Transform ConnectionError to APIError."""
        return APIError(
            message=f"Connection failed: {error}",
            recovery_suggestion="Check network connectivity and service availability"
        )
    
    def _log_error(self, error: AIDetectorException):
        """
        Log error with appropriate detail level.
        
        Args:
            error: Error to log
        """
        log_message = f"[{error.error_code}] {error.message}"
        
        if self.enable_detailed_logging:
            log_message += f"\nDetails: {json.dumps(error.details, indent=2)}"
            if error.recovery_suggestion:
                log_message += f"\nRecovery: {error.recovery_suggestion}"
            if error.traceback:
                log_message += f"\nTraceback:\n{error.traceback}"
        
        # Log at appropriate level
        if isinstance(error, (ValidationError, RetryableError)):
            logger.warning(log_message)
        elif isinstance(error, (APIError, TimeoutError)):
            logger.error(log_message)
        else:
            logger.error(log_message, exc_info=True)
    
    def _track_error(self, error: AIDetectorException):
        """
        Track error statistics for monitoring.
        
        Args:
            error: Error to track
        """
        # Count errors by code
        self.error_counts[error.error_code] = self.error_counts.get(error.error_code, 0) + 1
        
        # Store error history (with size limit)
        if self.enable_error_aggregation:
            self.error_history.append({
                "timestamp": error.timestamp,
                "error_code": error.error_code,
                "message": error.message[:200],  # Truncate for storage
                "details": error.details
            })
            
            # Keep only last 1000 errors
            if len(self.error_history) > 1000:
                self.error_history = self.error_history[-1000:]
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get error statistics for monitoring.
        
        Returns:
            Dictionary of error statistics
        """
        return {
            "total_errors": sum(self.error_counts.values()),
            "error_counts": self.error_counts.copy(),
            "recent_errors": self.error_history[-10:] if self.error_history else [],
            "most_common_error": max(self.error_counts.items(), key=lambda x: x[1])[0] 
                                 if self.error_counts else None
        }
    
    def clear_statistics(self):
        """Clear error statistics."""
        self.error_history.clear()
        self.error_counts.clear()


class BoundaryErrorHandler(ErrorHandler):
    """
    Specialized error handler for system boundaries.
    
    Provides additional safety and transformation for errors
    crossing component boundaries (API, Extension, Services).
    """
    
    def __init__(
        self,
        boundary_type: str,
        sanitize_errors: bool = True,
        include_stack_traces: bool = False,
        **kwargs
    ):
        """
        Initialize boundary error handler.
        
        Args:
            boundary_type: Type of boundary (api, extension, service)
            sanitize_errors: Whether to sanitize error messages
            include_stack_traces: Whether to include stack traces
        """
        super().__init__(**kwargs)
        self.boundary_type = boundary_type
        self.sanitize_errors = sanitize_errors
        self.include_stack_traces = include_stack_traces
    
    def handle_boundary_error(
        self,
        error: Exception,
        context: Optional[ErrorContext] = None,
        format_for_client: bool = True
    ) -> Dict[str, Any]:
        """
        Handle error at system boundary with appropriate formatting.
        
        Args:
            error: Error to handle
            context: Error context
            format_for_client: Whether to format for client consumption
            
        Returns:
            Formatted error response
        """
        # Handle the error without reraising
        self.handle_error(error, context, reraise=False)
        
        # Transform to AIDetectorException if needed
        if not isinstance(error, AIDetectorException):
            error = self._transform_error(error)
        
        # Format for boundary crossing
        if format_for_client:
            return self._format_for_client(error)
        else:
            return self._format_for_internal(error)
    
    def _format_for_client(self, error: AIDetectorException) -> Dict[str, Any]:
        """
        Format error for client consumption.
        
        Args:
            error: Error to format
            
        Returns:
            Client-safe error dictionary
        """
        response = {
            "error": {
                "code": error.error_code,
                "message": self._sanitize_message(error.message) if self.sanitize_errors else error.message,
                "type": self._get_error_type(error)
            },
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": error.details.get("request_id")
        }
        
        # Add recovery info if available
        if error.recovery_suggestion:
            response["error"]["suggestion"] = error.recovery_suggestion
        
        # Add retry info for retryable errors
        if isinstance(error, RetryableError):
            response["retry_info"] = {
                "retryable": True,
                "max_retries": error.max_retries,
                "retry_after": error.retry_after,
                "backoff_strategy": error.backoff_strategy
            }
        
        # Optionally include sanitized details
        if error.details and not self.sanitize_errors:
            response["error"]["details"] = self._sanitize_details(error.details)
        
        return response
    
    def _format_for_internal(self, error: AIDetectorException) -> Dict[str, Any]:
        """
        Format error for internal system use.
        
        Args:
            error: Error to format
            
        Returns:
            Internal error dictionary with full details
        """
        response = error.to_dict()
        
        # Add boundary information
        response["boundary"] = {
            "type": self.boundary_type,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Include stack trace if configured
        if self.include_stack_traces and not response.get("traceback"):
            response["traceback"] = traceback.format_exc()
        
        return response
    
    def _sanitize_message(self, message: str) -> str:
        """
        Sanitize error message for client consumption.
        
        Args:
            message: Message to sanitize
            
        Returns:
            Sanitized message
        """
        # Remove sensitive patterns
        sensitive_patterns = [
            r"password[=:]\S+",
            r"token[=:]\S+",
            r"api[_-]?key[=:]\S+",
            r"secret[=:]\S+",
            r"\/home\/\S+",
            r"C:\\Users\\\S+"
        ]
        
        import re
        sanitized = message
        for pattern in sensitive_patterns:
            sanitized = re.sub(pattern, "[REDACTED]", sanitized, flags=re.IGNORECASE)
        
        return sanitized
    
    def _sanitize_details(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize error details for client consumption.
        
        Args:
            details: Details to sanitize
            
        Returns:
            Sanitized details
        """
        sensitive_keys = ["password", "token", "api_key", "secret", "credential"]
        
        sanitized = {}
        for key, value in details.items():
            # Check if key contains sensitive information
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, str):
                sanitized[key] = self._sanitize_message(value)
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_details(value)
            else:
                sanitized[key] = value
        
        return sanitized
    
    def _get_error_type(self, error: AIDetectorException) -> str:
        """
        Get client-friendly error type.
        
        Args:
            error: Error to classify
            
        Returns:
            Error type string
        """
        if isinstance(error, ValidationError):
            return "validation_error"
        elif isinstance(error, APIError):
            status_code = error.details.get("status_code", 500)
            if 400 <= status_code < 500:
                return "client_error"
            else:
                return "server_error"
        elif isinstance(error, TimeoutError):
            return "timeout_error"
        elif isinstance(error, RetryableError):
            return "transient_error"
        else:
            return "internal_error"


# Global error handler instances
_global_handler: Optional[ErrorHandler] = None
_api_boundary_handler: Optional[BoundaryErrorHandler] = None
_extension_boundary_handler: Optional[BoundaryErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    global _global_handler
    if _global_handler is None:
        _global_handler = ErrorHandler()
    return _global_handler


def get_api_boundary_handler() -> BoundaryErrorHandler:
    """Get the API boundary error handler."""
    global _api_boundary_handler
    if _api_boundary_handler is None:
        _api_boundary_handler = BoundaryErrorHandler(
            boundary_type="api",
            sanitize_errors=True,
            include_stack_traces=False
        )
    return _api_boundary_handler


def get_extension_boundary_handler() -> BoundaryErrorHandler:
    """Get the extension boundary error handler."""
    global _extension_boundary_handler
    if _extension_boundary_handler is None:
        _extension_boundary_handler = BoundaryErrorHandler(
            boundary_type="extension",
            sanitize_errors=True,
            include_stack_traces=False
        )
    return _extension_boundary_handler