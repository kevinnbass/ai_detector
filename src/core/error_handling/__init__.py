"""
Comprehensive error handling system for AI Detector.

This package provides standardized error handling, recovery mechanisms,
and boundary error management across all system components.
"""

from .exceptions import *
from .handlers import ErrorHandler, BoundaryErrorHandler
from .recovery import RecoveryManager
from .context import ErrorContext
from .decorators import handle_errors, retry_on_failure, circuit_breaker

__all__ = [
    # Exceptions
    'AIDetectorException',
    'ValidationError', 
    'DetectionError',
    'APIError',
    'ConfigurationError',
    'ServiceError',
    'IntegrationError',
    'PerformanceError',
    'SecurityError',
    
    # Error Handlers
    'ErrorHandler',
    'BoundaryErrorHandler',
    
    # Recovery
    'RecoveryManager',
    
    # Context
    'ErrorContext',
    
    # Decorators
    'handle_errors',
    'retry_on_failure', 
    'circuit_breaker'
]