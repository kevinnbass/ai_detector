"""
Unified API Client System
Centralized API communication with queuing and retry mechanisms
"""

from .unified_client import UnifiedAPIClient, APIClientConfig
from .queue_manager import QueueManager, RequestQueue, QueuedRequest
from .retry_handler import RetryHandler, RetryConfig, RetryStrategy
from .rate_limiter import RateLimiter, RateLimitConfig
from .response_cache import ResponseCache, CacheConfig
from .middleware import APIMiddleware, AuthenticationMiddleware, LoggingMiddleware
from .exceptions import APIClientError, RateLimitError, TimeoutError, AuthenticationError

__all__ = [
    'UnifiedAPIClient',
    'APIClientConfig',
    'QueueManager',
    'RequestQueue', 
    'QueuedRequest',
    'RetryHandler',
    'RetryConfig',
    'RetryStrategy',
    'RateLimiter',
    'RateLimitConfig',
    'ResponseCache',
    'CacheConfig',
    'APIMiddleware',
    'AuthenticationMiddleware',
    'LoggingMiddleware',
    'APIClientError',
    'RateLimitError',
    'TimeoutError',
    'AuthenticationError'
]