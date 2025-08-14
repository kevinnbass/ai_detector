"""
Unified API Client
Central client for all API communications with advanced features
"""

import asyncio
import aiohttp
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime
from urllib.parse import urljoin
import logging

from ..interfaces.api_interfaces import IAPIClient, APIRequest, APIResponse, HTTPMethod, ResponseFormat
from .queue_manager import QueueManager, Priority
from .retry_handler import RetryHandler, RetryConfig, create_retry_handler
from .rate_limiter import RateLimiter, RateLimitConfig
from .response_cache import ResponseCache, CacheConfig
from .middleware import APIMiddleware, AuthenticationMiddleware, LoggingMiddleware
from .exceptions import (
    APIClientError, NetworkError, TimeoutError, AuthenticationError,
    create_error_from_response
)

logger = logging.getLogger(__name__)


@dataclass
class APIClientConfig:
    """Configuration for the unified API client"""
    
    # Basic settings
    base_url: str = ""
    timeout: float = 30.0
    max_connections: int = 100
    max_connections_per_host: int = 10
    
    # Authentication
    auth_type: str = "none"  # none, bearer, api_key, basic
    auth_credentials: Dict[str, str] = field(default_factory=dict)
    
    # Retry configuration
    enable_retries: bool = True
    retry_config: Optional[RetryConfig] = None
    adaptive_retries: bool = False
    
    # Rate limiting
    enable_rate_limiting: bool = True
    rate_limit_config: Optional[RateLimitConfig] = None
    
    # Caching
    enable_caching: bool = True
    cache_config: Optional[CacheConfig] = None
    
    # Queue management
    enable_queuing: bool = True
    max_queue_size: int = 1000
    max_concurrent_requests: int = 20
    
    # Middleware
    enable_logging_middleware: bool = True
    custom_middleware: List[APIMiddleware] = field(default_factory=list)
    
    # Headers
    default_headers: Dict[str, str] = field(default_factory=lambda: {
        "User-Agent": "AI-Detector-Client/1.0",
        "Accept": "application/json",
        "Content-Type": "application/json"
    })
    
    # SSL/TLS
    verify_ssl: bool = True
    client_cert: Optional[str] = None
    
    def __post_init__(self):
        if self.retry_config is None:
            self.retry_config = RetryConfig()
        
        if self.rate_limit_config is None:
            self.rate_limit_config = RateLimitConfig()
        
        if self.cache_config is None:
            self.cache_config = CacheConfig()


class UnifiedAPIClient(IAPIClient):
    """Unified API client with advanced features"""
    
    def __init__(self, config: APIClientConfig):
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._initialized = False
        
        # Components
        self.queue_manager: Optional[QueueManager] = None
        self.retry_handler: Optional[RetryHandler] = None
        self.rate_limiter: Optional[RateLimiter] = None
        self.response_cache: Optional[ResponseCache] = None
        self.middleware: List[APIMiddleware] = []
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cached_responses": 0,
            "rate_limited_requests": 0,
            "retried_requests": 0,
            "total_response_time": 0.0
        }
        
        # Request tracking
        self._active_requests: Dict[str, asyncio.Task] = {}
        self._request_history: List[Dict[str, Any]] = []
    
    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Initialize the API client"""
        if self._initialized:
            return True
        
        try:
            logger.info("Initializing unified API client...")
            
            # Create HTTP session
            connector = aiohttp.TCPConnector(
                limit=self.config.max_connections,
                limit_per_host=self.config.max_connections_per_host,
                verify_ssl=self.config.verify_ssl
            )
            
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=self.config.default_headers
            )
            
            # Initialize components
            if self.config.enable_queuing:
                self.queue_manager = QueueManager(
                    max_concurrent=self.config.max_concurrent_requests,
                    max_queue_size=self.config.max_queue_size
                )
                await self.queue_manager.start()
            
            if self.config.enable_retries:
                self.retry_handler = create_retry_handler(
                    strategy=self.config.retry_config.strategy.value,
                    max_retries=self.config.retry_config.max_retries,
                    adaptive=self.config.adaptive_retries
                )
            
            if self.config.enable_rate_limiting:
                self.rate_limiter = RateLimiter(self.config.rate_limit_config)
            
            if self.config.enable_caching:
                self.response_cache = ResponseCache(self.config.cache_config)
                await self.response_cache.initialize()
            
            # Setup middleware
            await self._setup_middleware()
            
            self._initialized = True
            logger.info("Unified API client initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize API client: {e}")
            return False
    
    def is_initialized(self) -> bool:
        """Check if client is initialized"""
        return self._initialized
    
    def configure(self, config: Dict[str, Any]) -> bool:
        """Configure the client"""
        try:
            for key, value in config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                else:
                    logger.warning(f"Unknown configuration key: {key}")
            return True
        except Exception as e:
            logger.error(f"Configuration failed: {e}")
            return False
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get current configuration"""
        return {
            "base_url": self.config.base_url,
            "timeout": self.config.timeout,
            "enable_retries": self.config.enable_retries,
            "enable_rate_limiting": self.config.enable_rate_limiting,
            "enable_caching": self.config.enable_caching,
            "enable_queuing": self.config.enable_queuing
        }
    
    def validate_configuration(self, config: Dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate configuration"""
        errors = []
        
        if "base_url" in config and not config["base_url"]:
            errors.append("base_url cannot be empty")
        
        if "timeout" in config and config["timeout"] <= 0:
            errors.append("timeout must be positive")
        
        if "max_connections" in config and config["max_connections"] <= 0:
            errors.append("max_connections must be positive")
        
        return len(errors) == 0, errors
    
    async def request(self, method: HTTPMethod, endpoint: str,
                     data: Optional[Any] = None,
                     headers: Optional[Dict[str, str]] = None,
                     params: Optional[Dict[str, Any]] = None,
                     priority: Priority = Priority.NORMAL,
                     use_queue: bool = True,
                     use_cache: bool = True) -> APIResponse:
        """Make API request"""
        
        if not self._initialized:
            raise APIClientError("Client not initialized")
        
        request_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        try:
            self.stats["total_requests"] += 1
            
            # Build full URL
            url = urljoin(self.config.base_url, endpoint) if self.config.base_url else endpoint
            
            # Check cache first (for GET requests)
            if use_cache and self.response_cache and method == HTTPMethod.GET:
                cached_response = await self.response_cache.get(url, params)
                if cached_response:
                    self.stats["cached_responses"] += 1
                    logger.debug(f"Cache hit for {method.value} {url}")
                    return cached_response
            
            # Create API request object
            api_request = APIRequest(
                method=method,
                path=url,
                headers=headers or {},
                params=params,
                body=data
            )
            
            # Process through middleware
            for middleware in self.middleware:
                api_request = await middleware.process_request(api_request)
                if api_request is None:
                    raise APIClientError("Request blocked by middleware")
            
            # Execute request (with or without queue)
            if use_queue and self.queue_manager:
                response = await self._execute_queued_request(
                    api_request, priority, request_id
                )
            else:
                response = await self._execute_direct_request(api_request, request_id)
            
            # Process response through middleware
            for middleware in reversed(self.middleware):
                response = await middleware.process_response(response, api_request)
                if response is None:
                    raise APIClientError("Response blocked by middleware")
            
            # Cache successful responses (for GET requests)
            if (use_cache and self.response_cache and 
                method == HTTPMethod.GET and 200 <= response.status_code < 300):
                await self.response_cache.set(url, params, response)
            
            self.stats["successful_requests"] += 1
            return response
            
        except Exception as e:
            self.stats["failed_requests"] += 1
            logger.error(f"Request {request_id} failed: {e}")
            raise
        
        finally:
            # Update statistics
            execution_time = (datetime.now() - start_time).total_seconds()
            self.stats["total_response_time"] += execution_time
            
            # Record request history
            self._record_request_history(
                request_id, method, endpoint, execution_time, 
                self.stats["successful_requests"] > 0
            )
    
    async def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> APIResponse:
        """Make GET request"""
        return await self.request(HTTPMethod.GET, endpoint, params=params)
    
    async def post(self, endpoint: str, data: Any, 
                   headers: Optional[Dict[str, str]] = None) -> APIResponse:
        """Make POST request"""
        return await self.request(HTTPMethod.POST, endpoint, data=data, headers=headers)
    
    async def put(self, endpoint: str, data: Any,
                  headers: Optional[Dict[str, str]] = None) -> APIResponse:
        """Make PUT request"""
        return await self.request(HTTPMethod.PUT, endpoint, data=data, headers=headers)
    
    async def delete(self, endpoint: str) -> APIResponse:
        """Make DELETE request"""
        return await self.request(HTTPMethod.DELETE, endpoint)
    
    def set_base_url(self, url: str) -> None:
        """Set base URL"""
        self.config.base_url = url
        logger.info(f"Base URL set to: {url}")
    
    def set_authentication(self, auth_type: str, credentials: Dict[str, str]) -> None:
        """Set authentication credentials"""
        self.config.auth_type = auth_type
        self.config.auth_credentials = credentials
        
        # Update authentication middleware if present
        for middleware in self.middleware:
            if isinstance(middleware, AuthenticationMiddleware):
                middleware.update_credentials(auth_type, credentials)
        
        logger.info(f"Authentication set to: {auth_type}")
    
    def get_request_stats(self) -> Dict[str, Any]:
        """Get request statistics"""
        avg_response_time = 0.0
        if self.stats["total_requests"] > 0:
            avg_response_time = self.stats["total_response_time"] / self.stats["total_requests"]
        
        stats = {
            **self.stats,
            "average_response_time": avg_response_time,
            "success_rate": (self.stats["successful_requests"] / max(1, self.stats["total_requests"])),
            "active_requests": len(self._active_requests),
            "request_history_size": len(self._request_history)
        }
        
        # Add component stats
        if self.queue_manager:
            stats["queue_stats"] = asyncio.create_task(self.queue_manager.get_stats())
        
        if self.rate_limiter:
            stats["rate_limit_stats"] = self.rate_limiter.get_stats()
        
        if self.response_cache:
            stats["cache_stats"] = self.response_cache.get_stats()
        
        if hasattr(self.retry_handler, "get_adaptation_stats"):
            stats["retry_stats"] = self.retry_handler.get_adaptation_stats()
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        health_status = {
            "initialized": self._initialized,
            "session_active": self._session and not self._session.closed,
            "components": {}
        }
        
        if self.queue_manager:
            health_status["components"]["queue_manager"] = self.queue_manager._running
        
        if self.response_cache:
            health_status["components"]["cache"] = True  # Could check cache connectivity
        
        if self.rate_limiter:
            health_status["components"]["rate_limiter"] = True
        
        # Overall health
        health_status["healthy"] = all([
            health_status["initialized"],
            health_status["session_active"],
            all(health_status["components"].values())
        ])
        
        return health_status
    
    async def _setup_middleware(self) -> None:
        """Setup middleware chain"""
        # Add authentication middleware
        if self.config.auth_type != "none":
            auth_middleware = AuthenticationMiddleware(
                self.config.auth_type,
                self.config.auth_credentials
            )
            self.middleware.append(auth_middleware)
        
        # Add logging middleware
        if self.config.enable_logging_middleware:
            logging_middleware = LoggingMiddleware()
            self.middleware.append(logging_middleware)
        
        # Add custom middleware
        self.middleware.extend(self.config.custom_middleware)
        
        logger.debug(f"Initialized {len(self.middleware)} middleware components")
    
    async def _execute_direct_request(self, request: APIRequest, request_id: str) -> APIResponse:
        """Execute request directly without queuing"""
        
        # Apply rate limiting
        if self.rate_limiter:
            await self.rate_limiter.acquire(request.path)
        
        # Execute with retry logic if enabled
        if self.retry_handler:
            return await self.retry_handler.execute_with_retry(
                self._make_http_request, request, request_id
            )
        else:
            return await self._make_http_request(request, request_id)
    
    async def _execute_queued_request(self, request: APIRequest, 
                                    priority: Priority, request_id: str) -> APIResponse:
        """Execute request through queue"""
        
        if not self.queue_manager:
            raise APIClientError("Queue manager not initialized")
        
        # Enqueue the request
        queued_request_id = await self.queue_manager.enqueue_request(
            method=request.method.value,
            url=request.path,
            data=request.body,
            headers=request.headers,
            params=request.params,
            priority=priority
        )
        
        # Wait for completion (this would be handled differently in practice)
        # For now, we'll execute directly
        return await self._execute_direct_request(request, request_id)
    
    async def _make_http_request(self, request: APIRequest, request_id: str) -> APIResponse:
        """Make actual HTTP request"""
        
        if not self._session:
            raise APIClientError("HTTP session not initialized")
        
        try:
            # Track active request
            request_task = asyncio.current_task()
            self._active_requests[request_id] = request_task
            
            # Prepare request data
            json_data = None
            if request.body and isinstance(request.body, dict):
                json_data = request.body
            
            # Make HTTP request
            async with self._session.request(
                method=request.method.value,
                url=request.path,
                json=json_data,
                headers=request.headers,
                params=request.params
            ) as response:
                
                # Read response
                try:
                    response_data = await response.json()
                except:
                    response_data = await response.text()
                
                # Create API response
                api_response = APIResponse(
                    status_code=response.status,
                    headers=dict(response.headers),
                    body=response_data,
                    processing_time=(datetime.now() - datetime.now()).total_seconds()
                )
                
                # Check for HTTP errors
                if response.status >= 400:
                    error = create_error_from_response(response, request_id)
                    logger.warning(f"HTTP error {response.status} for request {request_id}")
                    raise error
                
                return api_response
                
        except aiohttp.ClientTimeout:
            raise TimeoutError(f"Request {request_id} timed out")
        
        except aiohttp.ClientConnectionError as e:
            raise NetworkError(f"Connection error for request {request_id}", connection_error=e)
        
        except aiohttp.ClientError as e:
            raise APIClientError(f"HTTP client error for request {request_id}: {e}")
        
        finally:
            # Remove from active requests
            self._active_requests.pop(request_id, None)
    
    def _record_request_history(self, request_id: str, method: HTTPMethod, 
                               endpoint: str, execution_time: float, success: bool) -> None:
        """Record request in history for analysis"""
        
        history_entry = {
            "request_id": request_id,
            "method": method.value,
            "endpoint": endpoint,
            "execution_time": execution_time,
            "success": success,
            "timestamp": datetime.now().isoformat()
        }
        
        self._request_history.append(history_entry)
        
        # Keep only recent history
        max_history = 1000
        if len(self._request_history) > max_history:
            self._request_history = self._request_history[-max_history:]
    
    async def close(self) -> None:
        """Close the API client and cleanup resources"""
        logger.info("Closing unified API client...")
        
        # Stop queue manager
        if self.queue_manager:
            await self.queue_manager.stop()
        
        # Close response cache
        if self.response_cache:
            await self.response_cache.close()
        
        # Close HTTP session
        if self._session and not self._session.closed:
            await self._session.close()
        
        # Cancel active requests
        for request_task in self._active_requests.values():
            if not request_task.done():
                request_task.cancel()
        
        # Wait for active requests to complete
        if self._active_requests:
            await asyncio.gather(*self._active_requests.values(), return_exceptions=True)
        
        self._initialized = False
        logger.info("Unified API client closed")