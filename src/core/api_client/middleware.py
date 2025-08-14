"""
API Client Middleware
Middleware components for request/response processing
"""

import time
import json
import base64
from typing import Optional, Dict, Any
from datetime import datetime
import logging

from ..interfaces.api_interfaces import APIRequest, APIResponse, IMiddleware
from .exceptions import AuthenticationError, APIClientError

logger = logging.getLogger(__name__)


class APIMiddleware(IMiddleware):
    """Base class for API middleware"""
    
    def __init__(self, name: str, priority: int = 0):
        self._name = name
        self._priority = priority
        self._enabled = True
    
    def get_middleware_name(self) -> str:
        """Get middleware name"""
        return self._name
    
    def get_priority(self) -> int:
        """Get middleware priority"""
        return self._priority
    
    def is_enabled(self) -> bool:
        """Check if middleware is enabled"""
        return self._enabled
    
    def enable(self):
        """Enable middleware"""
        self._enabled = True
    
    def disable(self):
        """Disable middleware"""
        self._enabled = False
    
    async def process_request(self, request: APIRequest) -> Optional[APIRequest]:
        """Process incoming request"""
        if not self._enabled:
            return request
        return request
    
    async def process_response(self, response: APIResponse, 
                              request: APIRequest) -> Optional[APIResponse]:
        """Process outgoing response"""
        if not self._enabled:
            return response
        return response


class AuthenticationMiddleware(APIMiddleware):
    """Middleware for handling authentication"""
    
    def __init__(self, auth_type: str = "none", credentials: Dict[str, str] = None):
        super().__init__("authentication", priority=100)  # High priority
        self.auth_type = auth_type
        self.credentials = credentials or {}
    
    async def process_request(self, request: APIRequest) -> Optional[APIRequest]:
        """Add authentication headers to request"""
        if not self._enabled or self.auth_type == "none":
            return request
        
        try:
            auth_header = self._generate_auth_header()
            if auth_header:
                request.headers.update(auth_header)
                logger.debug(f"Added {self.auth_type} authentication to request")
            
            return request
            
        except Exception as e:
            logger.error(f"Authentication middleware error: {e}")
            raise AuthenticationError(f"Failed to add authentication: {e}")
    
    def _generate_auth_header(self) -> Optional[Dict[str, str]]:
        """Generate authentication header based on type"""
        
        if self.auth_type == "bearer":
            token = self.credentials.get("token")
            if not token:
                raise AuthenticationError("Bearer token not provided")
            return {"Authorization": f"Bearer {token}"}
        
        elif self.auth_type == "api_key":
            api_key = self.credentials.get("api_key")
            header_name = self.credentials.get("header_name", "X-API-Key")
            if not api_key:
                raise AuthenticationError("API key not provided")
            return {header_name: api_key}
        
        elif self.auth_type == "basic":
            username = self.credentials.get("username")
            password = self.credentials.get("password")
            if not username or not password:
                raise AuthenticationError("Username and password required for basic auth")
            
            credentials_string = f"{username}:{password}"
            encoded_credentials = base64.b64encode(credentials_string.encode()).decode()
            return {"Authorization": f"Basic {encoded_credentials}"}
        
        elif self.auth_type == "custom":
            # Allow custom headers to be passed directly
            return dict(self.credentials)
        
        return None
    
    def update_credentials(self, auth_type: str, credentials: Dict[str, str]):
        """Update authentication credentials"""
        self.auth_type = auth_type
        self.credentials = credentials
        logger.info(f"Updated authentication credentials for type: {auth_type}")


class LoggingMiddleware(APIMiddleware):
    """Middleware for logging requests and responses"""
    
    def __init__(self, log_requests: bool = True, log_responses: bool = True,
                 log_headers: bool = False, log_body: bool = False,
                 max_body_length: int = 1000):
        super().__init__("logging", priority=10)  # Low priority
        self.log_requests = log_requests
        self.log_responses = log_responses
        self.log_headers = log_headers
        self.log_body = log_body
        self.max_body_length = max_body_length
        self.request_times: Dict[str, float] = {}
    
    async def process_request(self, request: APIRequest) -> Optional[APIRequest]:
        """Log outgoing request"""
        if not self._enabled or not self.log_requests:
            return request
        
        # Record start time
        request_id = id(request)
        self.request_times[request_id] = time.time()
        
        # Build log message
        log_parts = [f"{request.method.value} {request.path}"]
        
        if self.log_headers and request.headers:
            # Filter out sensitive headers
            safe_headers = self._filter_sensitive_headers(request.headers)
            log_parts.append(f"Headers: {safe_headers}")
        
        if self.log_body and request.body:
            body_str = self._format_body(request.body)
            log_parts.append(f"Body: {body_str}")
        
        logger.info(f"Request: {' | '.join(log_parts)}")
        
        return request
    
    async def process_response(self, response: APIResponse, 
                              request: APIRequest) -> Optional[APIResponse]:
        """Log incoming response"""
        if not self._enabled or not self.log_responses:
            return response
        
        # Calculate response time
        request_id = id(request)
        response_time = None
        if request_id in self.request_times:
            response_time = time.time() - self.request_times[request_id]
            del self.request_times[request_id]
        
        # Build log message
        log_parts = [f"Response: {response.status_code}"]
        
        if response_time:
            log_parts.append(f"Time: {response_time:.3f}s")
        
        if self.log_headers and response.headers:
            safe_headers = self._filter_sensitive_headers(response.headers)
            log_parts.append(f"Headers: {safe_headers}")
        
        if self.log_body and response.body:
            body_str = self._format_body(response.body)
            log_parts.append(f"Body: {body_str}")
        
        # Choose log level based on status code
        if response.status_code >= 400:
            logger.warning(' | '.join(log_parts))
        else:
            logger.info(' | '.join(log_parts))
        
        return response
    
    def _filter_sensitive_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Filter out sensitive headers from logs"""
        sensitive_patterns = ['authorization', 'api-key', 'x-api-key', 'token', 'password']
        
        filtered = {}
        for key, value in headers.items():
            if any(pattern in key.lower() for pattern in sensitive_patterns):
                filtered[key] = "[REDACTED]"
            else:
                filtered[key] = value
        
        return filtered
    
    def _format_body(self, body: Any) -> str:
        """Format request/response body for logging"""
        try:
            if isinstance(body, (dict, list)):
                body_str = json.dumps(body, indent=None, separators=(',', ':'))
            else:
                body_str = str(body)
            
            # Truncate if too long
            if len(body_str) > self.max_body_length:
                body_str = body_str[:self.max_body_length] + "..."
            
            return body_str
            
        except Exception:
            return "[Unable to format body]"


class TimingMiddleware(APIMiddleware):
    """Middleware for tracking request timing"""
    
    def __init__(self):
        super().__init__("timing", priority=90)  # High priority to measure accurately
        self.request_times: Dict[str, float] = {}
        self.timing_stats = {
            "total_requests": 0,
            "total_time": 0.0,
            "min_time": float('inf'),
            "max_time": 0.0
        }
    
    async def process_request(self, request: APIRequest) -> Optional[APIRequest]:
        """Record request start time"""
        if not self._enabled:
            return request
        
        request_id = id(request)
        self.request_times[request_id] = time.time()
        
        return request
    
    async def process_response(self, response: APIResponse, 
                              request: APIRequest) -> Optional[APIResponse]:
        """Calculate and record response time"""
        if not self._enabled:
            return response
        
        request_id = id(request)
        start_time = self.request_times.get(request_id)
        
        if start_time:
            response_time = time.time() - start_time
            del self.request_times[request_id]
            
            # Store timing in response
            response.processing_time = response_time
            
            # Update statistics
            self.timing_stats["total_requests"] += 1
            self.timing_stats["total_time"] += response_time
            self.timing_stats["min_time"] = min(self.timing_stats["min_time"], response_time)
            self.timing_stats["max_time"] = max(self.timing_stats["max_time"], response_time)
            
            logger.debug(f"Request completed in {response_time:.3f}s")
        
        return response
    
    def get_timing_stats(self) -> Dict[str, Any]:
        """Get timing statistics"""
        stats = dict(self.timing_stats)
        
        if stats["total_requests"] > 0:
            stats["average_time"] = stats["total_time"] / stats["total_requests"]
        else:
            stats["average_time"] = 0.0
            stats["min_time"] = 0.0
        
        return stats
    
    def reset_stats(self):
        """Reset timing statistics"""
        self.timing_stats = {
            "total_requests": 0,
            "total_time": 0.0,
            "min_time": float('inf'),
            "max_time": 0.0
        }


class RetryContextMiddleware(APIMiddleware):
    """Middleware for adding retry context to requests"""
    
    def __init__(self):
        super().__init__("retry_context", priority=80)
        self.retry_attempts: Dict[str, int] = {}
    
    async def process_request(self, request: APIRequest) -> Optional[APIRequest]:
        """Add retry context to request"""
        if not self._enabled:
            return request
        
        request_key = f"{request.method.value}:{request.path}"
        
        # Track retry attempts
        if request_key not in self.retry_attempts:
            self.retry_attempts[request_key] = 0
        else:
            self.retry_attempts[request_key] += 1
        
        # Add retry headers
        retry_count = self.retry_attempts[request_key]
        if retry_count > 0:
            request.headers["X-Retry-Count"] = str(retry_count)
            request.headers["X-Request-Timestamp"] = datetime.now().isoformat()
        
        return request
    
    async def process_response(self, response: APIResponse, 
                              request: APIRequest) -> Optional[APIResponse]:
        """Reset retry context on success"""
        if not self._enabled:
            return response
        
        # Reset retry count on successful response
        if response.status_code < 400:
            request_key = f"{request.method.value}:{request.path}"
            if request_key in self.retry_attempts:
                del self.retry_attempts[request_key]
        
        return response


class CacheControlMiddleware(APIMiddleware):
    """Middleware for handling cache control headers"""
    
    def __init__(self):
        super().__init__("cache_control", priority=20)
    
    async def process_request(self, request: APIRequest) -> Optional[APIRequest]:
        """Add cache control headers to request"""
        if not self._enabled:
            return request
        
        # Add cache control headers for GET requests
        if request.method.value == "GET":
            if "Cache-Control" not in request.headers:
                request.headers["Cache-Control"] = "max-age=300"  # 5 minutes default
        
        return request
    
    async def process_response(self, response: APIResponse, 
                              request: APIRequest) -> Optional[APIResponse]:
        """Process cache control headers in response"""
        if not self._enabled:
            return response
        
        # Extract cache information from response headers
        cache_control = response.headers.get("Cache-Control", "")
        etag = response.headers.get("ETag", "")
        expires = response.headers.get("Expires", "")
        
        if cache_control or etag or expires:
            logger.debug(f"Cache info - Control: {cache_control}, ETag: {etag}, Expires: {expires}")
        
        return response


class UserAgentMiddleware(APIMiddleware):
    """Middleware for setting User-Agent header"""
    
    def __init__(self, user_agent: str = "AI-Detector-Client/1.0"):
        super().__init__("user_agent", priority=5)
        self.user_agent = user_agent
    
    async def process_request(self, request: APIRequest) -> Optional[APIRequest]:
        """Set User-Agent header"""
        if not self._enabled:
            return request
        
        if "User-Agent" not in request.headers:
            request.headers["User-Agent"] = self.user_agent
        
        return request
    
    def set_user_agent(self, user_agent: str):
        """Update User-Agent string"""
        self.user_agent = user_agent
        logger.debug(f"Updated User-Agent to: {user_agent}")


class ErrorHandlingMiddleware(APIMiddleware):
    """Middleware for standardized error handling"""
    
    def __init__(self):
        super().__init__("error_handling", priority=95)
        self.error_counts = {}
    
    async def process_response(self, response: APIResponse, 
                              request: APIRequest) -> Optional[APIResponse]:
        """Handle error responses"""
        if not self._enabled:
            return response
        
        if response.status_code >= 400:
            error_type = self._classify_error(response.status_code)
            
            # Count errors by type
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
            
            # Add error context to response
            if not hasattr(response, 'error_info'):
                response.error_info = {
                    "error_type": error_type,
                    "request_method": request.method.value,
                    "request_path": request.path,
                    "timestamp": datetime.now().isoformat()
                }
            
            logger.warning(f"API error: {error_type} ({response.status_code}) for {request.method.value} {request.path}")
        
        return response
    
    def _classify_error(self, status_code: int) -> str:
        """Classify error by status code"""
        if status_code == 400:
            return "bad_request"
        elif status_code == 401:
            return "unauthorized"
        elif status_code == 403:
            return "forbidden"
        elif status_code == 404:
            return "not_found"
        elif status_code == 408:
            return "timeout"
        elif status_code == 422:
            return "validation_error"
        elif status_code == 429:
            return "rate_limited"
        elif 500 <= status_code < 600:
            return "server_error"
        else:
            return "unknown_error"
    
    def get_error_stats(self) -> Dict[str, int]:
        """Get error statistics"""
        return dict(self.error_counts)
    
    def reset_error_stats(self):
        """Reset error statistics"""
        self.error_counts.clear()


def create_middleware_chain(config: Dict[str, Any]) -> list[APIMiddleware]:
    """Create middleware chain based on configuration"""
    middleware = []
    
    # Add authentication middleware if configured
    if config.get("auth_type", "none") != "none":
        auth_middleware = AuthenticationMiddleware(
            auth_type=config["auth_type"],
            credentials=config.get("auth_credentials", {})
        )
        middleware.append(auth_middleware)
    
    # Add logging middleware if enabled
    if config.get("enable_logging", True):
        logging_middleware = LoggingMiddleware(
            log_requests=config.get("log_requests", True),
            log_responses=config.get("log_responses", True),
            log_headers=config.get("log_headers", False),
            log_body=config.get("log_body", False)
        )
        middleware.append(logging_middleware)
    
    # Add timing middleware
    middleware.append(TimingMiddleware())
    
    # Add error handling middleware
    middleware.append(ErrorHandlingMiddleware())
    
    # Add retry context middleware
    middleware.append(RetryContextMiddleware())
    
    # Add cache control middleware
    middleware.append(CacheControlMiddleware())
    
    # Add user agent middleware
    user_agent = config.get("user_agent", "AI-Detector-Client/1.0")
    middleware.append(UserAgentMiddleware(user_agent))
    
    # Sort by priority (higher priority first)
    middleware.sort(key=lambda m: m.get_priority(), reverse=True)
    
    logger.info(f"Created middleware chain with {len(middleware)} components")
    return middleware