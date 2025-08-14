"""
FastAPI Middleware for AI Detector API
Custom middleware for logging, rate limiting, and error handling
"""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from typing import Dict, Any, Optional
import time
import logging
import json
import asyncio
from datetime import datetime, timedelta
from collections import defaultdict
import uuid

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging API requests and responses"""
    
    def __init__(self, app, log_body: bool = False, max_body_size: int = 1000):
        super().__init__(app)
        self.log_body = log_body
        self.max_body_size = max_body_size
    
    async def dispatch(self, request: Request, call_next):
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        start_time = time.time()
        
        # Log request
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "client_ip": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("user-agent", "unknown"),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Log request body if enabled
        if self.log_body and request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if len(body) <= self.max_body_size:
                    log_data["request_body"] = body.decode("utf-8")[:self.max_body_size]
                else:
                    log_data["request_body"] = f"<body too large: {len(body)} bytes>"
            except Exception as e:
                log_data["request_body_error"] = str(e)
        
        logger.info(f"Request started: {request.method} {request.url}", extra=log_data)
        
        # Process request
        try:
            response = await call_next(request)
            processing_time = time.time() - start_time
            
            # Log response
            log_data.update({
                "status_code": response.status_code,
                "processing_time": processing_time,
                "response_headers": dict(response.headers)
            })
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Processing-Time"] = f"{processing_time:.3f}"
            
            # Log completion
            if response.status_code >= 400:
                logger.warning(f"Request completed with error: {response.status_code}", extra=log_data)
            else:
                logger.info(f"Request completed successfully: {response.status_code}", extra=log_data)
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            log_data.update({
                "error": str(e),
                "processing_time": processing_time
            })
            logger.error(f"Request failed with exception", extra=log_data)
            raise


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware"""
    
    def __init__(self, app, default_rate: int = 100, window_seconds: int = 60):
        super().__init__(app)
        self.default_rate = default_rate
        self.window_seconds = window_seconds
        self.client_requests = defaultdict(list)
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()
        
        # Rate limits by endpoint pattern
        self.endpoint_limits = {
            "/api/v1/detect": {"rate": 60, "window": 60},
            "/api/v1/detect/batch": {"rate": 10, "window": 60},
            "/api/v1/train": {"rate": 5, "window": 300},
            "/health": {"rate": 200, "window": 60}
        }
    
    async def dispatch(self, request: Request, call_next):
        # Cleanup old entries periodically
        if time.time() - self.last_cleanup > self.cleanup_interval:
            await self._cleanup_old_entries()
        
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        endpoint = self._get_endpoint_pattern(str(request.url.path))
        
        # Get rate limit for endpoint
        rate_config = self.endpoint_limits.get(endpoint, {
            "rate": self.default_rate,
            "window": self.window_seconds
        })
        
        rate_limit = rate_config["rate"]
        window = rate_config["window"]
        
        # Check rate limit
        client_key = f"{client_ip}:{endpoint}"
        request_times = self.client_requests[client_key]
        
        # Remove old requests outside window
        cutoff_time = current_time - window
        request_times[:] = [t for t in request_times if t > cutoff_time]
        
        # Check if rate limit exceeded
        if len(request_times) >= rate_limit:
            reset_time = min(request_times) + window
            retry_after = int(reset_time - current_time) + 1
            
            logger.warning(f"Rate limit exceeded for {client_ip} on {endpoint}")
            
            return JSONResponse(
                status_code=429,
                content={
                    "success": False,
                    "message": "Rate limit exceeded",
                    "error_code": "RATE_LIMIT_EXCEEDED",
                    "rate_limit": {
                        "limit": rate_limit,
                        "window": window,
                        "remaining": 0,
                        "retry_after": retry_after
                    }
                },
                headers={
                    "X-RateLimit-Limit": str(rate_limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(reset_time)),
                    "Retry-After": str(retry_after)
                }
            )
        
        # Record this request
        request_times.append(current_time)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = max(0, rate_limit - len(request_times))
        response.headers["X-RateLimit-Limit"] = str(rate_limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(current_time + window))
        
        return response
    
    def _get_endpoint_pattern(self, path: str) -> str:
        """Get endpoint pattern for rate limiting"""
        for pattern in self.endpoint_limits.keys():
            if path.startswith(pattern):
                return pattern
        return "default"
    
    async def _cleanup_old_entries(self):
        """Cleanup old rate limiting entries"""
        current_time = time.time()
        
        # Remove entries older than max window
        max_window = max(config["window"] for config in self.endpoint_limits.values())
        cutoff_time = current_time - max_window
        
        for client_key in list(self.client_requests.keys()):
            request_times = self.client_requests[client_key]
            request_times[:] = [t for t in request_times if t > cutoff_time]
            
            # Remove empty entries
            if not request_times:
                del self.client_requests[client_key]
        
        self.last_cleanup = current_time


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Global error handling middleware"""
    
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
            
        except Exception as e:
            # Get request ID if available
            request_id = getattr(request.state, 'request_id', 'unknown')
            
            logger.error(
                f"Unhandled exception in request {request_id}",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "url": str(request.url),
                    "error": str(e),
                    "error_type": type(e).__name__
                },
                exc_info=True
            )
            
            # Return JSON error response
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": "Internal server error",
                    "error_code": "INTERNAL_ERROR",
                    "request_id": request_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )


class CacheMiddleware(BaseHTTPMiddleware):
    """Simple caching middleware for GET requests"""
    
    def __init__(self, app, cache_ttl: int = 300, max_cache_size: int = 1000):
        super().__init__(app)
        self.cache_ttl = cache_ttl
        self.max_cache_size = max_cache_size
        self.cache = {}
        self.cache_times = {}
        
        # Cacheable endpoints
        self.cacheable_paths = [
            "/health",
            "/api/v1/statistics"
        ]
    
    async def dispatch(self, request: Request, call_next):
        # Only cache GET requests
        if request.method != "GET":
            return await call_next(request)
        
        # Check if path is cacheable
        path = request.url.path
        if not any(path.startswith(cacheable) for cacheable in self.cacheable_paths):
            return await call_next(request)
        
        # Create cache key
        cache_key = f"{path}?{request.url.query}" if request.url.query else path
        current_time = time.time()
        
        # Check cache
        if cache_key in self.cache:
            cache_time = self.cache_times.get(cache_key, 0)
            if current_time - cache_time < self.cache_ttl:
                logger.debug(f"Cache hit for {cache_key}")
                cached_response = self.cache[cache_key]
                
                # Create response from cached data
                response = Response(
                    content=cached_response["body"],
                    status_code=cached_response["status_code"],
                    headers=cached_response["headers"]
                )
                response.headers["X-Cache"] = "HIT"
                return response
        
        # Process request
        response = await call_next(request)
        
        # Cache successful responses
        if response.status_code == 200:
            # Read response body
            body = b""
            async for chunk in response.body_iterator:
                body += chunk
            
            # Cache the response
            self.cache[cache_key] = {
                "body": body,
                "status_code": response.status_code,
                "headers": dict(response.headers)
            }
            self.cache_times[cache_key] = current_time
            
            # Cleanup cache if too large
            if len(self.cache) > self.max_cache_size:
                self._cleanup_cache()
            
            # Create new response with cached body
            response = Response(
                content=body,
                status_code=response.status_code,
                headers=response.headers
            )
            response.headers["X-Cache"] = "MISS"
        
        return response
    
    def _cleanup_cache(self):
        """Remove oldest cache entries"""
        # Sort by cache time and remove oldest 10%
        sorted_items = sorted(self.cache_times.items(), key=lambda x: x[1])
        items_to_remove = len(sorted_items) // 10
        
        for cache_key, _ in sorted_items[:items_to_remove]:
            self.cache.pop(cache_key, None)
            self.cache_times.pop(cache_key, None)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to responses"""
    
    def __init__(self, app):
        super().__init__(app)
        self.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline';",
        }
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Add security headers
        for header_name, header_value in self.security_headers.items():
            if header_name not in response.headers:
                response.headers[header_name] = header_value
        
        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """Collect API metrics"""
    
    def __init__(self, app):
        super().__init__(app)
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_response_time": 0.0,
            "requests_by_endpoint": defaultdict(int),
            "errors_by_code": defaultdict(int),
            "start_time": time.time()
        }
        self._lock = asyncio.Lock()
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        endpoint = request.url.path
        
        async with self._lock:
            self.metrics["total_requests"] += 1
            self.metrics["requests_by_endpoint"][endpoint] += 1
        
        try:
            response = await call_next(request)
            response_time = time.time() - start_time
            
            async with self._lock:
                self.metrics["total_response_time"] += response_time
                
                if response.status_code < 400:
                    self.metrics["successful_requests"] += 1
                else:
                    self.metrics["failed_requests"] += 1
                    self.metrics["errors_by_code"][response.status_code] += 1
            
            # Add metrics to response headers
            response.headers["X-Metrics-Total-Requests"] = str(self.metrics["total_requests"])
            response.headers["X-Metrics-Success-Rate"] = f"{self._get_success_rate():.2f}"
            
            return response
            
        except Exception as e:
            async with self._lock:
                self.metrics["failed_requests"] += 1
                self.metrics["errors_by_code"][500] += 1
            raise
    
    def _get_success_rate(self) -> float:
        """Calculate success rate"""
        total = self.metrics["total_requests"]
        if total == 0:
            return 0.0
        return (self.metrics["successful_requests"] / total) * 100
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        total_requests = self.metrics["total_requests"]
        uptime = time.time() - self.metrics["start_time"]
        
        return {
            "total_requests": total_requests,
            "successful_requests": self.metrics["successful_requests"],
            "failed_requests": self.metrics["failed_requests"],
            "success_rate": self._get_success_rate(),
            "average_response_time": (
                self.metrics["total_response_time"] / total_requests 
                if total_requests > 0 else 0
            ),
            "requests_per_minute": total_requests / (uptime / 60) if uptime > 0 else 0,
            "uptime_seconds": uptime,
            "requests_by_endpoint": dict(self.metrics["requests_by_endpoint"]),
            "errors_by_code": dict(self.metrics["errors_by_code"])
        }


# Global metrics instance for access from routes
metrics_middleware = MetricsMiddleware(None)


__all__ = [
    'RequestLoggingMiddleware', 'RateLimitingMiddleware', 'ErrorHandlingMiddleware',
    'CacheMiddleware', 'SecurityHeadersMiddleware', 'MetricsMiddleware',
    'metrics_middleware'
]