"""
Rate Limiter for API Requests
Implements various rate limiting strategies
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging

from .exceptions import RateLimitError

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting"""
    
    # Global limits
    global_requests_per_second: float = 10.0
    global_requests_per_minute: float = 600.0
    global_requests_per_hour: float = 3600.0
    
    # Per-endpoint limits (endpoint -> requests per second)
    endpoint_limits: Dict[str, float] = None
    
    # Burst allowance
    burst_size: int = 20  # Allow short bursts above the rate limit
    
    # Strategy
    strategy: str = "token_bucket"  # token_bucket, sliding_window, fixed_window
    
    def __post_init__(self):
        if self.endpoint_limits is None:
            self.endpoint_limits = {}


class TokenBucket:
    """Token bucket rate limiter"""
    
    def __init__(self, rate: float, capacity: int):
        self.rate = rate  # tokens per second
        self.capacity = capacity  # max tokens
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens from bucket"""
        async with self.lock:
            now = time.time()
            
            # Refill tokens based on elapsed time
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_refill = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    async def wait_for_tokens(self, tokens: int = 1, timeout: float = 60.0) -> bool:
        """Wait for tokens to become available"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if await self.acquire(tokens):
                return True
            
            # Calculate wait time
            async with self.lock:
                needed_tokens = tokens - self.tokens
                wait_time = needed_tokens / self.rate if self.rate > 0 else 1.0
                wait_time = min(wait_time, 1.0)  # Max 1 second wait
            
            await asyncio.sleep(wait_time)
        
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get token bucket statistics"""
        return {
            "rate": self.rate,
            "capacity": self.capacity,
            "available_tokens": self.tokens,
            "utilization": 1.0 - (self.tokens / self.capacity)
        }


class SlidingWindow:
    """Sliding window rate limiter"""
    
    def __init__(self, rate: float, window_size: float = 60.0):
        self.rate = rate  # requests per window
        self.window_size = window_size  # window size in seconds
        self.requests = deque()
        self.lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        """Check if request can be made"""
        async with self.lock:
            now = time.time()
            
            # Remove old requests outside the window
            cutoff_time = now - self.window_size
            while self.requests and self.requests[0] < cutoff_time:
                self.requests.popleft()
            
            # Check if we can make another request
            if len(self.requests) < self.rate:
                self.requests.append(now)
                return True
            
            return False
    
    async def wait_for_availability(self, timeout: float = 60.0) -> bool:
        """Wait for next available slot"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if await self.acquire():
                return True
            
            async with self.lock:
                if self.requests:
                    # Calculate wait time until oldest request expires
                    oldest_request = self.requests[0]
                    wait_time = (oldest_request + self.window_size) - time.time()
                    wait_time = max(0.1, min(wait_time, 1.0))
                else:
                    wait_time = 0.1
            
            await asyncio.sleep(wait_time)
        
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get sliding window statistics"""
        now = time.time()
        cutoff_time = now - self.window_size
        
        # Count requests in current window
        current_requests = sum(1 for req_time in self.requests if req_time >= cutoff_time)
        
        return {
            "rate": self.rate,
            "window_size": self.window_size,
            "current_requests": current_requests,
            "utilization": current_requests / self.rate if self.rate > 0 else 0
        }


class RateLimiter:
    """Multi-strategy rate limiter"""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        
        # Global rate limiters
        self.global_limiters = {}
        self._setup_global_limiters()
        
        # Per-endpoint rate limiters
        self.endpoint_limiters: Dict[str, TokenBucket] = {}
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "limited_requests": 0,
            "total_wait_time": 0.0
        }
    
    def _setup_global_limiters(self):
        """Setup global rate limiters"""
        if self.config.strategy == "token_bucket":
            if self.config.global_requests_per_second > 0:
                self.global_limiters["per_second"] = TokenBucket(
                    self.config.global_requests_per_second,
                    self.config.burst_size
                )
            
            if self.config.global_requests_per_minute > 0:
                self.global_limiters["per_minute"] = TokenBucket(
                    self.config.global_requests_per_minute / 60.0,  # Convert to per-second
                    max(self.config.burst_size, int(self.config.global_requests_per_minute * 0.1))
                )
            
            if self.config.global_requests_per_hour > 0:
                self.global_limiters["per_hour"] = TokenBucket(
                    self.config.global_requests_per_hour / 3600.0,  # Convert to per-second
                    max(self.config.burst_size, int(self.config.global_requests_per_hour * 0.02))
                )
        
        elif self.config.strategy == "sliding_window":
            if self.config.global_requests_per_second > 0:
                self.global_limiters["per_second"] = SlidingWindow(
                    self.config.global_requests_per_second, 1.0
                )
            
            if self.config.global_requests_per_minute > 0:
                self.global_limiters["per_minute"] = SlidingWindow(
                    self.config.global_requests_per_minute, 60.0
                )
            
            if self.config.global_requests_per_hour > 0:
                self.global_limiters["per_hour"] = SlidingWindow(
                    self.config.global_requests_per_hour, 3600.0
                )
    
    async def acquire(self, endpoint: str, tokens: int = 1, timeout: float = 30.0) -> None:
        """Acquire permission to make request"""
        self.stats["total_requests"] += 1
        start_time = time.time()
        
        try:
            # Check global limits first
            for limiter_name, limiter in self.global_limiters.items():
                if isinstance(limiter, TokenBucket):
                    if not await limiter.wait_for_tokens(tokens, timeout):
                        self._handle_rate_limit("global", limiter_name)
                elif isinstance(limiter, SlidingWindow):
                    if not await limiter.wait_for_availability(timeout):
                        self._handle_rate_limit("global", limiter_name)
            
            # Check endpoint-specific limits
            endpoint_limiter = self._get_endpoint_limiter(endpoint)
            if endpoint_limiter:
                if not await endpoint_limiter.wait_for_tokens(tokens, timeout):
                    self._handle_rate_limit("endpoint", endpoint)
            
            wait_time = time.time() - start_time
            self.stats["total_wait_time"] += wait_time
            
            if wait_time > 0.1:  # Log significant waits
                logger.debug(f"Rate limit wait time: {wait_time:.2f}s for {endpoint}")
        
        except Exception as e:
            logger.error(f"Rate limiter error: {e}")
            # Don't block requests on rate limiter errors
    
    def _get_endpoint_limiter(self, endpoint: str) -> Optional[TokenBucket]:
        """Get or create endpoint-specific limiter"""
        if endpoint not in self.config.endpoint_limits:
            return None
        
        if endpoint not in self.endpoint_limiters:
            rate = self.config.endpoint_limits[endpoint]
            capacity = max(self.config.burst_size, int(rate * 2))
            self.endpoint_limiters[endpoint] = TokenBucket(rate, capacity)
        
        return self.endpoint_limiters[endpoint]
    
    def _handle_rate_limit(self, limit_type: str, identifier: str):
        """Handle rate limit exceeded"""
        self.stats["limited_requests"] += 1
        
        error_msg = f"Rate limit exceeded for {limit_type}: {identifier}"
        logger.warning(error_msg)
        
        # Calculate retry-after based on limiter type
        retry_after = self._calculate_retry_after(limit_type, identifier)
        
        raise RateLimitError(
            message=error_msg,
            retry_after=retry_after
        )
    
    def _calculate_retry_after(self, limit_type: str, identifier: str) -> Optional[int]:
        """Calculate retry-after time"""
        if limit_type == "global":
            if identifier == "per_second":
                return 1
            elif identifier == "per_minute":
                return 60
            elif identifier == "per_hour":
                return 3600
        elif limit_type == "endpoint":
            endpoint_rate = self.config.endpoint_limits.get(identifier, 1.0)
            return max(1, int(1.0 / endpoint_rate))
        
        return 30  # Default fallback
    
    def add_endpoint_limit(self, endpoint: str, requests_per_second: float):
        """Add or update endpoint-specific rate limit"""
        self.config.endpoint_limits[endpoint] = requests_per_second
        
        # Remove existing limiter to force recreation with new rate
        if endpoint in self.endpoint_limiters:
            del self.endpoint_limiters[endpoint]
        
        logger.info(f"Set rate limit for {endpoint}: {requests_per_second} req/s")
    
    def remove_endpoint_limit(self, endpoint: str):
        """Remove endpoint-specific rate limit"""
        if endpoint in self.config.endpoint_limits:
            del self.config.endpoint_limits[endpoint]
        
        if endpoint in self.endpoint_limiters:
            del self.endpoint_limiters[endpoint]
        
        logger.info(f"Removed rate limit for {endpoint}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        stats = {**self.stats}
        
        # Add global limiter stats
        stats["global_limiters"] = {}
        for name, limiter in self.global_limiters.items():
            stats["global_limiters"][name] = limiter.get_stats()
        
        # Add endpoint limiter stats
        stats["endpoint_limiters"] = {}
        for endpoint, limiter in self.endpoint_limiters.items():
            stats["endpoint_limiters"][endpoint] = limiter.get_stats()
        
        # Calculate derived stats
        if self.stats["total_requests"] > 0:
            stats["average_wait_time"] = self.stats["total_wait_time"] / self.stats["total_requests"]
            stats["limit_hit_rate"] = self.stats["limited_requests"] / self.stats["total_requests"]
        else:
            stats["average_wait_time"] = 0.0
            stats["limit_hit_rate"] = 0.0
        
        return stats
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            "total_requests": 0,
            "limited_requests": 0,
            "total_wait_time": 0.0
        }
        logger.debug("Rate limiter statistics reset")
    
    def update_config(self, config: RateLimitConfig):
        """Update rate limiter configuration"""
        old_config = self.config
        self.config = config
        
        # Recreate global limiters if rates changed
        if (config.global_requests_per_second != old_config.global_requests_per_second or
            config.global_requests_per_minute != old_config.global_requests_per_minute or
            config.global_requests_per_hour != old_config.global_requests_per_hour):
            self.global_limiters.clear()
            self._setup_global_limiters()
        
        # Clear endpoint limiters that were removed or changed
        for endpoint in list(self.endpoint_limiters.keys()):
            if (endpoint not in config.endpoint_limits or
                config.endpoint_limits[endpoint] != old_config.endpoint_limits.get(endpoint)):
                del self.endpoint_limiters[endpoint]
        
        logger.info("Rate limiter configuration updated")


class AdaptiveRateLimiter(RateLimiter):
    """Rate limiter that adapts based on server responses"""
    
    def __init__(self, config: RateLimitConfig, adaptation_factor: float = 0.1):
        super().__init__(config)
        self.adaptation_factor = adaptation_factor
        self.response_times = deque(maxlen=100)  # Track recent response times
        self.error_count = 0
        self.success_count = 0
        self.last_adaptation = time.time()
    
    def record_response(self, success: bool, response_time: float, status_code: int = 200):
        """Record API response for adaptation"""
        self.response_times.append(response_time)
        
        if success and status_code == 200:
            self.success_count += 1
        else:
            self.error_count += 1
            
            # Rate limit errors should cause immediate slowdown
            if status_code == 429:
                self._adapt_to_rate_limit()
        
        # Periodic adaptation based on performance
        if time.time() - self.last_adaptation > 60:  # Adapt every minute
            self._adapt_to_performance()
            self.last_adaptation = time.time()
    
    def _adapt_to_rate_limit(self):
        """Adapt to rate limit responses by reducing limits"""
        logger.info("Adapting to rate limit - reducing request rates")
        
        # Reduce all rates by adaptation factor
        self.config.global_requests_per_second *= (1 - self.adaptation_factor)
        self.config.global_requests_per_minute *= (1 - self.adaptation_factor)
        self.config.global_requests_per_hour *= (1 - self.adaptation_factor)
        
        # Reduce endpoint limits
        for endpoint in self.config.endpoint_limits:
            self.config.endpoint_limits[endpoint] *= (1 - self.adaptation_factor)
        
        # Recreate limiters with new rates
        self.global_limiters.clear()
        self.endpoint_limiters.clear()
        self._setup_global_limiters()
    
    def _adapt_to_performance(self):
        """Adapt based on overall performance"""
        if not self.response_times:
            return
        
        total_requests = self.success_count + self.error_count
        if total_requests < 10:  # Need enough data
            return
        
        success_rate = self.success_count / total_requests
        avg_response_time = sum(self.response_times) / len(self.response_times)
        
        logger.debug(f"Performance metrics - Success rate: {success_rate:.2%}, "
                    f"Avg response time: {avg_response_time:.2f}s")
        
        # Increase rates if performing well
        if success_rate > 0.95 and avg_response_time < 1.0:
            adjustment = 1 + (self.adaptation_factor * 0.5)  # Smaller increases
            self.config.global_requests_per_second *= adjustment
            self.config.global_requests_per_minute *= adjustment
            self.config.global_requests_per_hour *= adjustment
            
            logger.debug("Performance good - slightly increasing request rates")
        
        # Decrease rates if performing poorly
        elif success_rate < 0.8 or avg_response_time > 5.0:
            adjustment = 1 - self.adaptation_factor
            self.config.global_requests_per_second *= adjustment
            self.config.global_requests_per_minute *= adjustment
            self.config.global_requests_per_hour *= adjustment
            
            logger.debug("Performance poor - decreasing request rates")
        
        # Reset counters
        self.success_count = 0
        self.error_count = 0
        
        # Recreate limiters if rates changed significantly
        self.global_limiters.clear()
        self.endpoint_limiters.clear()
        self._setup_global_limiters()