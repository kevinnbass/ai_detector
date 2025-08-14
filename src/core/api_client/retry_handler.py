"""
Retry Handler for API Requests
Intelligent retry mechanisms with backoff strategies
"""

import asyncio
import random
from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime, timedelta
from enum import Enum
import logging

from .exceptions import RetryExhaustedError, RateLimitError, NetworkError, ServerError

logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    """Retry strategy types"""
    FIXED = "fixed"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    FIBONACCI = "fibonacci"
    JITTERED = "jittered"


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_retries: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    base_delay: float = 1.0  # Base delay in seconds
    max_delay: float = 60.0  # Maximum delay in seconds
    backoff_multiplier: float = 2.0  # For exponential backoff
    jitter_max: float = 0.1  # Maximum jitter (0.0 to 1.0)
    
    # Conditions for retrying
    retry_on_network_error: bool = True
    retry_on_server_error: bool = True
    retry_on_timeout: bool = True
    retry_on_rate_limit: bool = True
    retry_on_status_codes: List[int] = None
    
    # Custom retry condition function
    retry_condition: Optional[Callable[[Exception], bool]] = None
    
    def __post_init__(self):
        if self.retry_on_status_codes is None:
            self.retry_on_status_codes = [502, 503, 504, 429]


class RetryAttempt:
    """Represents a retry attempt"""
    
    def __init__(self, attempt_number: int, delay: float, reason: str):
        self.attempt_number = attempt_number
        self.delay = delay
        self.reason = reason
        self.timestamp = datetime.now()
        self.error: Optional[Exception] = None
        self.success = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempt_number": self.attempt_number,
            "delay": self.delay,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
            "error": str(self.error) if self.error else None,
            "success": self.success
        }


class RetryHandler:
    """Handles retry logic for API requests"""
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self._fibonacci_cache = [1, 1]  # For Fibonacci backoff
    
    async def execute_with_retry(self, 
                                operation: Callable,
                                *args,
                                **kwargs) -> Any:
        """Execute operation with retry logic"""
        
        attempts: List[RetryAttempt] = []
        last_error = None
        
        for attempt_num in range(self.config.max_retries + 1):
            try:
                if attempt_num > 0:
                    # Calculate delay for retry
                    delay = self._calculate_delay(attempt_num)
                    reason = f"Retry attempt {attempt_num}"
                    
                    attempt = RetryAttempt(attempt_num, delay, reason)
                    attempts.append(attempt)
                    
                    logger.info(f"Retrying operation (attempt {attempt_num}/{self.config.max_retries}) "
                               f"after {delay:.2f}s delay")
                    
                    await asyncio.sleep(delay)
                
                # Execute the operation
                result = await operation(*args, **kwargs)
                
                if attempts:
                    attempts[-1].success = True
                
                logger.debug(f"Operation succeeded after {attempt_num} retries")
                return result
                
            except Exception as e:
                last_error = e
                
                if attempts:
                    attempts[-1].error = e
                
                # Check if we should retry this error
                if not self._should_retry(e, attempt_num):
                    logger.debug(f"Not retrying error: {e}")
                    break
                
                if attempt_num >= self.config.max_retries:
                    logger.warning(f"Max retries ({self.config.max_retries}) exceeded")
                    break
                
                logger.warning(f"Operation failed (attempt {attempt_num + 1}): {e}")
        
        # All retries exhausted
        raise RetryExhaustedError(
            message=f"Operation failed after {self.config.max_retries} retries",
            max_retries=self.config.max_retries,
            last_error=last_error
        )
    
    def _should_retry(self, error: Exception, attempt_num: int) -> bool:
        """Determine if error should trigger a retry"""
        
        # Don't retry if we've hit max attempts
        if attempt_num >= self.config.max_retries:
            return False
        
        # Check custom retry condition first
        if self.config.retry_condition:
            return self.config.retry_condition(error)
        
        # Rate limit errors - respect retry-after if available
        if isinstance(error, RateLimitError):
            if self.config.retry_on_rate_limit:
                # If retry-after is specified and reasonable, use it
                if error.retry_after and error.retry_after <= self.config.max_delay:
                    return True
                return True
            return False
        
        # Network errors
        if isinstance(error, NetworkError):
            return self.config.retry_on_network_error
        
        # Server errors (5xx)
        if isinstance(error, ServerError):
            return self.config.retry_on_server_error
        
        # Timeout errors
        if "timeout" in str(error).lower():
            return self.config.retry_on_timeout
        
        # Check specific status codes
        if hasattr(error, 'status_code') and error.status_code:
            return error.status_code in self.config.retry_on_status_codes
        
        # Default: don't retry
        return False
    
    def _calculate_delay(self, attempt_num: int, base_error: Exception = None) -> float:
        """Calculate delay for retry attempt"""
        
        # Handle rate limit with retry-after header
        if isinstance(base_error, RateLimitError) and base_error.retry_after:
            return min(base_error.retry_after, self.config.max_delay)
        
        # Calculate delay based on strategy
        if self.config.strategy == RetryStrategy.FIXED:
            delay = self.config.base_delay
            
        elif self.config.strategy == RetryStrategy.LINEAR:
            delay = self.config.base_delay * attempt_num
            
        elif self.config.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.config.base_delay * (self.config.backoff_multiplier ** (attempt_num - 1))
            
        elif self.config.strategy == RetryStrategy.FIBONACCI:
            delay = self.config.base_delay * self._get_fibonacci(attempt_num)
            
        elif self.config.strategy == RetryStrategy.JITTERED:
            # Exponential backoff with jitter
            base_delay = self.config.base_delay * (self.config.backoff_multiplier ** (attempt_num - 1))
            jitter = random.uniform(-self.config.jitter_max, self.config.jitter_max) * base_delay
            delay = base_delay + jitter
            
        else:
            delay = self.config.base_delay
        
        # Cap at maximum delay
        return min(delay, self.config.max_delay)
    
    def _get_fibonacci(self, n: int) -> int:
        """Get nth Fibonacci number (cached)"""
        while len(self._fibonacci_cache) <= n:
            next_fib = self._fibonacci_cache[-1] + self._fibonacci_cache[-2]
            self._fibonacci_cache.append(next_fib)
        
        return self._fibonacci_cache[n]
    
    def get_config(self) -> RetryConfig:
        """Get retry configuration"""
        return self.config
    
    def update_config(self, **kwargs) -> None:
        """Update retry configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                logger.warning(f"Unknown config parameter: {key}")


class AdaptiveRetryHandler(RetryHandler):
    """Retry handler that adapts based on success/failure patterns"""
    
    def __init__(self, config: RetryConfig, adaptation_window: int = 100):
        super().__init__(config)
        self.adaptation_window = adaptation_window
        self.success_history: List[bool] = []
        self.error_patterns: Dict[str, int] = {}
        self.last_adaptation = datetime.now()
        self.adaptation_interval = timedelta(minutes=5)
    
    async def execute_with_retry(self, operation: Callable, *args, **kwargs) -> Any:
        """Execute operation with adaptive retry logic"""
        
        # Adapt configuration based on recent patterns
        if datetime.now() - self.last_adaptation > self.adaptation_interval:
            self._adapt_configuration()
            self.last_adaptation = datetime.now()
        
        try:
            result = await super().execute_with_retry(operation, *args, **kwargs)
            self._record_success(True)
            return result
            
        except Exception as e:
            self._record_success(False)
            self._record_error_pattern(e)
            raise
    
    def _adapt_configuration(self) -> None:
        """Adapt retry configuration based on historical data"""
        if len(self.success_history) < 10:
            return  # Not enough data
        
        recent_successes = self.success_history[-50:]  # Last 50 attempts
        success_rate = sum(recent_successes) / len(recent_successes)
        
        logger.debug(f"Recent success rate: {success_rate:.2%}")
        
        # Adjust retry count based on success rate
        if success_rate < 0.7:  # Low success rate
            # Increase retries but reduce delays (fail fast approach)
            self.config.max_retries = min(self.config.max_retries + 1, 10)
            self.config.base_delay = max(self.config.base_delay * 0.9, 0.5)
            logger.debug("Low success rate: increased retries, reduced delays")
            
        elif success_rate > 0.9:  # High success rate
            # Reduce retries and increase delays (conservative approach)
            self.config.max_retries = max(self.config.max_retries - 1, 1)
            self.config.base_delay = min(self.config.base_delay * 1.1, 5.0)
            logger.debug("High success rate: reduced retries, increased delays")
        
        # Adapt strategy based on most common errors
        most_common_error = self._get_most_common_error()
        if most_common_error:
            if "rate" in most_common_error.lower():
                # Rate limit issues - use longer delays
                self.config.strategy = RetryStrategy.LINEAR
                self.config.base_delay = max(self.config.base_delay, 2.0)
            elif "network" in most_common_error.lower() or "timeout" in most_common_error.lower():
                # Network issues - use jittered exponential backoff
                self.config.strategy = RetryStrategy.JITTERED
                self.config.jitter_max = 0.2
    
    def _record_success(self, success: bool) -> None:
        """Record success/failure for adaptation"""
        self.success_history.append(success)
        
        # Keep only recent history
        if len(self.success_history) > self.adaptation_window:
            self.success_history = self.success_history[-self.adaptation_window:]
    
    def _record_error_pattern(self, error: Exception) -> None:
        """Record error patterns for adaptation"""
        error_type = type(error).__name__
        self.error_patterns[error_type] = self.error_patterns.get(error_type, 0) + 1
        
        # Clean old patterns periodically
        if len(self.error_patterns) > 50:
            # Keep only the most common errors
            sorted_errors = sorted(self.error_patterns.items(), key=lambda x: x[1], reverse=True)
            self.error_patterns = dict(sorted_errors[:25])
    
    def _get_most_common_error(self) -> Optional[str]:
        """Get the most common error type"""
        if not self.error_patterns:
            return None
        
        return max(self.error_patterns.items(), key=lambda x: x[1])[0]
    
    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Get adaptation statistics"""
        recent_successes = self.success_history[-50:] if self.success_history else []
        success_rate = sum(recent_successes) / len(recent_successes) if recent_successes else 0.0
        
        return {
            "success_rate": success_rate,
            "total_attempts": len(self.success_history),
            "error_patterns": dict(self.error_patterns),
            "most_common_error": self._get_most_common_error(),
            "current_config": {
                "max_retries": self.config.max_retries,
                "strategy": self.config.strategy.value,
                "base_delay": self.config.base_delay
            },
            "last_adaptation": self.last_adaptation.isoformat()
        }


def create_retry_handler(strategy: str = "exponential", 
                        max_retries: int = 3,
                        adaptive: bool = False,
                        **kwargs) -> RetryHandler:
    """Factory function to create retry handlers"""
    
    strategy_enum = RetryStrategy(strategy)
    config = RetryConfig(
        max_retries=max_retries,
        strategy=strategy_enum,
        **kwargs
    )
    
    if adaptive:
        return AdaptiveRetryHandler(config)
    else:
        return RetryHandler(config)