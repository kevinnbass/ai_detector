"""
Error recovery mechanisms for the AI Detector system.

Provides automated recovery strategies for various error types,
including fallbacks, retries, and circuit breakers.
"""

import time
import logging
from typing import Optional, Dict, Any, Callable, List, Type
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from functools import wraps

from .exceptions import (
    AIDetectorException,
    RetryableError,
    TimeoutError,
    APIError,
    ServiceError,
    DetectionError
)
from .context import ErrorContext


logger = logging.getLogger(__name__)


class RecoveryStrategy(Enum):
    """Available recovery strategies."""
    
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAK = "circuit_break"
    CACHE = "cache"
    DEGRADE = "degrade"
    SKIP = "skip"
    QUEUE = "queue"


@dataclass
class RecoveryConfig:
    """Configuration for error recovery."""
    
    # Retry configuration
    max_retries: int = 3
    initial_delay_ms: int = 1000
    max_delay_ms: int = 30000
    backoff_multiplier: float = 2.0
    jitter: bool = True
    
    # Circuit breaker configuration
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout_seconds: int = 60
    half_open_max_calls: int = 3
    
    # Fallback configuration
    enable_fallback: bool = True
    fallback_timeout_ms: int = 5000
    
    # Cache configuration
    use_cached_on_error: bool = True
    cache_ttl_seconds: int = 300


class CircuitState(Enum):
    """Circuit breaker states."""
    
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """
    Circuit breaker for preventing cascading failures.
    
    Monitors failures and opens circuit when threshold is exceeded,
    preventing further attempts until recovery.
    """
    
    name: str
    config: RecoveryConfig
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_state_change: datetime = field(default_factory=datetime.utcnow)
    half_open_calls: int = 0
    
    def is_open(self) -> bool:
        """Check if circuit is open."""
        if self.state == CircuitState.OPEN:
            # Check if timeout has passed
            if self.last_failure_time:
                elapsed = (datetime.utcnow() - self.last_failure_time).total_seconds()
                if elapsed > self.config.timeout_seconds:
                    self._transition_to_half_open()
                    return False
            return True
        return False
    
    def is_closed(self) -> bool:
        """Check if circuit is closed."""
        return self.state == CircuitState.CLOSED
    
    def is_half_open(self) -> bool:
        """Check if circuit is half-open."""
        return self.state == CircuitState.HALF_OPEN
    
    def record_success(self):
        """Record a successful operation."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._transition_to_closed()
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0
    
    def record_failure(self):
        """Record a failed operation."""
        self.last_failure_time = datetime.utcnow()
        
        if self.state == CircuitState.CLOSED:
            self.failure_count += 1
            if self.failure_count >= self.config.failure_threshold:
                self._transition_to_open()
        elif self.state == CircuitState.HALF_OPEN:
            self._transition_to_open()
    
    def allow_request(self) -> bool:
        """Check if request should be allowed."""
        if self.is_open():
            return False
        
        if self.is_half_open():
            if self.half_open_calls >= self.config.half_open_max_calls:
                return False
            self.half_open_calls += 1
        
        return True
    
    def _transition_to_open(self):
        """Transition to open state."""
        logger.warning(f"Circuit breaker '{self.name}' opening after {self.failure_count} failures")
        self.state = CircuitState.OPEN
        self.last_state_change = datetime.utcnow()
        self.failure_count = 0
        self.success_count = 0
    
    def _transition_to_closed(self):
        """Transition to closed state."""
        logger.info(f"Circuit breaker '{self.name}' closing after successful recovery")
        self.state = CircuitState.CLOSED
        self.last_state_change = datetime.utcnow()
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0
    
    def _transition_to_half_open(self):
        """Transition to half-open state."""
        logger.info(f"Circuit breaker '{self.name}' entering half-open state")
        self.state = CircuitState.HALF_OPEN
        self.last_state_change = datetime.utcnow()
        self.success_count = 0
        self.half_open_calls = 0


class RecoveryManager:
    """
    Manages error recovery strategies across the system.
    
    Coordinates retries, fallbacks, circuit breakers, and other
    recovery mechanisms.
    """
    
    def __init__(self, config: Optional[RecoveryConfig] = None):
        """
        Initialize the recovery manager.
        
        Args:
            config: Recovery configuration
        """
        self.config = config or RecoveryConfig()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.fallback_handlers: Dict[Type[Exception], Callable] = {}
        self.recovery_strategies: Dict[Type[Exception], List[RecoveryStrategy]] = {}
        
        # Register default strategies
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Register default recovery strategies."""
        # Retryable errors
        self.register_strategy(RetryableError, [RecoveryStrategy.RETRY])
        self.register_strategy(TimeoutError, [RecoveryStrategy.RETRY, RecoveryStrategy.FALLBACK])
        
        # API errors
        self.register_strategy(APIError, [RecoveryStrategy.RETRY, RecoveryStrategy.CIRCUIT_BREAK])
        
        # Service errors
        self.register_strategy(ServiceError, [RecoveryStrategy.CIRCUIT_BREAK, RecoveryStrategy.FALLBACK])
        
        # Detection errors
        self.register_strategy(DetectionError, [RecoveryStrategy.FALLBACK, RecoveryStrategy.CACHE])
    
    def register_strategy(
        self,
        error_type: Type[Exception],
        strategies: List[RecoveryStrategy]
    ):
        """
        Register recovery strategies for an error type.
        
        Args:
            error_type: Type of error
            strategies: List of recovery strategies to apply
        """
        self.recovery_strategies[error_type] = strategies
    
    def register_fallback(
        self,
        error_type: Type[Exception],
        fallback_handler: Callable[[Exception, ErrorContext], Any]
    ):
        """
        Register a fallback handler for an error type.
        
        Args:
            error_type: Type of error
            fallback_handler: Function to handle fallback
        """
        self.fallback_handlers[error_type] = fallback_handler
    
    def attempt_recovery(
        self,
        error: Exception,
        context: Optional[ErrorContext] = None
    ) -> Optional[Any]:
        """
        Attempt to recover from an error.
        
        Args:
            error: Error to recover from
            context: Error context
            
        Returns:
            Recovery result if successful, None otherwise
        """
        # Get applicable strategies
        strategies = self._get_strategies(error)
        
        for strategy in strategies:
            try:
                result = self._apply_strategy(strategy, error, context)
                if result is not None:
                    logger.info(f"Successfully recovered using {strategy.value} strategy")
                    return result
            except Exception as e:
                logger.warning(f"Recovery strategy {strategy.value} failed: {e}")
                continue
        
        return None
    
    def _get_strategies(self, error: Exception) -> List[RecoveryStrategy]:
        """
        Get applicable recovery strategies for an error.
        
        Args:
            error: Error to get strategies for
            
        Returns:
            List of applicable strategies
        """
        # Check registered strategies
        for error_type, strategies in self.recovery_strategies.items():
            if isinstance(error, error_type):
                return strategies
        
        # Default strategy
        if isinstance(error, RetryableError):
            return [RecoveryStrategy.RETRY]
        
        return []
    
    def _apply_strategy(
        self,
        strategy: RecoveryStrategy,
        error: Exception,
        context: Optional[ErrorContext]
    ) -> Optional[Any]:
        """
        Apply a recovery strategy.
        
        Args:
            strategy: Strategy to apply
            error: Error to recover from
            context: Error context
            
        Returns:
            Recovery result if successful
        """
        if strategy == RecoveryStrategy.RETRY:
            return self._apply_retry(error, context)
        elif strategy == RecoveryStrategy.FALLBACK:
            return self._apply_fallback(error, context)
        elif strategy == RecoveryStrategy.CIRCUIT_BREAK:
            return self._apply_circuit_breaker(error, context)
        elif strategy == RecoveryStrategy.CACHE:
            return self._apply_cache_recovery(error, context)
        elif strategy == RecoveryStrategy.DEGRADE:
            return self._apply_degradation(error, context)
        else:
            return None
    
    def _apply_retry(
        self,
        error: Exception,
        context: Optional[ErrorContext]
    ) -> Optional[Any]:
        """
        Apply retry strategy.
        
        Args:
            error: Error that triggered retry
            context: Error context
            
        Returns:
            Result if retry succeeds
        """
        # This is a placeholder - actual retry logic would be in decorators
        logger.info("Retry strategy would be applied here")
        return None
    
    def _apply_fallback(
        self,
        error: Exception,
        context: Optional[ErrorContext]
    ) -> Optional[Any]:
        """
        Apply fallback strategy.
        
        Args:
            error: Error that triggered fallback
            context: Error context
            
        Returns:
            Fallback result if available
        """
        # Check for registered fallback handler
        for error_type, handler in self.fallback_handlers.items():
            if isinstance(error, error_type):
                try:
                    return handler(error, context)
                except Exception as e:
                    logger.error(f"Fallback handler failed: {e}")
        
        return None
    
    def _apply_circuit_breaker(
        self,
        error: Exception,
        context: Optional[ErrorContext]
    ) -> Optional[Any]:
        """
        Apply circuit breaker strategy.
        
        Args:
            error: Error that triggered circuit breaker
            context: Error context
            
        Returns:
            None (circuit breaker doesn't provide alternative result)
        """
        # Get or create circuit breaker for this context
        breaker_name = context.component if context else "default"
        
        if breaker_name not in self.circuit_breakers:
            self.circuit_breakers[breaker_name] = CircuitBreaker(
                name=breaker_name,
                config=self.config
            )
        
        breaker = self.circuit_breakers[breaker_name]
        breaker.record_failure()
        
        return None
    
    def _apply_cache_recovery(
        self,
        error: Exception,
        context: Optional[ErrorContext]
    ) -> Optional[Any]:
        """
        Apply cache recovery strategy.
        
        Args:
            error: Error that triggered cache recovery
            context: Error context
            
        Returns:
            Cached result if available
        """
        # This would integrate with the caching system
        logger.info("Cache recovery would be applied here")
        return None
    
    def _apply_degradation(
        self,
        error: Exception,
        context: Optional[ErrorContext]
    ) -> Optional[Any]:
        """
        Apply service degradation strategy.
        
        Args:
            error: Error that triggered degradation
            context: Error context
            
        Returns:
            Degraded service result
        """
        # Return simplified/degraded response
        logger.info("Service degradation would be applied here")
        return None
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """
        Get a circuit breaker by name.
        
        Args:
            name: Circuit breaker name
            
        Returns:
            Circuit breaker if exists
        """
        return self.circuit_breakers.get(name)
    
    def reset_circuit_breaker(self, name: str):
        """
        Reset a circuit breaker.
        
        Args:
            name: Circuit breaker name
        """
        if name in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(
                name=name,
                config=self.config
            )
    
    def get_recovery_stats(self) -> Dict[str, Any]:
        """
        Get recovery statistics.
        
        Returns:
            Dictionary of recovery statistics
        """
        return {
            "circuit_breakers": {
                name: {
                    "state": breaker.state.value,
                    "failure_count": breaker.failure_count,
                    "success_count": breaker.success_count
                }
                for name, breaker in self.circuit_breakers.items()
            },
            "config": {
                "max_retries": self.config.max_retries,
                "failure_threshold": self.config.failure_threshold
            }
        }


# Global recovery manager instance
_recovery_manager: Optional[RecoveryManager] = None


def get_recovery_manager() -> RecoveryManager:
    """Get the global recovery manager instance."""
    global _recovery_manager
    if _recovery_manager is None:
        _recovery_manager = RecoveryManager()
    return _recovery_manager