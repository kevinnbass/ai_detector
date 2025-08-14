"""
Error handling decorators for the AI Detector system.

Provides decorators for automatic error handling, retries,
circuit breaking, and other resilience patterns.
"""

import time
import logging
import asyncio
import random
from typing import Optional, Callable, Any, Type, Union, Tuple, Dict
from functools import wraps

from .exceptions import (
    AIDetectorException,
    RetryableError,
    TimeoutError,
    APIError
)
from .handlers import get_error_handler
from .context import ErrorContext, get_error_context
from .recovery import get_recovery_manager, CircuitBreaker, RecoveryConfig


logger = logging.getLogger(__name__)


def handle_errors(
    reraise: bool = True,
    log_errors: bool = True,
    fallback_value: Any = None,
    fallback_function: Optional[Callable] = None,
    error_types: Tuple[Type[Exception], ...] = (Exception,)
):
    """
    Decorator for automatic error handling.
    
    Args:
        reraise: Whether to reraise errors after handling
        log_errors: Whether to log errors
        fallback_value: Static fallback value on error
        fallback_function: Function to call for fallback value
        error_types: Tuple of error types to handle
    
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except error_types as e:
                # Get current context
                context = get_error_context()
                if not context:
                    context = ErrorContext(
                        component=func.__module__,
                        operation=func.__name__
                    )
                
                # Handle the error
                handler = get_error_handler()
                
                try:
                    result = handler.handle_error(
                        e,
                        context=context,
                        reraise=False,
                        attempt_recovery=True
                    )
                    
                    if result is not None:
                        return result
                except Exception:
                    pass
                
                # Apply fallback if configured
                if fallback_function:
                    try:
                        return fallback_function(*args, **kwargs)
                    except Exception as fallback_error:
                        logger.error(f"Fallback function failed: {fallback_error}")
                elif fallback_value is not None:
                    return fallback_value
                
                # Reraise if configured
                if reraise:
                    raise
                
                return None
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except error_types as e:
                # Get current context
                context = get_error_context()
                if not context:
                    context = ErrorContext(
                        component=func.__module__,
                        operation=func.__name__
                    )
                
                # Handle the error
                handler = get_error_handler()
                
                try:
                    result = handler.handle_error(
                        e,
                        context=context,
                        reraise=False,
                        attempt_recovery=True
                    )
                    
                    if result is not None:
                        return result
                except Exception:
                    pass
                
                # Apply fallback if configured
                if fallback_function:
                    try:
                        if asyncio.iscoroutinefunction(fallback_function):
                            return await fallback_function(*args, **kwargs)
                        else:
                            return fallback_function(*args, **kwargs)
                    except Exception as fallback_error:
                        logger.error(f"Fallback function failed: {fallback_error}")
                elif fallback_value is not None:
                    return fallback_value
                
                # Reraise if configured
                if reraise:
                    raise
                
                return None
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator


def retry_on_failure(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_multiplier: float = 2.0,
    jitter: bool = True,
    retry_on: Tuple[Type[Exception], ...] = (RetryableError, TimeoutError, APIError),
    log_retries: bool = True
):
    """
    Decorator for automatic retry with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        backoff_multiplier: Multiplier for exponential backoff
        jitter: Add random jitter to delays
        retry_on: Tuple of exception types to retry on
        log_retries: Whether to log retry attempts
    
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            delay = initial_delay
            
            for attempt in range(max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    
                    # Success - reset circuit breaker if exists
                    recovery_manager = get_recovery_manager()
                    breaker_name = func.__name__
                    breaker = recovery_manager.get_circuit_breaker(breaker_name)
                    if breaker:
                        breaker.record_success()
                    
                    return result
                    
                except retry_on as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        if log_retries:
                            logger.error(f"Max retries ({max_retries}) exceeded for {func.__name__}")
                        raise
                    
                    # Calculate next delay
                    if jitter:
                        actual_delay = delay * (1 + random.random() * 0.1)
                    else:
                        actual_delay = delay
                    
                    if log_retries:
                        logger.warning(
                            f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                            f"after {actual_delay:.2f}s delay. Error: {e}"
                        )
                    
                    time.sleep(actual_delay)
                    
                    # Update delay for next retry
                    delay = min(delay * backoff_multiplier, max_delay)
            
            if last_exception:
                raise last_exception
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            delay = initial_delay
            
            for attempt in range(max_retries + 1):
                try:
                    result = await func(*args, **kwargs)
                    
                    # Success - reset circuit breaker if exists
                    recovery_manager = get_recovery_manager()
                    breaker_name = func.__name__
                    breaker = recovery_manager.get_circuit_breaker(breaker_name)
                    if breaker:
                        breaker.record_success()
                    
                    return result
                    
                except retry_on as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        if log_retries:
                            logger.error(f"Max retries ({max_retries}) exceeded for {func.__name__}")
                        raise
                    
                    # Calculate next delay
                    if jitter:
                        actual_delay = delay * (1 + random.random() * 0.1)
                    else:
                        actual_delay = delay
                    
                    if log_retries:
                        logger.warning(
                            f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                            f"after {actual_delay:.2f}s delay. Error: {e}"
                        )
                    
                    await asyncio.sleep(actual_delay)
                    
                    # Update delay for next retry
                    delay = min(delay * backoff_multiplier, max_delay)
            
            if last_exception:
                raise last_exception
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator


def circuit_breaker(
    failure_threshold: int = 5,
    success_threshold: int = 2,
    timeout: int = 60,
    fallback_function: Optional[Callable] = None,
    fallback_value: Any = None
):
    """
    Decorator for circuit breaker pattern.
    
    Args:
        failure_threshold: Number of failures before opening circuit
        success_threshold: Number of successes to close circuit
        timeout: Seconds before attempting to close circuit
        fallback_function: Function to call when circuit is open
        fallback_value: Value to return when circuit is open
    
    Returns:
        Decorated function
    """
    def decorator(func):
        # Create circuit breaker config
        config = RecoveryConfig(
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            timeout_seconds=timeout
        )
        
        # Get or create circuit breaker
        recovery_manager = get_recovery_manager()
        breaker_name = f"{func.__module__}.{func.__name__}"
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get or create circuit breaker
            breaker = recovery_manager.get_circuit_breaker(breaker_name)
            if not breaker:
                breaker = CircuitBreaker(name=breaker_name, config=config)
                recovery_manager.circuit_breakers[breaker_name] = breaker
            
            # Check if circuit is open
            if breaker.is_open():
                logger.warning(f"Circuit breaker open for {breaker_name}")
                
                if fallback_function:
                    return fallback_function(*args, **kwargs)
                elif fallback_value is not None:
                    return fallback_value
                else:
                    raise AIDetectorException(
                        f"Circuit breaker open for {breaker_name}",
                        error_code="CIRCUIT_BREAKER_OPEN",
                        recovery_suggestion="Service temporarily unavailable. Please try again later."
                    )
            
            # Check if request is allowed (for half-open state)
            if not breaker.allow_request():
                logger.warning(f"Circuit breaker limiting requests for {breaker_name}")
                
                if fallback_function:
                    return fallback_function(*args, **kwargs)
                elif fallback_value is not None:
                    return fallback_value
                else:
                    raise AIDetectorException(
                        f"Circuit breaker limiting requests for {breaker_name}",
                        error_code="CIRCUIT_BREAKER_LIMIT",
                        recovery_suggestion="Too many requests. Please try again later."
                    )
            
            try:
                result = func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure()
                raise
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get or create circuit breaker
            breaker = recovery_manager.get_circuit_breaker(breaker_name)
            if not breaker:
                breaker = CircuitBreaker(name=breaker_name, config=config)
                recovery_manager.circuit_breakers[breaker_name] = breaker
            
            # Check if circuit is open
            if breaker.is_open():
                logger.warning(f"Circuit breaker open for {breaker_name}")
                
                if fallback_function:
                    if asyncio.iscoroutinefunction(fallback_function):
                        return await fallback_function(*args, **kwargs)
                    else:
                        return fallback_function(*args, **kwargs)
                elif fallback_value is not None:
                    return fallback_value
                else:
                    raise AIDetectorException(
                        f"Circuit breaker open for {breaker_name}",
                        error_code="CIRCUIT_BREAKER_OPEN",
                        recovery_suggestion="Service temporarily unavailable. Please try again later."
                    )
            
            # Check if request is allowed (for half-open state)
            if not breaker.allow_request():
                logger.warning(f"Circuit breaker limiting requests for {breaker_name}")
                
                if fallback_function:
                    if asyncio.iscoroutinefunction(fallback_function):
                        return await fallback_function(*args, **kwargs)
                    else:
                        return fallback_function(*args, **kwargs)
                elif fallback_value is not None:
                    return fallback_value
                else:
                    raise AIDetectorException(
                        f"Circuit breaker limiting requests for {breaker_name}",
                        error_code="CIRCUIT_BREAKER_LIMIT",
                        recovery_suggestion="Too many requests. Please try again later."
                    )
            
            try:
                result = await func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure()
                raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator


def timeout(seconds: float):
    """
    Decorator to add timeout to functions.
    
    Args:
        seconds: Timeout in seconds
    
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(
                    f"Function {func.__name__} timed out after {seconds} seconds",
                    operation=func.__name__,
                    timeout_ms=int(seconds * 1000)
                )
            
            # Set the timeout handler
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(seconds))
            
            try:
                result = func(*args, **kwargs)
            finally:
                # Restore the old handler and cancel the alarm
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            
            return result
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=seconds
                )
                return result
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"Function {func.__name__} timed out after {seconds} seconds",
                    operation=func.__name__,
                    timeout_ms=int(seconds * 1000)
                )
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            import platform
            # Signal-based timeout only works on Unix-like systems
            if platform.system() == 'Windows':
                logger.warning(f"Timeout decorator not fully supported on Windows for {func.__name__}")
                return func
            return wrapper
    
    return decorator


def validate_input(**validators: Dict[str, Callable[[Any], bool]]):
    """
    Decorator to validate function inputs.
    
    Args:
        **validators: Dictionary of parameter names to validation functions
    
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate each parameter
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not validator(value):
                        raise ValidationError(
                            f"Validation failed for parameter '{param_name}'",
                            field=param_name,
                            actual_value=value
                        )
            
            return func(*args, **kwargs)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate each parameter
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not validator(value):
                        raise ValidationError(
                            f"Validation failed for parameter '{param_name}'",
                            field=param_name,
                            actual_value=value
                        )
            
            return await func(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator