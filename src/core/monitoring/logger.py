"""
Structured logging system for the AI Detector.

Provides consistent, structured logging with support for
JSON output, log aggregation, and correlation IDs.
"""

import logging
import json
import sys
import os
from typing import Optional, Dict, Any, Union, List
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import uuid
import traceback
from contextvars import ContextVar

# Context variables for request tracking
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
session_id_var: ContextVar[Optional[str]] = ContextVar('session_id', default=None)


@dataclass
class LoggerConfig:
    """Configuration for the logging system."""
    
    # Log levels
    level: str = "INFO"
    console_level: str = "INFO"
    file_level: str = "DEBUG"
    
    # Output configuration
    log_to_console: bool = True
    log_to_file: bool = True
    log_dir: str = "logs"
    log_file: str = "ai_detector.log"
    
    # Format configuration
    use_json_format: bool = True
    include_timestamp: bool = True
    include_context: bool = True
    include_location: bool = True
    
    # Rotation configuration
    max_file_size_mb: int = 10
    backup_count: int = 5
    rotation_interval: str = "midnight"  # For time-based rotation
    
    # Performance configuration
    async_logging: bool = False
    buffer_size: int = 1024
    
    # Additional fields
    service_name: str = "ai-detector"
    environment: str = "development"
    version: str = "1.0.0"


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter for structured logging.
    
    Outputs logs in JSON format with consistent structure
    and additional context information.
    """
    
    def __init__(self, config: LoggerConfig):
        """
        Initialize the structured formatter.
        
        Args:
            config: Logger configuration
        """
        super().__init__()
        self.config = config
        
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as structured JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            Formatted log string
        """
        # Build base log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() if self.config.include_timestamp else None,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.config.service_name,
            "environment": self.config.environment,
            "version": self.config.version
        }
        
        # Add context information
        if self.config.include_context:
            log_entry.update({
                "request_id": request_id_var.get(),
                "user_id": user_id_var.get(),
                "session_id": session_id_var.get(),
                "correlation_id": getattr(record, 'correlation_id', None)
            })
        
        # Add location information
        if self.config.include_location:
            log_entry.update({
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
                "file": record.pathname
            })
        
        # Add extra fields from record
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info) if record.exc_info else None
            }
        
        # Add performance metrics if present
        if hasattr(record, 'duration_ms'):
            log_entry["performance"] = {
                "duration_ms": record.duration_ms
            }
        
        # Remove None values for cleaner output
        log_entry = {k: v for k, v in log_entry.items() if v is not None}
        
        if self.config.use_json_format:
            return json.dumps(log_entry, default=str)
        else:
            # Fallback to human-readable format
            parts = [f"[{log_entry.get('timestamp', '')}]"]
            parts.append(f"[{log_entry['level']}]")
            if log_entry.get('request_id'):
                parts.append(f"[{log_entry['request_id']}]")
            parts.append(f"{log_entry['logger']}: {log_entry['message']}")
            
            if log_entry.get('exception'):
                parts.append(f"\nException: {log_entry['exception']['type']}: {log_entry['exception']['message']}")
                
            return " ".join(parts)


class StructuredLogger:
    """
    Enhanced logger with structured logging capabilities.
    
    Provides additional methods for logging with context
    and structured data.
    """
    
    def __init__(self, name: str, config: Optional[LoggerConfig] = None):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
            config: Logger configuration
        """
        self.logger = logging.getLogger(name)
        self.config = config or LoggerConfig()
        self._setup_logger()
    
    def _setup_logger(self):
        """Set up logger with handlers and formatters."""
        # Set base level
        self.logger.setLevel(getattr(logging, self.config.level))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = StructuredFormatter(self.config)
        
        # Console handler
        if self.config.log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, self.config.console_level))
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if self.config.log_to_file:
            # Create log directory if it doesn't exist
            log_dir = Path(self.config.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            log_file = log_dir / self.config.log_file
            
            # Use rotating file handler
            file_handler = RotatingFileHandler(
                filename=str(log_file),
                maxBytes=self.config.max_file_size_mb * 1024 * 1024,
                backupCount=self.config.backup_count
            )
            file_handler.setLevel(getattr(logging, self.config.file_level))
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with optional extra fields."""
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message with optional extra fields."""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with optional extra fields."""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, exc_info: bool = False, **kwargs):
        """Log error message with optional exception info."""
        self._log(logging.ERROR, message, exc_info=exc_info, **kwargs)
    
    def critical(self, message: str, exc_info: bool = False, **kwargs):
        """Log critical message with optional exception info."""
        self._log(logging.CRITICAL, message, exc_info=exc_info, **kwargs)
    
    def _log(self, level: int, message: str, exc_info: bool = False, **kwargs):
        """
        Internal logging method with structured data support.
        
        Args:
            level: Log level
            message: Log message
            exc_info: Whether to include exception info
            **kwargs: Additional fields to include
        """
        extra = {'extra_fields': kwargs} if kwargs else {}
        
        # Add any performance metrics
        if 'duration_ms' in kwargs:
            extra['duration_ms'] = kwargs.pop('duration_ms')
        
        self.logger.log(level, message, exc_info=exc_info, extra=extra)
    
    def log_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
        **kwargs
    ):
        """
        Log HTTP request with structured data.
        
        Args:
            method: HTTP method
            path: Request path
            status_code: Response status code
            duration_ms: Request duration
            **kwargs: Additional request data
        """
        self.info(
            f"{method} {path} - {status_code}",
            method=method,
            path=path,
            status_code=status_code,
            duration_ms=duration_ms,
            **kwargs
        )
    
    def log_performance(
        self,
        operation: str,
        duration_ms: float,
        success: bool = True,
        **kwargs
    ):
        """
        Log performance metrics.
        
        Args:
            operation: Operation name
            duration_ms: Operation duration
            success: Whether operation succeeded
            **kwargs: Additional metrics
        """
        level = logging.INFO if success else logging.WARNING
        self._log(
            level,
            f"Performance: {operation}",
            operation=operation,
            duration_ms=duration_ms,
            success=success,
            **kwargs
        )
    
    def log_error_with_context(
        self,
        error: Exception,
        context: Dict[str, Any],
        **kwargs
    ):
        """
        Log error with additional context.
        
        Args:
            error: Exception to log
            context: Error context
            **kwargs: Additional fields
        """
        self.error(
            f"Error: {error}",
            exc_info=True,
            error_type=error.__class__.__name__,
            error_context=context,
            **kwargs
        )


class RequestLogger:
    """
    Context manager for request-scoped logging.
    
    Automatically sets and clears request context for
    consistent request tracking.
    """
    
    def __init__(
        self,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """
        Initialize request logger context.
        
        Args:
            request_id: Request identifier
            user_id: User identifier
            session_id: Session identifier
        """
        self.request_id = request_id or str(uuid.uuid4())
        self.user_id = user_id
        self.session_id = session_id
        self._tokens = []
    
    def __enter__(self):
        """Enter request context."""
        self._tokens.append(request_id_var.set(self.request_id))
        if self.user_id:
            self._tokens.append(user_id_var.set(self.user_id))
        if self.session_id:
            self._tokens.append(session_id_var.set(self.session_id))
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit request context."""
        for token in self._tokens:
            try:
                request_id_var.reset(token)
            except:
                pass


# Global logger registry
_loggers: Dict[str, StructuredLogger] = {}
_global_config: Optional[LoggerConfig] = None


def setup_logging(config: Optional[LoggerConfig] = None):
    """
    Set up global logging configuration.
    
    Args:
        config: Logger configuration
    """
    global _global_config
    _global_config = config or LoggerConfig()
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, _global_config.level))
    
    # Clear existing handlers on root
    root_logger.handlers.clear()
    
    # Apply configuration to existing loggers
    for logger in _loggers.values():
        logger.config = _global_config
        logger._setup_logger()


def get_logger(name: str) -> StructuredLogger:
    """
    Get or create a structured logger.
    
    Args:
        name: Logger name
        
    Returns:
        Structured logger instance
    """
    if name not in _loggers:
        config = _global_config or LoggerConfig()
        _loggers[name] = StructuredLogger(name, config)
    
    return _loggers[name]


# Convenience function for module-level logging
def get_module_logger() -> StructuredLogger:
    """
    Get logger for the calling module.
    
    Returns:
        Structured logger for the module
    """
    import inspect
    frame = inspect.currentframe()
    if frame and frame.f_back:
        module = frame.f_back.f_globals.get('__name__', 'unknown')
        return get_logger(module)
    return get_logger('unknown')