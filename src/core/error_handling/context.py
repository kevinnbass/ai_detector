"""
Error context management for the AI Detector system.

Provides structured context information for errors, enabling
better debugging and error tracking across components.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from dataclasses import dataclass, field, asdict
import uuid


@dataclass
class ErrorContext:
    """
    Structured context information for error handling.
    
    Captures relevant context when errors occur, facilitating
    debugging and error analysis.
    """
    
    # Request context
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # Component context
    component: Optional[str] = None
    operation: Optional[str] = None
    method: Optional[str] = None
    
    # Data context
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Timing context
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_ms: Optional[float] = None
    
    # Environment context
    environment: Optional[str] = None
    version: Optional[str] = None
    host: Optional[str] = None
    
    # Error chain context
    parent_error_id: Optional[str] = None
    error_chain: List[str] = field(default_factory=list)
    
    # Additional context
    tags: List[str] = field(default_factory=list)
    custom_fields: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert context to dictionary format.
        
        Returns:
            Dictionary representation of context
        """
        data = asdict(self)
        
        # Convert datetime to ISO format
        if self.timestamp:
            data["timestamp"] = self.timestamp.isoformat()
        
        # Remove None values for cleaner output
        return {k: v for k, v in data.items() if v is not None}
    
    def add_tag(self, tag: str):
        """
        Add a tag to the context.
        
        Args:
            tag: Tag to add
        """
        if tag not in self.tags:
            self.tags.append(tag)
    
    def add_custom_field(self, key: str, value: Any):
        """
        Add a custom field to the context.
        
        Args:
            key: Field key
            value: Field value
        """
        self.custom_fields[key] = value
    
    def add_to_error_chain(self, error_id: str):
        """
        Add an error to the error chain.
        
        Args:
            error_id: Error identifier to add
        """
        self.error_chain.append(error_id)
    
    def merge(self, other: 'ErrorContext') -> 'ErrorContext':
        """
        Merge another context into this one.
        
        Args:
            other: Context to merge
            
        Returns:
            New merged context
        """
        merged_data = self.to_dict()
        other_data = other.to_dict()
        
        # Merge dictionaries
        for key, value in other_data.items():
            if key in merged_data:
                if isinstance(value, dict) and isinstance(merged_data[key], dict):
                    merged_data[key].update(value)
                elif isinstance(value, list) and isinstance(merged_data[key], list):
                    merged_data[key].extend(value)
                else:
                    # Other values override
                    merged_data[key] = value
            else:
                merged_data[key] = value
        
        # Create new context from merged data
        return ErrorContext(**merged_data)
    
    @classmethod
    def from_request(
        cls,
        request_data: Dict[str, Any],
        component: str = None,
        operation: str = None
    ) -> 'ErrorContext':
        """
        Create context from request data.
        
        Args:
            request_data: Request data dictionary
            component: Component handling the request
            operation: Operation being performed
            
        Returns:
            ErrorContext instance
        """
        return cls(
            request_id=request_data.get("request_id", str(uuid.uuid4())),
            user_id=request_data.get("user_id"),
            session_id=request_data.get("session_id"),
            component=component,
            operation=operation,
            input_data=request_data,
            metadata=request_data.get("metadata", {})
        )
    
    @classmethod
    def from_extension_message(
        cls,
        message: Dict[str, Any],
        component: str = None
    ) -> 'ErrorContext':
        """
        Create context from Chrome extension message.
        
        Args:
            message: Extension message dictionary
            component: Component handling the message
            
        Returns:
            ErrorContext instance
        """
        return cls(
            request_id=message.get("id", str(uuid.uuid4())),
            component=component or message.get("source"),
            operation=message.get("type"),
            input_data=message.get("payload"),
            metadata={
                "source": message.get("source"),
                "target": message.get("target"),
                "correlation_id": message.get("correlation_id")
            },
            tags=[f"extension_{message.get('type', 'unknown').lower()}"]
        )


class ContextManager:
    """
    Manages error context throughout the application lifecycle.
    
    Provides thread-local context storage and propagation.
    """
    
    def __init__(self):
        """Initialize the context manager."""
        import threading
        self._local = threading.local()
        self._global_context: Optional[ErrorContext] = None
    
    def set_context(self, context: ErrorContext):
        """
        Set the current error context.
        
        Args:
            context: Context to set
        """
        self._local.context = context
    
    def get_context(self) -> Optional[ErrorContext]:
        """
        Get the current error context.
        
        Returns:
            Current context or None
        """
        return getattr(self._local, 'context', None) or self._global_context
    
    def clear_context(self):
        """Clear the current error context."""
        if hasattr(self._local, 'context'):
            delattr(self._local, 'context')
    
    def set_global_context(self, context: ErrorContext):
        """
        Set global context that applies to all threads.
        
        Args:
            context: Global context to set
        """
        self._global_context = context
    
    def with_context(self, **kwargs):
        """
        Context manager for temporarily setting error context.
        
        Args:
            **kwargs: Context fields to set
            
        Returns:
            Context manager
        """
        class ContextScope:
            def __init__(scope_self, manager: ContextManager, **context_kwargs):
                scope_self.manager = manager
                scope_self.new_context = ErrorContext(**context_kwargs)
                scope_self.old_context = None
            
            def __enter__(scope_self):
                scope_self.old_context = scope_self.manager.get_context()
                
                # Merge with existing context if present
                if scope_self.old_context:
                    scope_self.new_context = scope_self.old_context.merge(scope_self.new_context)
                
                scope_self.manager.set_context(scope_self.new_context)
                return scope_self.new_context
            
            def __exit__(scope_self, exc_type, exc_val, exc_tb):
                # Restore old context
                if scope_self.old_context:
                    scope_self.manager.set_context(scope_self.old_context)
                else:
                    scope_self.manager.clear_context()
        
        return ContextScope(self, **kwargs)


# Global context manager instance
_context_manager: Optional[ContextManager] = None


def get_context_manager() -> ContextManager:
    """Get the global context manager instance."""
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextManager()
    return _context_manager


def set_error_context(**kwargs):
    """
    Set error context using the global manager.
    
    Args:
        **kwargs: Context fields to set
    """
    context = ErrorContext(**kwargs)
    get_context_manager().set_context(context)


def get_error_context() -> Optional[ErrorContext]:
    """
    Get current error context from the global manager.
    
    Returns:
        Current error context or None
    """
    return get_context_manager().get_context()


def with_error_context(**kwargs):
    """
    Context manager for temporarily setting error context.
    
    Args:
        **kwargs: Context fields to set
        
    Returns:
        Context manager
    """
    return get_context_manager().with_context(**kwargs)