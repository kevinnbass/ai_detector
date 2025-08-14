"""
Base Interface Definitions
Core interfaces that define fundamental system contracts
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
from datetime import datetime
from enum import Enum


class HealthStatus(Enum):
    """System health status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class LogLevel(Enum):
    """Logging level enumeration"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class IInitializable(ABC):
    """Interface for components that require initialization"""
    
    @abstractmethod
    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize the component
        
        Args:
            config: Optional initialization configuration
            
        Returns:
            True if initialization succeeded, False otherwise
        """
        pass
    
    @abstractmethod
    def is_initialized(self) -> bool:
        """Check if component is initialized"""
        pass


class IConfigurable(ABC):
    """Interface for configurable components"""
    
    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> bool:
        """
        Configure the component
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if configuration succeeded, False otherwise
        """
        pass
    
    @abstractmethod
    def get_configuration(self) -> Dict[str, Any]:
        """Get current configuration"""
        pass
    
    @abstractmethod
    def validate_configuration(self, config: Dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Validate configuration
        
        Returns:
            Tuple of (is_valid, validation_errors)
        """
        pass


class IHealthCheckable(ABC):
    """Interface for components that support health checks"""
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check
        
        Returns:
            Health check result with status and details
        """
        pass
    
    @abstractmethod
    def get_health_status(self) -> HealthStatus:
        """Get current health status"""
        pass


class IMetricsProvider(ABC):
    """Interface for components that provide metrics"""
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get component metrics"""
        pass
    
    @abstractmethod
    def reset_metrics(self) -> None:
        """Reset all metrics"""
        pass
    
    @abstractmethod
    def get_metric(self, metric_name: str) -> Optional[Union[int, float, str]]:
        """Get specific metric value"""
        pass


class ILoggable(ABC):
    """Interface for components with logging capabilities"""
    
    @abstractmethod
    def log(self, level: LogLevel, message: str, **kwargs) -> None:
        """Log a message"""
        pass
    
    @abstractmethod
    def set_log_level(self, level: LogLevel) -> None:
        """Set logging level"""
        pass
    
    @abstractmethod
    def get_log_level(self) -> LogLevel:
        """Get current logging level"""
        pass


class IDisposable(ABC):
    """Interface for components that require cleanup"""
    
    @abstractmethod
    async def dispose(self) -> None:
        """Clean up resources"""
        pass
    
    @abstractmethod
    def is_disposed(self) -> bool:
        """Check if component is disposed"""
        pass


class IValidatable(ABC):
    """Interface for validatable components"""
    
    @abstractmethod
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate component state
        
        Returns:
            Tuple of (is_valid, validation_errors)
        """
        pass


class IVersioned(ABC):
    """Interface for versioned components"""
    
    @abstractmethod
    def get_version(self) -> str:
        """Get component version"""
        pass
    
    @abstractmethod
    def is_compatible(self, required_version: str) -> bool:
        """Check version compatibility"""
        pass


class IMonitorable(ABC):
    """Interface for monitorable components"""
    
    @abstractmethod
    def start_monitoring(self) -> None:
        """Start monitoring"""
        pass
    
    @abstractmethod
    def stop_monitoring(self) -> None:
        """Stop monitoring"""
        pass
    
    @abstractmethod
    def is_monitoring(self) -> bool:
        """Check if monitoring is active"""
        pass
    
    @abstractmethod
    def get_monitoring_data(self) -> Dict[str, Any]:
        """Get monitoring data"""
        pass


class IAsyncContextManager(ABC):
    """Interface for async context managers"""
    
    @abstractmethod
    async def __aenter__(self):
        """Async context entry"""
        pass
    
    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context exit"""
        pass


class IEventEmitter(ABC):
    """Interface for event-emitting components"""
    
    @abstractmethod
    def emit_event(self, event_type: str, data: Any = None) -> None:
        """Emit an event"""
        pass
    
    @abstractmethod
    def subscribe_to_event(self, event_type: str, handler: callable) -> str:
        """Subscribe to events, returns subscription ID"""
        pass
    
    @abstractmethod
    def unsubscribe_from_event(self, subscription_id: str) -> bool:
        """Unsubscribe from events"""
        pass


class IRateLimited(ABC):
    """Interface for rate-limited components"""
    
    @abstractmethod
    def is_rate_limited(self) -> bool:
        """Check if currently rate limited"""
        pass
    
    @abstractmethod
    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Get rate limit information"""
        pass
    
    @abstractmethod
    def reset_rate_limit(self) -> None:
        """Reset rate limit counter"""
        pass


class ICacheable(ABC):
    """Interface for cacheable components"""
    
    @abstractmethod
    def get_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key"""
        pass
    
    @abstractmethod
    def is_cacheable(self, *args, **kwargs) -> bool:
        """Check if result can be cached"""
        pass
    
    @abstractmethod
    def get_cache_ttl(self, *args, **kwargs) -> Optional[int]:
        """Get cache time-to-live in seconds"""
        pass


class IRetryable(ABC):
    """Interface for retryable operations"""
    
    @abstractmethod
    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Check if operation should be retried"""
        pass
    
    @abstractmethod
    def get_retry_delay(self, attempt: int) -> float:
        """Get delay before retry in seconds"""
        pass
    
    @abstractmethod
    def get_max_retries(self) -> int:
        """Get maximum number of retries"""
        pass


class ISecurable(ABC):
    """Interface for securable components"""
    
    @abstractmethod
    def is_authenticated(self) -> bool:
        """Check if authenticated"""
        pass
    
    @abstractmethod
    def is_authorized(self, permission: str) -> bool:
        """Check if authorized for permission"""
        pass
    
    @abstractmethod
    def get_security_context(self) -> Dict[str, Any]:
        """Get security context"""
        pass