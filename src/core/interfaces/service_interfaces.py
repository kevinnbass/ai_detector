"""
Service Interface Definitions
Interfaces for service layer components
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime
from enum import Enum

from .base_interfaces import (
    IInitializable, IConfigurable, IHealthCheckable, 
    IMetricsProvider, IDisposable, ISecurable
)
from .detector_interfaces import IDetectionResult, IDetectionConfig
from .data_interfaces import DataSample, DataBatch


class ServiceStatus(Enum):
    """Service status enumeration"""
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class CacheStrategy(Enum):
    """Cache strategy enumeration"""
    NO_CACHE = "no_cache"
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    WRITE_THROUGH = "write_through"
    WRITE_BACK = "write_back"


class AuthenticationMethod(Enum):
    """Authentication method enumeration"""
    JWT = "jwt"
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    BASIC = "basic"
    NONE = "none"


class IBaseService(IInitializable, IConfigurable, IHealthCheckable, 
                   IMetricsProvider, IDisposable, ABC):
    """Base interface for all services"""
    
    @abstractmethod
    def get_service_name(self) -> str:
        """Get service name"""
        pass
    
    @abstractmethod
    def get_service_version(self) -> str:
        """Get service version"""
        pass
    
    @abstractmethod
    def get_status(self) -> ServiceStatus:
        """Get current service status"""
        pass
    
    @abstractmethod
    async def start(self) -> bool:
        """Start the service"""
        pass
    
    @abstractmethod
    async def stop(self) -> bool:
        """Stop the service"""
        pass
    
    @abstractmethod
    def get_dependencies(self) -> List[str]:
        """Get service dependencies"""
        pass


class IAnalysisService(IBaseService):
    """Interface for analysis services"""
    
    @abstractmethod
    async def analyze_text(self, text: str, 
                          config: Optional[IDetectionConfig] = None) -> IDetectionResult:
        """Analyze text for AI detection"""
        pass
    
    @abstractmethod
    async def analyze_batch(self, texts: List[str], 
                           config: Optional[IDetectionConfig] = None) -> List[IDetectionResult]:
        """Analyze batch of texts"""
        pass
    
    @abstractmethod
    async def get_analysis_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get analysis history"""
        pass
    
    @abstractmethod
    def get_analysis_stats(self) -> Dict[str, Any]:
        """Get analysis statistics"""
        pass


class IDetectionService(IBaseService):
    """Interface for detection services"""
    
    @abstractmethod
    async def detect(self, text: str, detector_type: Optional[str] = None) -> IDetectionResult:
        """Perform AI detection"""
        pass
    
    @abstractmethod
    async def detect_with_explanation(self, text: str) -> Dict[str, Any]:
        """Detect with explanation"""
        pass
    
    @abstractmethod
    def get_available_detectors(self) -> List[str]:
        """Get available detector types"""
        pass
    
    @abstractmethod
    async def calibrate_detector(self, detector_type: str, 
                                calibration_data: List[DataSample]) -> bool:
        """Calibrate detector with data"""
        pass


class ITrainingService(IBaseService):
    """Interface for training services"""
    
    @abstractmethod
    async def train_model(self, training_data: List[DataSample], 
                         config: Dict[str, Any]) -> str:
        """Train a new model"""
        pass
    
    @abstractmethod
    async def evaluate_model(self, model_id: str, 
                            test_data: List[DataSample]) -> Dict[str, Any]:
        """Evaluate trained model"""
        pass
    
    @abstractmethod
    async def get_training_status(self, job_id: str) -> Dict[str, Any]:
        """Get training job status"""
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get available trained models"""
        pass


class ICacheService(IBaseService):
    """Interface for caching services"""
    
    @abstractmethod
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        pass
    
    @abstractmethod
    async def clear(self, pattern: Optional[str] = None) -> int:
        """Clear cache entries"""
        pass
    
    @abstractmethod
    def get_cache_strategy(self) -> CacheStrategy:
        """Get current cache strategy"""
        pass
    
    @abstractmethod
    def set_cache_strategy(self, strategy: CacheStrategy) -> None:
        """Set cache strategy"""
        pass


class IAuthenticationService(IBaseService, ISecurable):
    """Interface for authentication services"""
    
    @abstractmethod
    async def authenticate(self, credentials: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Authenticate user"""
        pass
    
    @abstractmethod
    async def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate authentication token"""
        pass
    
    @abstractmethod
    async def refresh_token(self, refresh_token: str) -> Optional[str]:
        """Refresh authentication token"""
        pass
    
    @abstractmethod
    async def revoke_token(self, token: str) -> bool:
        """Revoke authentication token"""
        pass
    
    @abstractmethod
    def get_supported_methods(self) -> List[AuthenticationMethod]:
        """Get supported authentication methods"""
        pass


class IConfigurationService(IBaseService):
    """Interface for configuration services"""
    
    @abstractmethod
    async def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        pass
    
    @abstractmethod
    async def set_config(self, key: str, value: Any) -> bool:
        """Set configuration value"""
        pass
    
    @abstractmethod
    async def delete_config(self, key: str) -> bool:
        """Delete configuration key"""
        pass
    
    @abstractmethod
    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration"""
        pass
    
    @abstractmethod
    async def reload_config(self) -> bool:
        """Reload configuration from source"""
        pass
    
    @abstractmethod
    def watch_config(self, key: str, callback: Callable[[Any], None]) -> str:
        """Watch for config changes"""
        pass


class INotificationService(IBaseService):
    """Interface for notification services"""
    
    @abstractmethod
    async def send_notification(self, recipient: str, message: str, 
                               notification_type: str = "info") -> bool:
        """Send notification"""
        pass
    
    @abstractmethod
    async def send_bulk_notification(self, recipients: List[str], 
                                    message: str, 
                                    notification_type: str = "info") -> Dict[str, bool]:
        """Send bulk notifications"""
        pass
    
    @abstractmethod
    def subscribe_to_topic(self, topic: str, recipient: str) -> bool:
        """Subscribe to notification topic"""
        pass
    
    @abstractmethod
    def unsubscribe_from_topic(self, topic: str, recipient: str) -> bool:
        """Unsubscribe from notification topic"""
        pass


class IQueueService(IBaseService):
    """Interface for queue services"""
    
    @abstractmethod
    async def enqueue(self, queue_name: str, item: Any, 
                     priority: int = 0) -> str:
        """Enqueue item"""
        pass
    
    @abstractmethod
    async def dequeue(self, queue_name: str, timeout: Optional[float] = None) -> Any:
        """Dequeue item"""
        pass
    
    @abstractmethod
    def get_queue_size(self, queue_name: str) -> int:
        """Get queue size"""
        pass
    
    @abstractmethod
    async def clear_queue(self, queue_name: str) -> int:
        """Clear queue"""
        pass
    
    @abstractmethod
    def get_queue_stats(self, queue_name: str) -> Dict[str, Any]:
        """Get queue statistics"""
        pass


class ISchedulingService(IBaseService):
    """Interface for scheduling services"""
    
    @abstractmethod
    def schedule_task(self, task: Callable, trigger: str, 
                     **kwargs) -> str:
        """Schedule a task"""
        pass
    
    @abstractmethod
    def cancel_task(self, task_id: str) -> bool:
        """Cancel scheduled task"""
        pass
    
    @abstractmethod
    def get_scheduled_tasks(self) -> List[Dict[str, Any]]:
        """Get all scheduled tasks"""
        pass
    
    @abstractmethod
    def pause_task(self, task_id: str) -> bool:
        """Pause scheduled task"""
        pass
    
    @abstractmethod
    def resume_task(self, task_id: str) -> bool:
        """Resume scheduled task"""
        pass


class ILogService(IBaseService):
    """Interface for logging services"""
    
    @abstractmethod
    async def log(self, level: str, message: str, 
                 component: str, **kwargs) -> None:
        """Log a message"""
        pass
    
    @abstractmethod
    async def get_logs(self, filters: Dict[str, Any], 
                      limit: int = 100) -> List[Dict[str, Any]]:
        """Get logs with filters"""
        pass
    
    @abstractmethod
    def set_log_level(self, component: str, level: str) -> None:
        """Set log level for component"""
        pass
    
    @abstractmethod
    def get_log_stats(self) -> Dict[str, Any]:
        """Get logging statistics"""
        pass


class IServiceRegistry(ABC):
    """Interface for service registry"""
    
    @abstractmethod
    def register_service(self, name: str, service: IBaseService) -> None:
        """Register a service"""
        pass
    
    @abstractmethod
    def unregister_service(self, name: str) -> bool:
        """Unregister a service"""
        pass
    
    @abstractmethod
    def get_service(self, name: str) -> Optional[IBaseService]:
        """Get service by name"""
        pass
    
    @abstractmethod
    def get_all_services(self) -> Dict[str, IBaseService]:
        """Get all registered services"""
        pass
    
    @abstractmethod
    def is_service_available(self, name: str) -> bool:
        """Check if service is available"""
        pass