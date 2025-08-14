"""
Base Service Layer Implementation
Provides abstract base class and common functionality for all services
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import logging
from datetime import datetime
from contextlib import asynccontextmanager

from src.utils.common import Timer, RetryManager
from src.core.repositories.base_repository import IRepository, UnitOfWork

logger = logging.getLogger(__name__)


class ServiceError(Exception):
    """Base service error"""
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(message)


class ValidationError(ServiceError):
    """Validation error"""
    def __init__(self, message: str, field: str = None, value: Any = None):
        super().__init__(message, "VALIDATION_ERROR", {"field": field, "value": value})


class NotFoundError(ServiceError):
    """Entity not found error"""
    def __init__(self, entity_type: str, entity_id: str):
        super().__init__(
            f"{entity_type} with ID {entity_id} not found",
            "NOT_FOUND",
            {"entity_type": entity_type, "entity_id": entity_id}
        )


class BusinessRuleError(ServiceError):
    """Business rule violation error"""
    def __init__(self, rule: str, message: str):
        super().__init__(message, "BUSINESS_RULE_VIOLATION", {"rule": rule})


class IBaseService(ABC):
    """Base service interface"""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize service"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup service resources"""
        pass
    
    @abstractmethod
    def get_service_info(self) -> Dict[str, Any]:
        """Get service information"""
        pass


class BaseService(IBaseService):
    """Base service implementation with common functionality"""
    
    def __init__(self, name: str):
        self.name = name
        self.initialized = False
        self.repositories = {}
        self.retry_manager = RetryManager()
        self.metrics = {
            "operations_count": 0,
            "errors_count": 0,
            "last_operation": None,
            "average_response_time": 0.0,
            "total_response_time": 0.0
        }
        logger.info(f"Initialized service: {name}")
    
    async def initialize(self) -> None:
        """Initialize service"""
        if not self.initialized:
            await self._initialize_dependencies()
            self.initialized = True
            logger.info(f"Service {self.name} initialized successfully")
    
    async def _initialize_dependencies(self) -> None:
        """Initialize service dependencies - override in subclasses"""
        pass
    
    async def cleanup(self) -> None:
        """Cleanup service resources"""
        logger.info(f"Cleaning up service: {self.name}")
        self.initialized = False
    
    def register_repository(self, name: str, repository: IRepository) -> None:
        """Register repository with service"""
        self.repositories[name] = repository
        logger.debug(f"Registered repository {name} with service {self.name}")
    
    def get_repository(self, name: str) -> Optional[IRepository]:
        """Get registered repository"""
        return self.repositories.get(name)
    
    def validate_input(self, data: Dict[str, Any], required_fields: List[str]) -> None:
        """Validate input data"""
        missing_fields = []
        for field in required_fields:
            if field not in data or data[field] is None:
                missing_fields.append(field)
        
        if missing_fields:
            raise ValidationError(f"Missing required fields: {', '.join(missing_fields)}")
    
    def validate_range(self, value: Union[int, float], min_val: float = None, 
                      max_val: float = None, field_name: str = "value") -> None:
        """Validate numeric range"""
        if min_val is not None and value < min_val:
            raise ValidationError(f"{field_name} must be >= {min_val}", field_name, value)
        if max_val is not None and value > max_val:
            raise ValidationError(f"{field_name} must be <= {max_val}", field_name, value)
    
    def validate_enum(self, value: str, valid_values: List[str], field_name: str = "value") -> None:
        """Validate enum value"""
        if value not in valid_values:
            raise ValidationError(
                f"{field_name} must be one of: {', '.join(valid_values)}", 
                field_name, value
            )
    
    @asynccontextmanager
    async def measure_operation(self, operation_name: str):
        """Context manager to measure operation performance"""
        start_time = datetime.utcnow()
        
        try:
            with Timer(f"{self.name}.{operation_name}") as timer:
                yield timer
            
            # Update metrics
            self.metrics["operations_count"] += 1
            self.metrics["last_operation"] = operation_name
            
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            self.metrics["total_response_time"] += elapsed
            self.metrics["average_response_time"] = (
                self.metrics["total_response_time"] / self.metrics["operations_count"]
            )
            
        except Exception as e:
            self.metrics["errors_count"] += 1
            logger.error(f"Error in {self.name}.{operation_name}: {e}")
            raise
    
    async def with_retry(self, operation, *args, **kwargs):
        """Execute operation with retry logic"""
        return await self.retry_manager.execute(operation, *args, **kwargs)
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get service information"""
        return {
            "name": self.name,
            "initialized": self.initialized,
            "repositories": list(self.repositories.keys()),
            "metrics": self.metrics.copy()
        }
    
    def log_business_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Log business event"""
        logger.info(f"Business event in {self.name}: {event_type}", extra={
            "event_type": event_type,
            "service": self.name,
            "details": details
        })
    
    async def check_preconditions(self, operation: str, data: Dict[str, Any]) -> None:
        """Check operation preconditions - override in subclasses"""
        if not self.initialized:
            raise ServiceError(f"Service {self.name} not initialized")
    
    async def execute_with_transaction(self, operation_func, *args, **kwargs):
        """Execute operation within transaction"""
        async with UnitOfWork() as uow:
            # Register repositories with unit of work
            for name, repo in self.repositories.items():
                uow.register_repository(name, repo)
            
            try:
                result = await operation_func(uow, *args, **kwargs)
                await uow.commit()
                return result
            except Exception as e:
                await uow.rollback()
                raise


class ServiceRegistry:
    """Registry for managing services"""
    
    def __init__(self):
        self._services = {}
        self._initialized = False
    
    def register(self, name: str, service: IBaseService) -> None:
        """Register service"""
        self._services[name] = service
        logger.debug(f"Registered service: {name}")
    
    def get(self, name: str) -> Optional[IBaseService]:
        """Get service by name"""
        return self._services.get(name)
    
    def get_all(self) -> Dict[str, IBaseService]:
        """Get all services"""
        return self._services.copy()
    
    async def initialize_all(self) -> None:
        """Initialize all registered services"""
        if self._initialized:
            return
        
        logger.info("Initializing all services...")
        errors = []
        
        for name, service in self._services.items():
            try:
                await service.initialize()
                logger.info(f"Successfully initialized service: {name}")
            except Exception as e:
                error_msg = f"Failed to initialize service {name}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        if errors:
            raise ServiceError(f"Failed to initialize services: {'; '.join(errors)}")
        
        self._initialized = True
        logger.info("All services initialized successfully")
    
    async def cleanup_all(self) -> None:
        """Cleanup all services"""
        logger.info("Cleaning up all services...")
        
        for name, service in self._services.items():
            try:
                await service.cleanup()
                logger.info(f"Successfully cleaned up service: {name}")
            except Exception as e:
                logger.error(f"Error cleaning up service {name}: {e}")
        
        self._initialized = False
        logger.info("All services cleaned up")
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all services"""
        status = {
            "registry_initialized": self._initialized,
            "total_services": len(self._services),
            "services": {}
        }
        
        for name, service in self._services.items():
            status["services"][name] = service.get_service_info()
        
        return status


# Service decorator for automatic error handling and metrics
def service_operation(operation_name: str = None):
    """Decorator for service operations"""
    def decorator(func):
        async def wrapper(self, *args, **kwargs):
            op_name = operation_name or func.__name__
            
            try:
                async with self.measure_operation(op_name):
                    return await func(self, *args, **kwargs)
            except ServiceError:
                raise  # Re-raise service errors as-is
            except Exception as e:
                # Convert unexpected errors to service errors
                logger.error(f"Unexpected error in {self.name}.{op_name}: {e}")
                raise ServiceError(f"Internal error in {op_name}", "INTERNAL_ERROR", {"original_error": str(e)})
        
        return wrapper
    return decorator


# Global service registry instance
service_registry = ServiceRegistry()


__all__ = [
    'ServiceError', 'ValidationError', 'NotFoundError', 'BusinessRuleError',
    'IBaseService', 'BaseService', 'ServiceRegistry', 'service_operation',
    'service_registry'
]