"""
Dependency Injection and IoC Container Implementation
Provides dependency injection capabilities for the AI Detector system
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar, Callable, Union
from enum import Enum
import inspect
import asyncio
import logging
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ServiceLifetime(Enum):
    """Service lifetime options"""
    SINGLETON = "singleton"      # Single instance for application lifetime
    SCOPED = "scoped"           # Single instance per scope (e.g., per request)
    TRANSIENT = "transient"     # New instance every time


class ServiceDescriptor:
    """Describes how to create and manage a service"""
    
    def __init__(self, service_type: Type, implementation: Union[Type, Callable],
                 lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT,
                 name: Optional[str] = None):
        self.service_type = service_type
        self.implementation = implementation
        self.lifetime = lifetime
        self.name = name or service_type.__name__
        self.instance = None  # For singleton storage
    
    def __repr__(self):
        return f"ServiceDescriptor({self.service_type.__name__}, {self.lifetime.value})"


class ServiceScope:
    """Represents a dependency injection scope"""
    
    def __init__(self, container: 'Container'):
        self.container = container
        self.scoped_instances = {}
        self._disposed = False
    
    def get_service(self, service_type: Type[T]) -> T:
        """Get service instance from scope"""
        if self._disposed:
            raise RuntimeError("Service scope has been disposed")
        
        descriptor = self.container._get_descriptor(service_type)
        if not descriptor:
            raise ValueError(f"Service {service_type.__name__} not registered")
        
        if descriptor.lifetime == ServiceLifetime.SINGLETON:
            return self.container._get_singleton(descriptor)
        elif descriptor.lifetime == ServiceLifetime.SCOPED:
            return self._get_scoped_instance(descriptor)
        else:  # TRANSIENT
            return self.container._create_instance(descriptor, self)
    
    def _get_scoped_instance(self, descriptor: ServiceDescriptor):
        """Get or create scoped instance"""
        if descriptor.service_type not in self.scoped_instances:
            self.scoped_instances[descriptor.service_type] = self.container._create_instance(descriptor, self)
        return self.scoped_instances[descriptor.service_type]
    
    async def dispose_async(self):
        """Dispose of scoped services asynchronously"""
        if self._disposed:
            return
        
        for instance in self.scoped_instances.values():
            if hasattr(instance, 'dispose_async'):
                await instance.dispose_async()
            elif hasattr(instance, 'dispose'):
                instance.dispose()
        
        self.scoped_instances.clear()
        self._disposed = True
    
    def dispose(self):
        """Dispose of scoped services synchronously"""
        if self._disposed:
            return
        
        for instance in self.scoped_instances.values():
            if hasattr(instance, 'dispose'):
                instance.dispose()
        
        self.scoped_instances.clear()
        self._disposed = True
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dispose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.dispose_async()


class IContainer(ABC):
    """Dependency injection container interface"""
    
    @abstractmethod
    def register(self, service_type: Type, implementation: Union[Type, Callable] = None,
                lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT) -> None:
        """Register service with container"""
        pass
    
    @abstractmethod
    def register_instance(self, service_type: Type, instance: Any) -> None:
        """Register service instance (singleton)"""
        pass
    
    @abstractmethod
    def get_service(self, service_type: Type[T]) -> T:
        """Get service instance"""
        pass
    
    @abstractmethod
    def create_scope(self) -> ServiceScope:
        """Create new service scope"""
        pass
    
    @abstractmethod
    def is_registered(self, service_type: Type) -> bool:
        """Check if service is registered"""
        pass


class Container(IContainer):
    """Main dependency injection container"""
    
    def __init__(self):
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._singletons: Dict[Type, Any] = {}
        self._building = set()  # Track services being built to detect circular dependencies
    
    def register(self, service_type: Type, implementation: Union[Type, Callable] = None,
                lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT) -> None:
        """Register service with container"""
        impl = implementation or service_type
        
        # Validate implementation
        if inspect.isclass(impl):
            if not issubclass(impl, service_type) and service_type != impl:
                # Allow registration of concrete class as itself
                if implementation is None:
                    pass  # Registering concrete class as itself
                else:
                    logger.warning(f"Implementation {impl.__name__} does not inherit from {service_type.__name__}")
        
        descriptor = ServiceDescriptor(service_type, impl, lifetime)
        self._services[service_type] = descriptor
        
        logger.debug(f"Registered {service_type.__name__} with lifetime {lifetime.value}")
    
    def register_singleton(self, service_type: Type, implementation: Union[Type, Callable] = None) -> None:
        """Register service as singleton"""
        self.register(service_type, implementation, ServiceLifetime.SINGLETON)
    
    def register_scoped(self, service_type: Type, implementation: Union[Type, Callable] = None) -> None:
        """Register service as scoped"""
        self.register(service_type, implementation, ServiceLifetime.SCOPED)
    
    def register_transient(self, service_type: Type, implementation: Union[Type, Callable] = None) -> None:
        """Register service as transient"""
        self.register(service_type, implementation, ServiceLifetime.TRANSIENT)
    
    def register_instance(self, service_type: Type, instance: Any) -> None:
        """Register service instance (singleton)"""
        descriptor = ServiceDescriptor(service_type, type(instance), ServiceLifetime.SINGLETON)
        descriptor.instance = instance
        self._services[service_type] = descriptor
        self._singletons[service_type] = instance
        
        logger.debug(f"Registered instance of {service_type.__name__}")
    
    def register_factory(self, service_type: Type, factory: Callable, 
                        lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT) -> None:
        """Register service with factory function"""
        descriptor = ServiceDescriptor(service_type, factory, lifetime)
        self._services[service_type] = descriptor
        
        logger.debug(f"Registered factory for {service_type.__name__}")
    
    def get_service(self, service_type: Type[T]) -> T:
        """Get service instance"""
        descriptor = self._get_descriptor(service_type)
        if not descriptor:
            raise ValueError(f"Service {service_type.__name__} not registered")
        
        if descriptor.lifetime == ServiceLifetime.SINGLETON:
            return self._get_singleton(descriptor)
        else:
            # For non-scoped calls, create transient instances
            return self._create_instance(descriptor, None)
    
    def get_required_service(self, service_type: Type[T]) -> T:
        """Get service instance, raise exception if not found"""
        if not self.is_registered(service_type):
            raise ValueError(f"Required service {service_type.__name__} not registered")
        return self.get_service(service_type)
    
    def try_get_service(self, service_type: Type[T]) -> Optional[T]:
        """Try to get service instance, return None if not found"""
        try:
            return self.get_service(service_type)
        except ValueError:
            return None
    
    def create_scope(self) -> ServiceScope:
        """Create new service scope"""
        return ServiceScope(self)
    
    def is_registered(self, service_type: Type) -> bool:
        """Check if service is registered"""
        return service_type in self._services
    
    def _get_descriptor(self, service_type: Type) -> Optional[ServiceDescriptor]:
        """Get service descriptor"""
        return self._services.get(service_type)
    
    def _get_singleton(self, descriptor: ServiceDescriptor):
        """Get or create singleton instance"""
        if descriptor.service_type not in self._singletons:
            if descriptor.instance is not None:
                self._singletons[descriptor.service_type] = descriptor.instance
            else:
                self._singletons[descriptor.service_type] = self._create_instance(descriptor, None)
        
        return self._singletons[descriptor.service_type]
    
    def _create_instance(self, descriptor: ServiceDescriptor, scope: Optional[ServiceScope]):
        """Create service instance"""
        if descriptor.service_type in self._building:
            raise RuntimeError(f"Circular dependency detected for {descriptor.service_type.__name__}")
        
        self._building.add(descriptor.service_type)
        
        try:
            implementation = descriptor.implementation
            
            # Handle factory functions
            if callable(implementation) and not inspect.isclass(implementation):
                # Factory function
                return self._invoke_factory(implementation, scope)
            
            # Handle class constructors
            if inspect.isclass(implementation):
                return self._create_class_instance(implementation, scope)
            
            raise ValueError(f"Invalid implementation type for {descriptor.service_type.__name__}")
            
        finally:
            self._building.remove(descriptor.service_type)
    
    def _invoke_factory(self, factory: Callable, scope: Optional[ServiceScope]):
        """Invoke factory function with dependency injection"""
        sig = inspect.signature(factory)
        kwargs = {}
        
        for param_name, param in sig.parameters.items():
            if param.annotation and param.annotation != inspect.Parameter.empty:
                dependency = self._resolve_dependency(param.annotation, scope)
                kwargs[param_name] = dependency
        
        return factory(**kwargs)
    
    def _create_class_instance(self, cls: Type, scope: Optional[ServiceScope]):
        """Create class instance with dependency injection"""
        constructor = cls.__init__
        sig = inspect.signature(constructor)
        args = []
        kwargs = {}
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            
            if param.annotation and param.annotation != inspect.Parameter.empty:
                dependency = self._resolve_dependency(param.annotation, scope)
                if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                    args.append(dependency)
                else:
                    kwargs[param_name] = dependency
            elif param.default == inspect.Parameter.empty:
                raise ValueError(f"Cannot resolve parameter '{param_name}' for {cls.__name__}: no type annotation")
        
        return cls(*args, **kwargs)
    
    def _resolve_dependency(self, service_type: Type, scope: Optional[ServiceScope]):
        """Resolve dependency"""
        if scope:
            return scope.get_service(service_type)
        else:
            return self.get_service(service_type)
    
    def get_registered_services(self) -> List[Type]:
        """Get list of registered service types"""
        return list(self._services.keys())
    
    def clear(self):
        """Clear all registrations"""
        # Dispose singletons
        for instance in self._singletons.values():
            if hasattr(instance, 'dispose'):
                instance.dispose()
        
        self._services.clear()
        self._singletons.clear()
        self._building.clear()
        
        logger.info("Container cleared")


# Decorators for dependency injection

def inject(container: Container):
    """Decorator to inject dependencies into function/method"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            sig = inspect.signature(func)
            injected_kwargs = {}
            
            for param_name, param in sig.parameters.items():
                if param_name not in kwargs and param.annotation != inspect.Parameter.empty:
                    try:
                        dependency = container.get_service(param.annotation)
                        injected_kwargs[param_name] = dependency
                    except ValueError:
                        # Service not registered, skip injection
                        pass
            
            return func(*args, **kwargs, **injected_kwargs)
        return wrapper
    return decorator


def injectable(lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT):
    """Class decorator to mark as injectable service"""
    def decorator(cls):
        cls._di_lifetime = lifetime
        return cls
    return decorator


# Global container instance
container = Container()


# Service registration helpers

class ServiceCollection:
    """Helper class for fluent service registration"""
    
    def __init__(self, container: Container):
        self.container = container
    
    def add_singleton(self, service_type: Type, implementation: Union[Type, Callable] = None) -> 'ServiceCollection':
        """Add singleton service"""
        self.container.register_singleton(service_type, implementation)
        return self
    
    def add_scoped(self, service_type: Type, implementation: Union[Type, Callable] = None) -> 'ServiceCollection':
        """Add scoped service"""
        self.container.register_scoped(service_type, implementation)
        return self
    
    def add_transient(self, service_type: Type, implementation: Union[Type, Callable] = None) -> 'ServiceCollection':
        """Add transient service"""
        self.container.register_transient(service_type, implementation)
        return self
    
    def add_instance(self, service_type: Type, instance: Any) -> 'ServiceCollection':
        """Add service instance"""
        self.container.register_instance(service_type, instance)
        return self
    
    def add_factory(self, service_type: Type, factory: Callable,
                   lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT) -> 'ServiceCollection':
        """Add service factory"""
        self.container.register_factory(service_type, factory, lifetime)
        return self


def configure_services(container: Container = None) -> ServiceCollection:
    """Create service collection for configuration"""
    return ServiceCollection(container or globals()['container'])


# Application startup helper

class ServiceProvider:
    """Service provider for application"""
    
    def __init__(self, container: Container):
        self.container = container
        self.root_scope = None
    
    def get_service(self, service_type: Type[T]) -> T:
        """Get service from root scope"""
        if not self.root_scope:
            self.root_scope = self.container.create_scope()
        return self.root_scope.get_service(service_type)
    
    def create_scope(self) -> ServiceScope:
        """Create new scope"""
        return self.container.create_scope()
    
    async def shutdown_async(self):
        """Shutdown service provider"""
        if self.root_scope:
            await self.root_scope.dispose_async()
    
    def shutdown(self):
        """Shutdown service provider"""
        if self.root_scope:
            self.root_scope.dispose()


class Application:
    """Application with dependency injection"""
    
    def __init__(self):
        self.container = Container()
        self.service_provider = None
        self.configured = False
    
    def configure_services(self, configure_func: Callable[[ServiceCollection], None]):
        """Configure services"""
        services = ServiceCollection(self.container)
        configure_func(services)
        self.configured = True
    
    def build(self) -> ServiceProvider:
        """Build service provider"""
        if not self.configured:
            raise RuntimeError("Services not configured. Call configure_services first.")
        
        self.service_provider = ServiceProvider(self.container)
        return self.service_provider
    
    async def run_async(self, startup_func: Callable[[ServiceProvider], None] = None):
        """Run application asynchronously"""
        if not self.service_provider:
            self.build()
        
        try:
            if startup_func:
                await startup_func(self.service_provider)
            
            # Application would continue running here
            
        finally:
            await self.service_provider.shutdown_async()


__all__ = [
    'ServiceLifetime', 'ServiceDescriptor', 'ServiceScope',
    'IContainer', 'Container', 'ServiceCollection', 'ServiceProvider',
    'Application', 'inject', 'injectable', 'configure_services', 'container'
]