"""
Service Configuration for AI Detector System
Configures all services and their dependencies in the IoC container
"""

import os
from typing import Dict, Any

from .container import ServiceCollection, ServiceLifetime, Container
from src.utils.common import Config

# Import all service interfaces and implementations
from src.core.repositories.base_repository import IRepository, FileBasedRepository, InMemoryRepository
from src.core.repositories.detection_repository import IDetectionRepository, DetectionRepository
from src.core.abstractions.data_access_layer import IDataAccessLayer, IDetectionDataAccess, DetectionDataAccess
from src.core.abstractions.business_logic_layer import (
    IBusinessRuleEngine, BusinessRuleEngine,
    IDetectionBusinessLogic, DetectionBusinessLogic,
    ITrainingBusinessLogic, BusinessLogicFacade
)
from src.core.abstractions.presentation_layer import (
    IRequestValidator, RequestValidator,
    ISerializer, Serializer,
    IDetectionController, DetectionController
)
from src.core.factories.detector_factory import IDetectorFactory, DetectorFactory
from src.core.events.event_bus import IEventBus, EventBus
from src.core.services.base_service import ServiceRegistry


def configure_core_services(services: ServiceCollection) -> None:
    """Configure core infrastructure services"""
    
    # Configuration service
    config = Config()
    services.add_instance(Config, config)
    
    # Event bus (singleton)
    services.add_singleton(IEventBus, EventBus)
    
    # Service registry (singleton)
    services.add_singleton(ServiceRegistry, ServiceRegistry)


def configure_repositories(services: ServiceCollection) -> None:
    """Configure repository services"""
    
    # Base repository configuration based on environment
    storage_mode = os.getenv('STORAGE_MODE', 'file')  # file, memory, database
    
    if storage_mode == 'memory':
        # For testing - use in-memory repositories
        services.add_transient(IRepository, InMemoryRepository)
    else:
        # Default to file-based storage
        def create_file_repository():
            storage_path = os.getenv('DATA_STORAGE_PATH', 'data')
            return FileBasedRepository(storage_path, "generic")
        
        services.add_factory(IRepository, create_file_repository, ServiceLifetime.SCOPED)
    
    # Detection repository (scoped - one per request/scope)
    def create_detection_repository():
        storage_path = os.getenv('DETECTION_DATA_PATH', 'data/detections')
        return DetectionRepository(storage_path)
    
    services.add_factory(IDetectionRepository, create_detection_repository, ServiceLifetime.SCOPED)


def configure_data_access_layer(services: ServiceCollection) -> None:
    """Configure data access layer services"""
    
    # Generic data access layer (scoped)
    services.add_scoped(IDataAccessLayer, DataAccessLayer)
    
    # Detection data access (scoped)
    services.add_scoped(IDetectionDataAccess, DetectionDataAccess)


def configure_factories(services: ServiceCollection) -> None:
    """Configure factory services"""
    
    # Detector factory (singleton - stateless)
    services.add_singleton(IDetectorFactory, DetectorFactory)


def configure_business_logic(services: ServiceCollection) -> None:
    """Configure business logic services"""
    
    # Business rule engine (singleton)
    services.add_singleton(IBusinessRuleEngine, BusinessRuleEngine)
    
    # Detection business logic (scoped)
    services.add_scoped(IDetectionBusinessLogic, DetectionBusinessLogic)
    
    # Training business logic would be configured here
    # services.add_scoped(ITrainingBusinessLogic, TrainingBusinessLogic)
    
    # Business logic facade (scoped)
    def create_business_logic_facade(detection_logic: IDetectionBusinessLogic,
                                   training_logic: ITrainingBusinessLogic = None,
                                   rule_engine: IBusinessRuleEngine = None):
        return BusinessLogicFacade(detection_logic, training_logic, rule_engine)
    
    services.add_factory(BusinessLogicFacade, create_business_logic_facade, ServiceLifetime.SCOPED)


def configure_presentation_layer(services: ServiceCollection) -> None:
    """Configure presentation layer services"""
    
    # Request validator (singleton - stateless)
    services.add_singleton(IRequestValidator, RequestValidator)
    
    # Serializer (singleton - stateless)
    services.add_singleton(ISerializer, Serializer)
    
    # Controllers (scoped)
    services.add_scoped(IDetectionController, DetectionController)


def configure_external_services(services: ServiceCollection) -> None:
    """Configure external service integrations"""
    
    # LLM providers would be configured here
    # Gemini API client
    def create_gemini_client():
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable required")
        
        from src.integrations.unified_analyzer import GeminiProvider
        return GeminiProvider(api_key)
    
    # Only register if API key is available
    if os.getenv('GEMINI_API_KEY'):
        from src.integrations.unified_analyzer import GeminiProvider
        services.add_factory(GeminiProvider, create_gemini_client, ServiceLifetime.SINGLETON)


def configure_application_services(services: ServiceCollection) -> None:
    """Configure application-level services"""
    
    # Application configuration
    app_config = {
        'name': 'AI Detector',
        'version': '1.0.0',
        'debug': os.getenv('DEBUG', 'false').lower() == 'true',
        'environment': os.getenv('ENVIRONMENT', 'development')
    }
    
    services.add_instance(Dict[str, Any], app_config)


def configure_all_services(services: ServiceCollection) -> None:
    """Configure all services for the application"""
    
    # Configure in dependency order
    configure_core_services(services)
    configure_repositories(services)
    configure_data_access_layer(services)
    configure_factories(services)
    configure_business_logic(services)
    configure_presentation_layer(services)
    configure_external_services(services)
    configure_application_services(services)


class ServiceConfigurator:
    """Centralized service configuration"""
    
    def __init__(self):
        self.container = Container()
        self._configured = False
    
    def configure(self, additional_config_func=None) -> Container:
        """Configure all services"""
        if self._configured:
            return self.container
        
        services = ServiceCollection(self.container)
        
        # Configure all default services
        configure_all_services(services)
        
        # Apply additional configuration if provided
        if additional_config_func:
            additional_config_func(services)
        
        self._configured = True
        return self.container
    
    def configure_for_testing(self) -> Container:
        """Configure services for testing environment"""
        def test_config(services: ServiceCollection):
            # Override with in-memory implementations for testing
            services.add_transient(IRepository, InMemoryRepository)
            
            # Use mock implementations
            from unittest.mock import Mock
            services.add_instance(Dict[str, Any], {'environment': 'testing'})
        
        return self.configure(test_config)
    
    def configure_for_development(self) -> Container:
        """Configure services for development environment"""
        def dev_config(services: ServiceCollection):
            # Development-specific configuration
            services.add_instance(Dict[str, Any], {
                'environment': 'development',
                'debug': True,
                'verbose_logging': True
            })
        
        return self.configure(dev_config)
    
    def configure_for_production(self) -> Container:
        """Configure services for production environment"""
        def prod_config(services: ServiceCollection):
            # Production-specific configuration
            services.add_instance(Dict[str, Any], {
                'environment': 'production',
                'debug': False,
                'cache_enabled': True,
                'performance_monitoring': True
            })
        
        return self.configure(prod_config)


# Environment-based configuration factory
def create_configured_container(environment: str = None) -> Container:
    """Create container configured for specific environment"""
    env = environment or os.getenv('ENVIRONMENT', 'development')
    configurator = ServiceConfigurator()
    
    if env == 'testing':
        return configurator.configure_for_testing()
    elif env == 'production':
        return configurator.configure_for_production()
    else:
        return configurator.configure_for_development()


# Validation helpers
def validate_service_configuration(container: Container) -> Dict[str, Any]:
    """Validate that all required services are properly configured"""
    validation_results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'registered_services': len(container.get_registered_services())
    }
    
    # Required services for basic functionality
    required_services = [
        IDetectionRepository,
        IDetectionDataAccess,
        IDetectionBusinessLogic,
        IDetectionController,
        IBusinessRuleEngine,
        IDetectorFactory
    ]
    
    for service_type in required_services:
        if not container.is_registered(service_type):
            validation_results['errors'].append(f"Required service {service_type.__name__} not registered")
            validation_results['valid'] = False
    
    # Optional services (warnings if missing)
    optional_services = [
        IEventBus,
        ServiceRegistry
    ]
    
    for service_type in optional_services:
        if not container.is_registered(service_type):
            validation_results['warnings'].append(f"Optional service {service_type.__name__} not registered")
    
    # Test service resolution for critical services
    try:
        with container.create_scope() as scope:
            # Test that we can create main services
            scope.get_service(IDetectionBusinessLogic)
            scope.get_service(IDetectionController)
    except Exception as e:
        validation_results['errors'].append(f"Service resolution failed: {str(e)}")
        validation_results['valid'] = False
    
    return validation_results


# Health check for dependency injection system
async def health_check_services(container: Container) -> Dict[str, Any]:
    """Perform health check on all registered services"""
    health_status = {
        'overall_healthy': True,
        'services': {},
        'timestamp': None
    }
    
    from datetime import datetime
    health_status['timestamp'] = datetime.utcnow().isoformat()
    
    try:
        with container.create_scope() as scope:
            # Check core services
            core_services = [
                IDetectionBusinessLogic,
                IDetectionController,
                IBusinessRuleEngine
            ]
            
            for service_type in core_services:
                service_name = service_type.__name__
                try:
                    instance = scope.get_service(service_type)
                    
                    # Check if service has health check method
                    if hasattr(instance, 'get_health_status'):
                        status = instance.get_health_status()
                    elif hasattr(instance, 'get_service_info'):
                        status = instance.get_service_info()
                    else:
                        status = {'status': 'healthy', 'note': 'No health check method'}
                    
                    health_status['services'][service_name] = {
                        'healthy': True,
                        'details': status
                    }
                    
                except Exception as e:
                    health_status['services'][service_name] = {
                        'healthy': False,
                        'error': str(e)
                    }
                    health_status['overall_healthy'] = False
    
    except Exception as e:
        health_status['overall_healthy'] = False
        health_status['error'] = str(e)
    
    return health_status


__all__ = [
    'configure_core_services', 'configure_repositories', 'configure_data_access_layer',
    'configure_factories', 'configure_business_logic', 'configure_presentation_layer',
    'configure_external_services', 'configure_application_services', 'configure_all_services',
    'ServiceConfigurator', 'create_configured_container',
    'validate_service_configuration', 'health_check_services'
]