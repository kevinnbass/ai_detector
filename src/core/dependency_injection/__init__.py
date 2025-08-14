"""
Dependency Injection Module
Provides IoC container and service configuration for the AI Detector system
"""

# Core DI container
from .container import (
    ServiceLifetime, ServiceDescriptor, ServiceScope,
    IContainer, Container, ServiceCollection, ServiceProvider,
    Application, inject, injectable, configure_services, container
)

# Service configuration
from .service_configuration import (
    configure_core_services, configure_repositories, configure_data_access_layer,
    configure_factories, configure_business_logic, configure_presentation_layer,
    configure_external_services, configure_application_services, configure_all_services,
    ServiceConfigurator, create_configured_container,
    validate_service_configuration, health_check_services
)

__all__ = [
    # Core DI
    'ServiceLifetime', 'ServiceDescriptor', 'ServiceScope',
    'IContainer', 'Container', 'ServiceCollection', 'ServiceProvider',
    'Application', 'inject', 'injectable', 'configure_services', 'container',
    
    # Service configuration
    'configure_core_services', 'configure_repositories', 'configure_data_access_layer',
    'configure_factories', 'configure_business_logic', 'configure_presentation_layer',
    'configure_external_services', 'configure_application_services', 'configure_all_services',
    'ServiceConfigurator', 'create_configured_container',
    'validate_service_configuration', 'health_check_services'
]