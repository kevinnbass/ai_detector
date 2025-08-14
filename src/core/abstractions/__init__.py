"""
Core Abstractions Module
Provides abstraction layers between different components of the system
"""

# Data Access Layer
from .data_access_layer import (
    QueryOperator, SortDirection, QueryFilter, QuerySort, Query, PagedResult,
    IDataAccessLayer, DataAccessLayer, IDetectionDataAccess, DetectionDataAccess,
    query, detection_query, CommonQueries
)

# Business Logic Layer  
from .business_logic_layer import (
    BusinessRuleError, ValidationResult, DetectionRequest, TrainingRequest,
    IBusinessRuleEngine, BusinessRuleEngine,
    IDetectionBusinessLogic, DetectionBusinessLogic,
    ITrainingBusinessLogic, BusinessLogicFacade
)

# Presentation Layer
from .presentation_layer import (
    ResponseStatus, APIResponse, PaginationRequest, PaginatedResponse,
    IRequestValidator, RequestValidator,
    ISerializer, Serializer,
    IDetectionController, DetectionController,
    ControllerRegistry, controller_registry
)

__all__ = [
    # Data Access Layer
    'QueryOperator', 'SortDirection', 'QueryFilter', 'QuerySort', 'Query', 'PagedResult',
    'IDataAccessLayer', 'DataAccessLayer', 'IDetectionDataAccess', 'DetectionDataAccess',
    'query', 'detection_query', 'CommonQueries',
    
    # Business Logic Layer
    'BusinessRuleError', 'ValidationResult', 'DetectionRequest', 'TrainingRequest',
    'IBusinessRuleEngine', 'BusinessRuleEngine',
    'IDetectionBusinessLogic', 'DetectionBusinessLogic',
    'ITrainingBusinessLogic', 'BusinessLogicFacade',
    
    # Presentation Layer
    'ResponseStatus', 'APIResponse', 'PaginationRequest', 'PaginatedResponse',
    'IRequestValidator', 'RequestValidator',
    'ISerializer', 'Serializer',
    'IDetectionController', 'DetectionController',
    'ControllerRegistry', 'controller_registry'
]