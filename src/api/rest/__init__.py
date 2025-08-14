"""
REST API Module for AI Detector System
FastAPI-based REST API with authentication, validation, and documentation
"""

from .routes import app
from .schemas import *
from .middleware import *
from .auth import *

__all__ = [
    'app',
    # Schemas
    'BaseResponse', 'ErrorResponse', 'ValidationErrorResponse',
    'DetectionRequestModel', 'DetectionResultModel', 'DetectionResponse',
    'BatchDetectionRequestModel', 'BatchDetectionResultModel', 'BatchDetectionResponse',
    'PaginationModel', 'PaginatedDetectionResultsModel', 'PaginatedDetectionResponse',
    'DetectionStatisticsModel', 'StatisticsResponse',
    'TrainingDataModel', 'TrainingRequestModel', 'TrainingResultModel', 'TrainingResponse',
    'HealthCheckResponse', 'DetailedHealthResponse', 'AdminHealthResponse',
    'WebSocketMessage', 'WebSocketDetectionRequest', 'WebSocketDetectionResponse',
    'RateLimitInfo', 'MetricsModel', 'MetricsResponse',
    'DetectionModeEnum', 'TrainingModelTypeEnum', 'HealthStatusEnum',
    
    # Middleware
    'RequestLoggingMiddleware', 'RateLimitingMiddleware', 'ErrorHandlingMiddleware',
    'CacheMiddleware', 'SecurityHeadersMiddleware', 'MetricsMiddleware', 'metrics_middleware',
    
    # Authentication
    'User', 'AuthenticationError', 'AuthorizationError',
    'get_current_user', 'require_auth', 'require_role', 'require_admin',
    'login', 'refresh_access_token', 'api_key_auth', 'user_rate_limiter',
    'check_permission', 'session_manager'
]