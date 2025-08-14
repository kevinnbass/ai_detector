"""
API Module for AI Detector System
RESTful and WebSocket APIs for text analysis
"""

from .rest import *
from .websocket import *

__all__ = [
    # REST API
    'app',  # FastAPI application
    
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
    'check_permission', 'session_manager',
    
    # WebSocket
    'WebSocketConnection', 'ConnectionManager', 'WebSocketHandler',
    'connection_manager', 'websocket_handler',
    'start_connection_cleanup', 'stop_connection_cleanup',
    'websocket_endpoint', 'websocket_admin_endpoint', 'websocket_health_check',
    'WebSocketNotificationService', 'notification_service'
]