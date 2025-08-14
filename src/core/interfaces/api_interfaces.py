"""
API Interface Definitions
Interfaces for API layer components
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable, AsyncGenerator
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

from .base_interfaces import IInitializable, IConfigurable, IValidatable, IHealthCheckable


class HTTPMethod(Enum):
    """HTTP method enumeration"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class APIVersion(Enum):
    """API version enumeration"""
    V1 = "v1"
    V2 = "v2"
    BETA = "beta"


class ResponseFormat(Enum):
    """Response format enumeration"""
    JSON = "json"
    XML = "xml"
    YAML = "yaml"
    PLAIN_TEXT = "plain_text"
    BINARY = "binary"


@dataclass
class APIRequest:
    """Standardized API request"""
    method: HTTPMethod
    path: str
    headers: Dict[str, str]
    params: Optional[Dict[str, Any]] = None
    body: Optional[Any] = None
    user_id: Optional[str] = None
    timestamp: Optional[datetime] = None


@dataclass
class APIResponse:
    """Standardized API response"""
    status_code: int
    headers: Dict[str, str]
    body: Any
    format: ResponseFormat = ResponseFormat.JSON
    timestamp: Optional[datetime] = None
    processing_time: Optional[float] = None


class IAPIClient(IInitializable, IConfigurable, ABC):
    """Interface for API clients"""
    
    @abstractmethod
    async def request(self, method: HTTPMethod, endpoint: str, 
                     data: Optional[Any] = None, 
                     headers: Optional[Dict[str, str]] = None,
                     params: Optional[Dict[str, Any]] = None) -> APIResponse:
        """Make API request"""
        pass
    
    @abstractmethod
    async def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> APIResponse:
        """Make GET request"""
        pass
    
    @abstractmethod
    async def post(self, endpoint: str, data: Any, 
                  headers: Optional[Dict[str, str]] = None) -> APIResponse:
        """Make POST request"""
        pass
    
    @abstractmethod
    async def put(self, endpoint: str, data: Any,
                 headers: Optional[Dict[str, str]] = None) -> APIResponse:
        """Make PUT request"""
        pass
    
    @abstractmethod
    async def delete(self, endpoint: str) -> APIResponse:
        """Make DELETE request"""
        pass
    
    @abstractmethod
    def set_base_url(self, url: str) -> None:
        """Set base URL"""
        pass
    
    @abstractmethod
    def set_authentication(self, auth_type: str, credentials: Dict[str, str]) -> None:
        """Set authentication credentials"""
        pass
    
    @abstractmethod
    def get_request_stats(self) -> Dict[str, Any]:
        """Get request statistics"""
        pass


class IAPIHandler(ABC):
    """Interface for API request handlers"""
    
    @abstractmethod
    async def handle(self, request: APIRequest) -> APIResponse:
        """Handle API request"""
        pass
    
    @abstractmethod
    def can_handle(self, request: APIRequest) -> bool:
        """Check if can handle request"""
        pass
    
    @abstractmethod
    def get_supported_methods(self) -> List[HTTPMethod]:
        """Get supported HTTP methods"""
        pass
    
    @abstractmethod
    def get_endpoint_pattern(self) -> str:
        """Get endpoint pattern"""
        pass


class IRequestValidator(IValidatable, ABC):
    """Interface for request validators"""
    
    @abstractmethod
    def validate_request(self, request: APIRequest) -> tuple[bool, List[str]]:
        """Validate API request"""
        pass
    
    @abstractmethod
    def validate_headers(self, headers: Dict[str, str]) -> tuple[bool, List[str]]:
        """Validate request headers"""
        pass
    
    @abstractmethod
    def validate_params(self, params: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate request parameters"""
        pass
    
    @abstractmethod
    def validate_body(self, body: Any, content_type: str) -> tuple[bool, List[str]]:
        """Validate request body"""
        pass
    
    @abstractmethod
    def get_validation_schema(self, endpoint: str, method: HTTPMethod) -> Dict[str, Any]:
        """Get validation schema"""
        pass


class IResponseFormatter(ABC):
    """Interface for response formatters"""
    
    @abstractmethod
    def format_response(self, data: Any, format: ResponseFormat) -> APIResponse:
        """Format response data"""
        pass
    
    @abstractmethod
    def format_error(self, error: Exception, request: APIRequest) -> APIResponse:
        """Format error response"""
        pass
    
    @abstractmethod
    def format_validation_error(self, errors: List[str], request: APIRequest) -> APIResponse:
        """Format validation error response"""
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[ResponseFormat]:
        """Get supported response formats"""
        pass


class IMiddleware(ABC):
    """Interface for API middleware"""
    
    @abstractmethod
    async def process_request(self, request: APIRequest) -> Optional[APIRequest]:
        """Process incoming request"""
        pass
    
    @abstractmethod
    async def process_response(self, response: APIResponse, 
                              request: APIRequest) -> Optional[APIResponse]:
        """Process outgoing response"""
        pass
    
    @abstractmethod
    def get_middleware_name(self) -> str:
        """Get middleware name"""
        pass
    
    @abstractmethod
    def get_priority(self) -> int:
        """Get middleware priority (higher = earlier execution)"""
        pass


class IRateLimiter(IMiddleware):
    """Interface for rate limiting middleware"""
    
    @abstractmethod
    def is_rate_limited(self, request: APIRequest) -> bool:
        """Check if request is rate limited"""
        pass
    
    @abstractmethod
    def get_rate_limit_info(self, user_id: str) -> Dict[str, Any]:
        """Get rate limit information"""
        pass
    
    @abstractmethod
    def reset_rate_limit(self, user_id: str) -> None:
        """Reset rate limit for user"""
        pass
    
    @abstractmethod
    def set_rate_limit(self, endpoint: str, limit: int, window: int) -> None:
        """Set rate limit for endpoint"""
        pass


class IAuthenticationMiddleware(IMiddleware):
    """Interface for authentication middleware"""
    
    @abstractmethod
    async def authenticate_request(self, request: APIRequest) -> Optional[Dict[str, Any]]:
        """Authenticate request"""
        pass
    
    @abstractmethod
    def is_authenticated(self, request: APIRequest) -> bool:
        """Check if request is authenticated"""
        pass
    
    @abstractmethod
    def extract_user_info(self, request: APIRequest) -> Optional[Dict[str, Any]]:
        """Extract user information from request"""
        pass


class IWebSocketHandler(IInitializable, ABC):
    """Interface for WebSocket handlers"""
    
    @abstractmethod
    async def on_connect(self, websocket: Any) -> bool:
        """Handle WebSocket connection"""
        pass
    
    @abstractmethod
    async def on_disconnect(self, websocket: Any) -> None:
        """Handle WebSocket disconnection"""
        pass
    
    @abstractmethod
    async def on_message(self, websocket: Any, message: Dict[str, Any]) -> None:
        """Handle WebSocket message"""
        pass
    
    @abstractmethod
    async def send_message(self, websocket: Any, message: Dict[str, Any]) -> None:
        """Send WebSocket message"""
        pass
    
    @abstractmethod
    def get_connection_count(self) -> int:
        """Get active connection count"""
        pass


class IWebSocketConnectionManager(ABC):
    """Interface for WebSocket connection management"""
    
    @abstractmethod
    async def add_connection(self, websocket: Any, user_id: Optional[str] = None) -> str:
        """Add WebSocket connection"""
        pass
    
    @abstractmethod
    async def remove_connection(self, connection_id: str) -> None:
        """Remove WebSocket connection"""
        pass
    
    @abstractmethod
    async def broadcast(self, message: Dict[str, Any], 
                       filter_func: Optional[Callable] = None) -> int:
        """Broadcast message to connections"""
        pass
    
    @abstractmethod
    async def send_to_user(self, user_id: str, message: Dict[str, Any]) -> bool:
        """Send message to specific user"""
        pass
    
    @abstractmethod
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        pass


class IAPIDocumentationGenerator(ABC):
    """Interface for API documentation generation"""
    
    @abstractmethod
    def generate_openapi_spec(self) -> Dict[str, Any]:
        """Generate OpenAPI specification"""
        pass
    
    @abstractmethod
    def generate_markdown_docs(self) -> str:
        """Generate markdown documentation"""
        pass
    
    @abstractmethod
    def add_endpoint_documentation(self, endpoint: str, method: HTTPMethod, 
                                  documentation: Dict[str, Any]) -> None:
        """Add endpoint documentation"""
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Get supported documentation formats"""
        pass


class IAPIVersionManager(ABC):
    """Interface for API version management"""
    
    @abstractmethod
    def get_current_version(self) -> APIVersion:
        """Get current API version"""
        pass
    
    @abstractmethod
    def get_supported_versions(self) -> List[APIVersion]:
        """Get supported API versions"""
        pass
    
    @abstractmethod
    def is_version_supported(self, version: APIVersion) -> bool:
        """Check if version is supported"""
        pass
    
    @abstractmethod
    def get_version_from_request(self, request: APIRequest) -> APIVersion:
        """Extract version from request"""
        pass
    
    @abstractmethod
    def route_request(self, request: APIRequest, version: APIVersion) -> IAPIHandler:
        """Route request to appropriate handler"""
        pass


class IAPIMonitor(ABC):
    """Interface for API monitoring"""
    
    @abstractmethod
    def record_request(self, request: APIRequest, response: APIResponse) -> None:
        """Record API request/response"""
        pass
    
    @abstractmethod
    def get_request_metrics(self, time_window: int = 3600) -> Dict[str, Any]:
        """Get request metrics"""
        pass
    
    @abstractmethod
    def get_error_metrics(self, time_window: int = 3600) -> Dict[str, Any]:
        """Get error metrics"""
        pass
    
    @abstractmethod
    def get_performance_metrics(self, time_window: int = 3600) -> Dict[str, Any]:
        """Get performance metrics"""
        pass
    
    @abstractmethod
    def set_alert_threshold(self, metric: str, threshold: float) -> None:
        """Set alert threshold"""
        pass


class IAPIGateway(IInitializable, IConfigurable, IHealthCheckable, ABC):
    """Interface for API gateway"""
    
    @abstractmethod
    async def route_request(self, request: APIRequest) -> APIResponse:
        """Route API request"""
        pass
    
    @abstractmethod
    def add_route(self, pattern: str, handler: IAPIHandler) -> None:
        """Add route to gateway"""
        pass
    
    @abstractmethod
    def remove_route(self, pattern: str) -> bool:
        """Remove route from gateway"""
        pass
    
    @abstractmethod
    def add_middleware(self, middleware: IMiddleware) -> None:
        """Add middleware to gateway"""
        pass
    
    @abstractmethod
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics"""
        pass