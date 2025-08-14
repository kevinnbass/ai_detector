"""
API-specific error handlers for the AI Detector system.

Provides FastAPI exception handlers and middleware for consistent
error handling at the API boundary.
"""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError, HTTPException
from starlette.exceptions import HTTPException as StarletteHTTPException
import logging
from typing import Optional, Dict, Any
import traceback

from src.core.error_handling import (
    AIDetectorException,
    ValidationError,
    APIError,
    DetectionError,
    ServiceError,
    TimeoutError,
    SecurityError,
    ResourceError,
    BoundaryErrorHandler,
    ErrorContext
)
from src.utils.schema_validator import get_validator, SchemaType


logger = logging.getLogger(__name__)


class APIErrorHandler:
    """
    Handles errors at the API boundary with proper formatting
    and client-safe responses.
    """
    
    def __init__(self, app: FastAPI):
        """
        Initialize API error handler.
        
        Args:
            app: FastAPI application instance
        """
        self.app = app
        self.boundary_handler = BoundaryErrorHandler(
            boundary_type="api",
            sanitize_errors=True,
            include_stack_traces=False
        )
        
        # Register exception handlers
        self._register_handlers()
    
    def _register_handlers(self):
        """Register exception handlers with the FastAPI app."""
        
        # FastAPI validation errors
        self.app.add_exception_handler(
            RequestValidationError,
            self.handle_validation_error
        )
        
        # HTTP exceptions
        self.app.add_exception_handler(
            HTTPException,
            self.handle_http_exception
        )
        
        self.app.add_exception_handler(
            StarletteHTTPException,
            self.handle_http_exception
        )
        
        # Custom exceptions
        self.app.add_exception_handler(
            ValidationError,
            self.handle_custom_validation_error
        )
        
        self.app.add_exception_handler(
            APIError,
            self.handle_api_error
        )
        
        self.app.add_exception_handler(
            DetectionError,
            self.handle_detection_error
        )
        
        self.app.add_exception_handler(
            ServiceError,
            self.handle_service_error
        )
        
        self.app.add_exception_handler(
            TimeoutError,
            self.handle_timeout_error
        )
        
        self.app.add_exception_handler(
            SecurityError,
            self.handle_security_error
        )
        
        self.app.add_exception_handler(
            ResourceError,
            self.handle_resource_error
        )
        
        self.app.add_exception_handler(
            AIDetectorException,
            self.handle_ai_detector_exception
        )
        
        # Generic exception handler
        self.app.add_exception_handler(
            Exception,
            self.handle_generic_exception
        )
    
    async def handle_validation_error(
        self,
        request: Request,
        exc: RequestValidationError
    ) -> JSONResponse:
        """
        Handle FastAPI request validation errors.
        
        Args:
            request: FastAPI request
            exc: Validation exception
            
        Returns:
            JSON error response
        """
        # Create context from request
        context = await self._create_context(request)
        
        # Transform to our validation error
        validation_error = ValidationError(
            message="Request validation failed",
            details={
                "validation_errors": exc.errors(),
                "body": exc.body if hasattr(exc, 'body') else None
            }
        )
        
        # Handle through boundary handler
        error_response = self.boundary_handler.handle_boundary_error(
            validation_error,
            context,
            format_for_client=True
        )
        
        # Validate against schema
        self._validate_error_response(error_response)
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=error_response
        )
    
    async def handle_http_exception(
        self,
        request: Request,
        exc: HTTPException
    ) -> JSONResponse:
        """
        Handle HTTP exceptions.
        
        Args:
            request: FastAPI request
            exc: HTTP exception
            
        Returns:
            JSON error response
        """
        # Create context from request
        context = await self._create_context(request)
        
        # Transform to API error
        api_error = APIError(
            message=exc.detail if hasattr(exc, 'detail') else str(exc),
            status_code=exc.status_code if hasattr(exc, 'status_code') else 500,
            endpoint=str(request.url.path)
        )
        
        # Handle through boundary handler
        error_response = self.boundary_handler.handle_boundary_error(
            api_error,
            context,
            format_for_client=True
        )
        
        # Validate against schema
        self._validate_error_response(error_response)
        
        return JSONResponse(
            status_code=exc.status_code if hasattr(exc, 'status_code') else 500,
            content=error_response
        )
    
    async def handle_custom_validation_error(
        self,
        request: Request,
        exc: ValidationError
    ) -> JSONResponse:
        """
        Handle custom validation errors.
        
        Args:
            request: FastAPI request
            exc: Validation error
            
        Returns:
            JSON error response
        """
        context = await self._create_context(request)
        
        error_response = self.boundary_handler.handle_boundary_error(
            exc,
            context,
            format_for_client=True
        )
        
        self._validate_error_response(error_response)
        
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=error_response
        )
    
    async def handle_api_error(
        self,
        request: Request,
        exc: APIError
    ) -> JSONResponse:
        """
        Handle API errors.
        
        Args:
            request: FastAPI request
            exc: API error
            
        Returns:
            JSON error response
        """
        context = await self._create_context(request)
        
        error_response = self.boundary_handler.handle_boundary_error(
            exc,
            context,
            format_for_client=True
        )
        
        self._validate_error_response(error_response)
        
        # Determine status code
        status_code = exc.details.get("status_code", status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        return JSONResponse(
            status_code=status_code,
            content=error_response
        )
    
    async def handle_detection_error(
        self,
        request: Request,
        exc: DetectionError
    ) -> JSONResponse:
        """
        Handle detection errors.
        
        Args:
            request: FastAPI request
            exc: Detection error
            
        Returns:
            JSON error response
        """
        context = await self._create_context(request)
        
        error_response = self.boundary_handler.handle_boundary_error(
            exc,
            context,
            format_for_client=True
        )
        
        self._validate_error_response(error_response)
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_response
        )
    
    async def handle_service_error(
        self,
        request: Request,
        exc: ServiceError
    ) -> JSONResponse:
        """
        Handle service errors.
        
        Args:
            request: FastAPI request
            exc: Service error
            
        Returns:
            JSON error response
        """
        context = await self._create_context(request)
        
        error_response = self.boundary_handler.handle_boundary_error(
            exc,
            context,
            format_for_client=True
        )
        
        self._validate_error_response(error_response)
        
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=error_response
        )
    
    async def handle_timeout_error(
        self,
        request: Request,
        exc: TimeoutError
    ) -> JSONResponse:
        """
        Handle timeout errors.
        
        Args:
            request: FastAPI request
            exc: Timeout error
            
        Returns:
            JSON error response
        """
        context = await self._create_context(request)
        
        error_response = self.boundary_handler.handle_boundary_error(
            exc,
            context,
            format_for_client=True
        )
        
        self._validate_error_response(error_response)
        
        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content=error_response
        )
    
    async def handle_security_error(
        self,
        request: Request,
        exc: SecurityError
    ) -> JSONResponse:
        """
        Handle security errors.
        
        Args:
            request: FastAPI request
            exc: Security error
            
        Returns:
            JSON error response
        """
        context = await self._create_context(request)
        
        error_response = self.boundary_handler.handle_boundary_error(
            exc,
            context,
            format_for_client=True
        )
        
        self._validate_error_response(error_response)
        
        # Determine status code based on security type
        security_type = exc.details.get("security_type", "")
        if "authentication" in security_type.lower():
            status_code = status.HTTP_401_UNAUTHORIZED
        elif "authorization" in security_type.lower():
            status_code = status.HTTP_403_FORBIDDEN
        else:
            status_code = status.HTTP_403_FORBIDDEN
        
        return JSONResponse(
            status_code=status_code,
            content=error_response
        )
    
    async def handle_resource_error(
        self,
        request: Request,
        exc: ResourceError
    ) -> JSONResponse:
        """
        Handle resource errors.
        
        Args:
            request: FastAPI request
            exc: Resource error
            
        Returns:
            JSON error response
        """
        context = await self._create_context(request)
        
        error_response = self.boundary_handler.handle_boundary_error(
            exc,
            context,
            format_for_client=True
        )
        
        # Add rate limit headers if applicable
        headers = {}
        if exc.details.get("resource_type") == "rate_limit":
            headers["X-RateLimit-Limit"] = str(exc.details.get("limit", 0))
            headers["X-RateLimit-Remaining"] = str(exc.details.get("remaining", 0))
            if exc.details.get("reset_time"):
                headers["X-RateLimit-Reset"] = exc.details["reset_time"]
        
        self._validate_error_response(error_response)
        
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content=error_response,
            headers=headers
        )
    
    async def handle_ai_detector_exception(
        self,
        request: Request,
        exc: AIDetectorException
    ) -> JSONResponse:
        """
        Handle generic AI Detector exceptions.
        
        Args:
            request: FastAPI request
            exc: AI Detector exception
            
        Returns:
            JSON error response
        """
        context = await self._create_context(request)
        
        error_response = self.boundary_handler.handle_boundary_error(
            exc,
            context,
            format_for_client=True
        )
        
        self._validate_error_response(error_response)
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_response
        )
    
    async def handle_generic_exception(
        self,
        request: Request,
        exc: Exception
    ) -> JSONResponse:
        """
        Handle generic unhandled exceptions.
        
        Args:
            request: FastAPI request
            exc: Generic exception
            
        Returns:
            JSON error response
        """
        # Log the full exception for debugging
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        
        context = await self._create_context(request)
        
        # Wrap in AIDetectorException
        wrapped_error = AIDetectorException(
            message="An unexpected error occurred",
            error_code="INTERNAL_ERROR",
            details={
                "exception_type": exc.__class__.__name__,
                "exception_message": str(exc)[:200]  # Truncate for safety
            },
            recovery_suggestion="Please try again later or contact support if the issue persists"
        )
        
        error_response = self.boundary_handler.handle_boundary_error(
            wrapped_error,
            context,
            format_for_client=True
        )
        
        self._validate_error_response(error_response)
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_response
        )
    
    async def _create_context(self, request: Request) -> ErrorContext:
        """
        Create error context from request.
        
        Args:
            request: FastAPI request
            
        Returns:
            Error context
        """
        # Get request body if available
        try:
            body = await request.json() if request.method in ["POST", "PUT", "PATCH"] else None
        except:
            body = None
        
        return ErrorContext(
            request_id=request.headers.get("X-Request-ID", None),
            user_id=request.headers.get("X-User-ID", None),
            session_id=request.headers.get("X-Session-ID", None),
            component="api",
            operation=request.url.path,
            method=request.method,
            input_data=body,
            metadata={
                "path": str(request.url.path),
                "query_params": dict(request.query_params),
                "headers": dict(request.headers),
                "client": request.client.host if request.client else None
            },
            tags=["api_request"]
        )
    
    def _validate_error_response(self, response: Dict[str, Any]):
        """
        Validate error response against schema.
        
        Args:
            response: Error response to validate
        """
        try:
            validator = get_validator()
            result = validator.validate_api_error(response)
            
            if not result.is_valid:
                logger.warning(f"Invalid error response format: {result.errors}")
        except Exception as e:
            logger.error(f"Failed to validate error response: {e}")


def setup_error_handling(app: FastAPI) -> APIErrorHandler:
    """
    Set up error handling for a FastAPI application.
    
    Args:
        app: FastAPI application instance
        
    Returns:
        Configured API error handler
    """
    return APIErrorHandler(app)