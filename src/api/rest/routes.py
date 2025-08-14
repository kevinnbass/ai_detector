"""
RESTful API Routes for AI Detector System
Implements FastAPI-based REST endpoints with proper validation and documentation
"""

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncio
import logging

# Internal imports
from src.core.dependency_injection import create_configured_container, ServiceScope
from src.core.abstractions.presentation_layer import (
    IDetectionController, APIResponse, PaginationRequest, ResponseStatus
)
from src.core.abstractions.business_logic_layer import DetectionRequest, TrainingRequest
from src.api.rest.schemas import *
from src.api.rest.middleware import *
from src.api.rest.auth import get_current_user, require_auth
from src.utils.common import Timer

logger = logging.getLogger(__name__)

# Initialize dependency injection container
container = create_configured_container()
security = HTTPBearer(auto_error=False)

# Create FastAPI app
app = FastAPI(
    title="AI Detector API",
    description="RESTful API for detecting AI-generated text content",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "your-domain.com"]
)

# Custom middleware
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(RateLimitingMiddleware)


# Dependency injection helpers
async def get_service_scope() -> ServiceScope:
    """Get service scope for dependency injection"""
    scope = container.create_scope()
    try:
        yield scope
    finally:
        await scope.dispose_async()


async def get_detection_controller(scope: ServiceScope = Depends(get_service_scope)) -> IDetectionController:
    """Get detection controller from DI container"""
    return scope.get_service(IDetectionController)


# Health check endpoints
@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """Basic health check endpoint"""
    from src.core.dependency_injection.service_configuration import health_check_services
    
    health_status = await health_check_services(container)
    
    return HealthCheckResponse(
        status="healthy" if health_status['overall_healthy'] else "unhealthy",
        timestamp=datetime.utcnow(),
        services=health_status['services'],
        version="1.0.0"
    )


@app.get("/health/detailed", response_model=DetailedHealthResponse, tags=["Health"])
async def detailed_health_check():
    """Detailed health check with service information"""
    from src.core.dependency_injection.service_configuration import (
        health_check_services, validate_service_configuration
    )
    
    # Check service configuration
    config_validation = validate_service_configuration(container)
    
    # Check service health
    health_status = await health_check_services(container)
    
    return DetailedHealthResponse(
        status="healthy" if health_status['overall_healthy'] and config_validation['valid'] else "unhealthy",
        timestamp=datetime.utcnow(),
        services=health_status['services'],
        configuration=config_validation,
        system_info={
            "registered_services": config_validation['registered_services'],
            "memory_usage": "N/A",  # Could add actual memory monitoring
            "uptime": "N/A"
        }
    )


# Detection endpoints
@app.post("/api/v1/detect", response_model=DetectionResponse, tags=["Detection"])
async def detect_text(
    request: DetectionRequestModel,
    background_tasks: BackgroundTasks,
    controller: IDetectionController = Depends(get_detection_controller),
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Detect AI-generated content in text
    
    This endpoint analyzes the provided text using the specified detection mode
    and returns the detection result with confidence score and indicators.
    """
    
    with Timer("API.detect_text") as timer:
        try:
            # Add user context if authenticated
            request_data = request.dict()
            if current_user:
                request_data['user_id'] = current_user.get('user_id')
            
            # Call controller
            api_response = await controller.detect_text(request_data)
            
            # Handle different response statuses
            if api_response.status == ResponseStatus.SUCCESS:
                response = DetectionResponse(
                    success=True,
                    data=api_response.data,
                    message=api_response.message,
                    processing_time=timer.elapsed if hasattr(timer, 'elapsed') else None
                )
                
                # Log successful detection in background
                background_tasks.add_task(
                    log_detection_event,
                    user_id=current_user.get('user_id') if current_user else None,
                    text_length=len(request.text),
                    mode=request.mode,
                    result=api_response.data
                )
                
                return response
                
            elif api_response.status == ResponseStatus.VALIDATION_ERROR:
                raise HTTPException(
                    status_code=422,
                    detail={"message": api_response.message, "errors": api_response.errors}
                )
            elif api_response.status == ResponseStatus.RATE_LIMIT_ERROR:
                raise HTTPException(
                    status_code=429,
                    detail={"message": api_response.message}
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail={"message": api_response.message, "errors": api_response.errors}
                )
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in detect_text: {e}")
            raise HTTPException(
                status_code=500,
                detail={"message": "Internal server error", "error": str(e)}
            )


@app.post("/api/v1/detect/batch", response_model=BatchDetectionResponse, tags=["Detection"])
async def batch_detect_text(
    request: BatchDetectionRequestModel,
    background_tasks: BackgroundTasks,
    controller: IDetectionController = Depends(get_detection_controller),
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Batch detect AI-generated content in multiple texts
    
    Analyzes multiple texts in a single request for efficiency.
    Limited to 50 texts per batch.
    """
    
    with Timer("API.batch_detect") as timer:
        try:
            # Validate batch size
            if len(request.requests) > 50:
                raise HTTPException(
                    status_code=422,
                    detail={"message": "Batch size limited to 50 requests"}
                )
            
            # Add user context to all requests
            request_data = {"requests": []}
            for req in request.requests:
                req_dict = req.dict()
                if current_user:
                    req_dict['user_id'] = current_user.get('user_id')
                request_data["requests"].append(req_dict)
            
            # Call controller
            api_response = await controller.batch_detect(request_data)
            
            if api_response.status == ResponseStatus.SUCCESS:
                response = BatchDetectionResponse(
                    success=True,
                    data=api_response.data,
                    message=api_response.message,
                    processing_time=timer.elapsed if hasattr(timer, 'elapsed') else None
                )
                
                # Log batch detection
                background_tasks.add_task(
                    log_batch_detection_event,
                    user_id=current_user.get('user_id') if current_user else None,
                    batch_size=len(request.requests),
                    results=api_response.data.get('results', [])
                )
                
                return response
            else:
                raise HTTPException(
                    status_code=422 if api_response.status == ResponseStatus.VALIDATION_ERROR else 500,
                    detail={"message": api_response.message, "errors": api_response.errors}
                )
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in batch_detect: {e}")
            raise HTTPException(
                status_code=500,
                detail={"message": "Internal server error", "error": str(e)}
            )


@app.get("/api/v1/history", response_model=PaginatedDetectionResponse, tags=["Detection"])
async def get_detection_history(
    page: int = 1,
    page_size: int = 20,
    sort_by: str = "timestamp",
    sort_desc: bool = True,
    controller: IDetectionController = Depends(get_detection_controller),
    current_user: Dict = Depends(require_auth)
):
    """
    Get user's detection history with pagination
    
    Returns paginated list of previous detection results for the authenticated user.
    """
    
    try:
        # Create pagination request
        pagination = PaginationRequest(
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_desc=sort_desc
        )
        
        # Get history from controller
        api_response = await controller.get_detection_history(
            current_user['user_id'], 
            pagination
        )
        
        if api_response.status == ResponseStatus.SUCCESS:
            return PaginatedDetectionResponse(
                success=True,
                data=api_response.data,
                message=api_response.message
            )
        else:
            raise HTTPException(
                status_code=500,
                detail={"message": api_response.message, "errors": api_response.errors}
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting detection history: {e}")
        raise HTTPException(
            status_code=500,
            detail={"message": "Failed to retrieve detection history"}
        )


@app.get("/api/v1/statistics", response_model=StatisticsResponse, tags=["Analytics"])
async def get_statistics(
    controller: IDetectionController = Depends(get_detection_controller),
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Get detection statistics
    
    Returns overall statistics or user-specific statistics if authenticated.
    """
    
    try:
        user_id = current_user.get('user_id') if current_user else None
        api_response = await controller.get_statistics(user_id)
        
        if api_response.status == ResponseStatus.SUCCESS:
            return StatisticsResponse(
                success=True,
                data=api_response.data,
                message=api_response.message
            )
        else:
            raise HTTPException(
                status_code=500,
                detail={"message": api_response.message}
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail={"message": "Failed to retrieve statistics"}
        )


# Training endpoints (future implementation)
@app.post("/api/v1/train", response_model=TrainingResponse, tags=["Training"])
async def train_model(
    request: TrainingRequestModel,
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(require_auth)
):
    """
    Train a new AI detection model
    
    Requires authentication and special permissions.
    Training is performed asynchronously.
    """
    
    # This would be implemented with training business logic
    raise HTTPException(
        status_code=501,
        detail={"message": "Training endpoint not yet implemented"}
    )


# Admin endpoints
@app.get("/api/v1/admin/health", response_model=AdminHealthResponse, tags=["Admin"])
async def admin_health_check(current_user: Dict = Depends(require_auth)):
    """
    Detailed health check for administrators
    """
    
    # Check if user has admin permissions
    if not current_user.get('is_admin', False):
        raise HTTPException(
            status_code=403,
            detail={"message": "Admin access required"}
        )
    
    from src.core.dependency_injection.service_configuration import (
        health_check_services, validate_service_configuration
    )
    
    try:
        health_status = await health_check_services(container)
        config_validation = validate_service_configuration(container)
        
        return AdminHealthResponse(
            status="healthy" if health_status['overall_healthy'] and config_validation['valid'] else "unhealthy",
            timestamp=datetime.utcnow(),
            services=health_status['services'],
            configuration=config_validation,
            container_info={
                "registered_services": len(container.get_registered_services()),
                "service_types": [s.__name__ for s in container.get_registered_services()]
            }
        )
        
    except Exception as e:
        logger.error(f"Admin health check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={"message": "Health check failed", "error": str(e)}
        )


# Background task functions
async def log_detection_event(user_id: Optional[str], text_length: int, 
                            mode: str, result: Dict[str, Any]):
    """Log detection event for analytics"""
    try:
        # This would integrate with logging/analytics system
        logger.info(f"Detection completed - User: {user_id}, Length: {text_length}, "
                   f"Mode: {mode}, AI: {result.get('is_ai')}, Confidence: {result.get('confidence')}")
    except Exception as e:
        logger.error(f"Failed to log detection event: {e}")


async def log_batch_detection_event(user_id: Optional[str], batch_size: int, 
                                  results: List[Dict[str, Any]]):
    """Log batch detection event for analytics"""
    try:
        ai_count = sum(1 for r in results if r.get('is_ai', False))
        logger.info(f"Batch detection completed - User: {user_id}, Batch size: {batch_size}, "
                   f"AI detected: {ai_count}")
    except Exception as e:
        logger.error(f"Failed to log batch detection event: {e}")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail.get("message", "An error occurred") if isinstance(exc.detail, dict) else str(exc.detail),
            "errors": exc.detail.get("errors", []) if isinstance(exc.detail, dict) else [],
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception in {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error",
            "errors": [str(exc)],
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting AI Detector API...")
    
    # Validate service configuration
    from src.core.dependency_injection.service_configuration import validate_service_configuration
    validation = validate_service_configuration(container)
    
    if not validation['valid']:
        logger.error(f"Service configuration invalid: {validation['errors']}")
        raise RuntimeError("Invalid service configuration")
    
    logger.info(f"API started with {validation['registered_services']} registered services")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down AI Detector API...")
    # Container cleanup would happen here
    logger.info("API shutdown complete")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")