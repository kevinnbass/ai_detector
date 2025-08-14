"""
Pydantic Schemas for REST API
Defines request/response models with validation and documentation
"""

from pydantic import BaseModel, Field, validator, root_validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum


# Enums for API models
class DetectionModeEnum(str, Enum):
    """Detection modes"""
    pattern = "pattern"
    llm = "llm" 
    hybrid = "hybrid"


class TrainingModelTypeEnum(str, Enum):
    """Training model types"""
    basic = "basic"
    enhanced = "enhanced"
    ensemble = "ensemble"


class HealthStatusEnum(str, Enum):
    """Health status values"""
    healthy = "healthy"
    unhealthy = "unhealthy"
    degraded = "degraded"


# Base models
class BaseResponse(BaseModel):
    """Base response model"""
    success: bool = Field(description="Whether the request was successful")
    message: str = Field(default="", description="Response message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


class ErrorResponse(BaseResponse):
    """Error response model"""
    success: bool = Field(default=False)
    errors: List[str] = Field(default_factory=list, description="List of error messages")
    error_code: Optional[str] = Field(None, description="Error code for programmatic handling")


class ValidationErrorResponse(ErrorResponse):
    """Validation error response"""
    field_errors: Dict[str, List[str]] = Field(default_factory=dict, description="Field-specific errors")


# Detection request/response models
class DetectionRequestModel(BaseModel):
    """Request model for text detection"""
    text: str = Field(..., min_length=1, max_length=10000, description="Text to analyze")
    mode: DetectionModeEnum = Field(default=DetectionModeEnum.hybrid, description="Detection mode")
    source: Optional[str] = Field(None, max_length=50, description="Source of the text")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty or whitespace only')
        return v.strip()
    
    @validator('metadata')
    def metadata_size_limit(cls, v):
        if len(str(v)) > 1000:  # Reasonable limit for metadata size
            raise ValueError('Metadata too large')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "text": "This is a sample text that I want to analyze for AI generation.",
                "mode": "hybrid",
                "source": "twitter",
                "metadata": {"url": "https://example.com/tweet/123"}
            }
        }


class DetectionResultModel(BaseModel):
    """Detection result model"""
    id: Optional[str] = Field(None, description="Result ID")
    text: str = Field(..., description="Analyzed text (may be truncated)")
    is_ai: bool = Field(..., description="Whether text is AI-generated")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")
    mode: DetectionModeEnum = Field(..., description="Detection mode used")
    indicators: List[str] = Field(default_factory=list, description="Detected AI indicators")
    timestamp: Optional[datetime] = Field(None, description="Detection timestamp")
    processing_time: Optional[float] = Field(None, ge=0, description="Processing time in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class DetectionResponse(BaseResponse):
    """Response model for detection"""
    data: DetectionResultModel = Field(..., description="Detection result")
    processing_time: Optional[float] = Field(None, description="Total processing time")


# Batch detection models
class BatchDetectionRequestModel(BaseModel):
    """Batch detection request model"""
    requests: List[DetectionRequestModel] = Field(..., min_items=1, max_items=50, description="List of detection requests")
    
    @validator('requests')
    def validate_batch_size(cls, v):
        if len(v) > 50:
            raise ValueError('Batch size limited to 50 requests')
        return v


class BatchDetectionResultModel(BaseModel):
    """Batch detection result"""
    results: List[DetectionResultModel] = Field(..., description="Detection results")
    total_processed: int = Field(..., description="Number of texts processed")
    failed_count: int = Field(default=0, description="Number of failed detections")


class BatchDetectionResponse(BaseResponse):
    """Batch detection response"""
    data: BatchDetectionResultModel = Field(..., description="Batch detection results")
    processing_time: Optional[float] = Field(None, description="Total processing time")


# Pagination models
class PaginationModel(BaseModel):
    """Pagination metadata"""
    page: int = Field(..., ge=1, description="Current page number")
    page_size: int = Field(..., ge=1, le=100, description="Items per page")
    total_count: int = Field(..., ge=0, description="Total number of items")
    total_pages: int = Field(..., ge=0, description="Total number of pages")
    has_next: bool = Field(..., description="Whether there are more pages")
    has_previous: bool = Field(..., description="Whether there are previous pages")


class PaginatedDetectionResultsModel(BaseModel):
    """Paginated detection results"""
    items: List[DetectionResultModel] = Field(..., description="Detection results")
    pagination: PaginationModel = Field(..., description="Pagination information")


class PaginatedDetectionResponse(BaseResponse):
    """Paginated detection response"""
    data: PaginatedDetectionResultsModel = Field(..., description="Paginated results")


# Statistics models
class DetectionStatisticsModel(BaseModel):
    """Detection statistics"""
    total_detections: int = Field(..., ge=0, description="Total number of detections")
    ai_detected: int = Field(..., ge=0, description="Number of AI texts detected")
    human_detected: int = Field(..., ge=0, description="Number of human texts detected")
    ai_percentage: float = Field(..., ge=0, le=100, description="Percentage of AI texts")
    average_confidence: float = Field(..., ge=0, le=1, description="Average confidence score")
    recent_24h: Optional[int] = Field(None, ge=0, description="Detections in last 24 hours")
    confidence_distribution: Dict[str, int] = Field(default_factory=dict, description="Confidence distribution")
    statistics_by_mode: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Statistics by mode")


class StatisticsResponse(BaseResponse):
    """Statistics response"""
    data: DetectionStatisticsModel = Field(..., description="Detection statistics")


# Training models (future implementation)
class TrainingDataModel(BaseModel):
    """Training data sample"""
    text: str = Field(..., min_length=10, max_length=5000, description="Training text")
    label: int = Field(..., ge=0, le=1, description="Label (0=human, 1=AI)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Sample metadata")


class TrainingRequestModel(BaseModel):
    """Training request model"""
    training_data: List[TrainingDataModel] = Field(..., min_items=10, description="Training samples")
    model_type: TrainingModelTypeEnum = Field(default=TrainingModelTypeEnum.enhanced, description="Model type")
    validation_split: float = Field(default=0.2, ge=0.1, le=0.5, description="Validation split ratio")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Model hyperparameters")
    
    @validator('training_data')
    def validate_training_data_balance(cls, v):
        if len(v) < 10:
            raise ValueError('Minimum 10 training samples required')
        
        ai_count = sum(1 for sample in v if sample.label == 1)
        human_count = len(v) - ai_count
        
        if ai_count == 0 or human_count == 0:
            raise ValueError('Training data must contain both AI and human samples')
        
        # Check balance - warn if very imbalanced
        ratio = min(ai_count, human_count) / max(ai_count, human_count)
        if ratio < 0.2:
            raise ValueError('Training data is too imbalanced (ratio < 0.2)')
        
        return v


class TrainingResultModel(BaseModel):
    """Training result model"""
    model_id: str = Field(..., description="Trained model ID")
    model_type: str = Field(..., description="Model type")
    training_samples: int = Field(..., description="Number of training samples")
    validation_samples: int = Field(..., description="Number of validation samples")
    metrics: Dict[str, float] = Field(..., description="Training metrics")
    training_time: float = Field(..., description="Training time in seconds")
    timestamp: datetime = Field(..., description="Training completion timestamp")


class TrainingResponse(BaseResponse):
    """Training response"""
    data: TrainingResultModel = Field(..., description="Training result")


# Health check models
class ServiceHealthModel(BaseModel):
    """Individual service health"""
    healthy: bool = Field(..., description="Whether service is healthy")
    details: Dict[str, Any] = Field(default_factory=dict, description="Service details")
    error: Optional[str] = Field(None, description="Error message if unhealthy")


class HealthCheckResponse(BaseResponse):
    """Basic health check response"""
    status: HealthStatusEnum = Field(..., description="Overall health status")
    services: Dict[str, ServiceHealthModel] = Field(default_factory=dict, description="Individual service health")
    version: str = Field(..., description="API version")


class SystemInfoModel(BaseModel):
    """System information"""
    registered_services: int = Field(..., description="Number of registered services")
    memory_usage: str = Field(..., description="Memory usage")
    uptime: str = Field(..., description="System uptime")


class ConfigurationValidationModel(BaseModel):
    """Configuration validation result"""
    valid: bool = Field(..., description="Whether configuration is valid")
    errors: List[str] = Field(default_factory=list, description="Configuration errors")
    warnings: List[str] = Field(default_factory=list, description="Configuration warnings")
    registered_services: int = Field(..., description="Number of registered services")


class DetailedHealthResponse(BaseResponse):
    """Detailed health check response"""
    status: HealthStatusEnum = Field(..., description="Overall health status")
    services: Dict[str, ServiceHealthModel] = Field(..., description="Service health details")
    configuration: ConfigurationValidationModel = Field(..., description="Configuration validation")
    system_info: SystemInfoModel = Field(..., description="System information")


# Admin models
class ContainerInfoModel(BaseModel):
    """DI Container information"""
    registered_services: int = Field(..., description="Number of registered services")
    service_types: List[str] = Field(..., description="List of service types")


class AdminHealthResponse(DetailedHealthResponse):
    """Admin health check response with container info"""
    container_info: ContainerInfoModel = Field(..., description="Container information")


# WebSocket models (for future real-time features)
class WebSocketMessage(BaseModel):
    """WebSocket message model"""
    type: str = Field(..., description="Message type")
    data: Any = Field(..., description="Message data")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = Field(None, description="Correlation ID")


class WebSocketDetectionRequest(WebSocketMessage):
    """WebSocket detection request"""
    type: str = Field(default="detection_request")
    data: DetectionRequestModel = Field(..., description="Detection request data")


class WebSocketDetectionResponse(WebSocketMessage):
    """WebSocket detection response"""
    type: str = Field(default="detection_response")
    data: DetectionResultModel = Field(..., description="Detection result")


# Rate limiting models
class RateLimitInfo(BaseModel):
    """Rate limit information"""
    limit: int = Field(..., description="Rate limit")
    remaining: int = Field(..., description="Remaining requests")
    reset_time: datetime = Field(..., description="Reset time")
    retry_after: Optional[int] = Field(None, description="Retry after seconds")


# Monitoring and metrics models
class MetricsModel(BaseModel):
    """API metrics"""
    total_requests: int = Field(..., description="Total requests")
    successful_requests: int = Field(..., description="Successful requests") 
    failed_requests: int = Field(..., description="Failed requests")
    average_response_time: float = Field(..., description="Average response time")
    requests_per_minute: float = Field(..., description="Requests per minute")


class MetricsResponse(BaseResponse):
    """Metrics response"""
    data: MetricsModel = Field(..., description="API metrics")


# File upload models (for future features)
class FileUploadModel(BaseModel):
    """File upload information"""
    filename: str = Field(..., description="Original filename")
    content_type: str = Field(..., description="Content type")
    size: int = Field(..., description="File size in bytes")
    upload_time: datetime = Field(default_factory=datetime.utcnow)


# Export models
class ExportRequestModel(BaseModel):
    """Export request model"""
    format: str = Field(..., regex="^(json|csv|xlsx)$", description="Export format")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Export filters")
    include_metadata: bool = Field(default=True, description="Include metadata")


class ExportResponse(BaseResponse):
    """Export response"""
    data: Dict[str, Any] = Field(..., description="Export information")
    download_url: str = Field(..., description="Download URL")
    expires_at: datetime = Field(..., description="URL expiration time")


# Configuration update models (for dynamic configuration)
class ConfigUpdateModel(BaseModel):
    """Configuration update model"""
    key: str = Field(..., description="Configuration key")
    value: Any = Field(..., description="Configuration value")
    environment: Optional[str] = Field(None, description="Target environment")


class ConfigUpdateResponse(BaseResponse):
    """Configuration update response"""
    data: Dict[str, Any] = Field(..., description="Updated configuration")


__all__ = [
    # Base models
    'BaseResponse', 'ErrorResponse', 'ValidationErrorResponse',
    
    # Detection models
    'DetectionRequestModel', 'DetectionResultModel', 'DetectionResponse',
    'BatchDetectionRequestModel', 'BatchDetectionResultModel', 'BatchDetectionResponse',
    
    # Pagination models  
    'PaginationModel', 'PaginatedDetectionResultsModel', 'PaginatedDetectionResponse',
    
    # Statistics models
    'DetectionStatisticsModel', 'StatisticsResponse',
    
    # Training models
    'TrainingDataModel', 'TrainingRequestModel', 'TrainingResultModel', 'TrainingResponse',
    
    # Health check models
    'ServiceHealthModel', 'HealthCheckResponse', 'SystemInfoModel', 
    'ConfigurationValidationModel', 'DetailedHealthResponse', 'AdminHealthResponse',
    
    # WebSocket models
    'WebSocketMessage', 'WebSocketDetectionRequest', 'WebSocketDetectionResponse',
    
    # Utility models
    'RateLimitInfo', 'MetricsModel', 'MetricsResponse', 'FileUploadModel',
    'ExportRequestModel', 'ExportResponse', 'ConfigUpdateModel', 'ConfigUpdateResponse',
    
    # Enums
    'DetectionModeEnum', 'TrainingModelTypeEnum', 'HealthStatusEnum'
]