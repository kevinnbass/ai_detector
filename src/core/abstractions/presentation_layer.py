"""
Presentation Layer Abstraction
Provides abstraction between external interfaces (API/UI) and business logic
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

from src.core.abstractions.business_logic_layer import (
    DetectionRequest, TrainingRequest, IDetectionBusinessLogic, ITrainingBusinessLogic
)


class ResponseStatus(Enum):
    """Response status codes"""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    VALIDATION_ERROR = "validation_error"
    AUTHORIZATION_ERROR = "authorization_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    INTERNAL_ERROR = "internal_error"


@dataclass
class APIResponse:
    """Standard API response format"""
    status: ResponseStatus
    data: Any = None
    message: str = ""
    errors: List[str] = None
    warnings: List[str] = None
    metadata: Dict[str, Any] = None
    timestamp: str = None
    request_id: str = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def success(cls, data: Any = None, message: str = "", **kwargs) -> 'APIResponse':
        """Create success response"""
        return cls(
            status=ResponseStatus.SUCCESS,
            data=data,
            message=message,
            **kwargs
        )
    
    @classmethod
    def error(cls, message: str, errors: List[str] = None, 
              status: ResponseStatus = ResponseStatus.ERROR, **kwargs) -> 'APIResponse':
        """Create error response"""
        return cls(
            status=status,
            message=message,
            errors=errors or [],
            **kwargs
        )
    
    @classmethod
    def validation_error(cls, errors: List[str], message: str = "Validation failed") -> 'APIResponse':
        """Create validation error response"""
        return cls(
            status=ResponseStatus.VALIDATION_ERROR,
            message=message,
            errors=errors
        )


@dataclass
class PaginationRequest:
    """Pagination parameters for requests"""
    page: int = 1
    page_size: int = 20
    sort_by: str = "created_at"
    sort_desc: bool = True
    
    def __post_init__(self):
        # Validate pagination parameters
        self.page = max(1, self.page)
        self.page_size = max(1, min(100, self.page_size))  # Cap at 100
    
    @property
    def offset(self) -> int:
        """Calculate offset from page and page_size"""
        return (self.page - 1) * self.page_size


@dataclass
class PaginatedResponse:
    """Paginated response data"""
    items: List[Any]
    total_count: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_previous: bool
    
    @classmethod
    def from_paged_result(cls, paged_result, serializer_func=None) -> 'PaginatedResponse':
        """Create from PagedResult"""
        items = paged_result.items
        if serializer_func:
            items = [serializer_func(item) for item in items]
        
        return cls(
            items=items,
            total_count=paged_result.total_count,
            page=paged_result.page_number,
            page_size=paged_result.page_size,
            total_pages=paged_result.total_pages,
            has_next=paged_result.has_next,
            has_previous=paged_result.has_previous
        )


class IRequestValidator(ABC):
    """Interface for request validation"""
    
    @abstractmethod
    def validate_detection_request(self, data: Dict[str, Any]) -> List[str]:
        """Validate detection request data"""
        pass
    
    @abstractmethod
    def validate_training_request(self, data: Dict[str, Any]) -> List[str]:
        """Validate training request data"""
        pass
    
    @abstractmethod
    def validate_pagination(self, data: Dict[str, Any]) -> List[str]:
        """Validate pagination parameters"""
        pass


class RequestValidator(IRequestValidator):
    """Implementation of request validator"""
    
    def validate_detection_request(self, data: Dict[str, Any]) -> List[str]:
        """Validate detection request data"""
        errors = []
        
        # Required fields
        if 'text' not in data:
            errors.append("Field 'text' is required")
        elif not data['text'] or not data['text'].strip():
            errors.append("Field 'text' cannot be empty")
        elif len(data['text']) > 10000:
            errors.append("Text length exceeds maximum of 10,000 characters")
        
        # Optional fields validation
        if 'mode' in data:
            valid_modes = ['pattern', 'llm', 'hybrid']
            if data['mode'] not in valid_modes:
                errors.append(f"Field 'mode' must be one of: {', '.join(valid_modes)}")
        
        if 'user_id' in data and data['user_id']:
            if not isinstance(data['user_id'], str):
                errors.append("Field 'user_id' must be a string")
        
        return errors
    
    def validate_training_request(self, data: Dict[str, Any]) -> List[str]:
        """Validate training request data"""
        errors = []
        
        # Required fields
        if 'training_data' not in data:
            errors.append("Field 'training_data' is required")
        elif not isinstance(data['training_data'], list):
            errors.append("Field 'training_data' must be a list")
        elif len(data['training_data']) < 10:
            errors.append("Minimum 10 training samples required")
        else:
            # Validate training samples
            for i, sample in enumerate(data['training_data'][:5]):  # Check first 5
                if not isinstance(sample, dict):
                    errors.append(f"Training sample {i} must be a dictionary")
                    continue
                
                if 'text' not in sample:
                    errors.append(f"Training sample {i} missing 'text' field")
                if 'label' not in sample:
                    errors.append(f"Training sample {i} missing 'label' field")
                elif sample['label'] not in [0, 1]:
                    errors.append(f"Training sample {i} label must be 0 or 1")
        
        # Optional fields
        if 'validation_split' in data:
            split = data['validation_split']
            if not isinstance(split, (int, float)) or not 0.1 <= split <= 0.5:
                errors.append("Field 'validation_split' must be between 0.1 and 0.5")
        
        return errors
    
    def validate_pagination(self, data: Dict[str, Any]) -> List[str]:
        """Validate pagination parameters"""
        errors = []
        
        if 'page' in data:
            if not isinstance(data['page'], int) or data['page'] < 1:
                errors.append("Field 'page' must be a positive integer")
        
        if 'page_size' in data:
            if not isinstance(data['page_size'], int) or not 1 <= data['page_size'] <= 100:
                errors.append("Field 'page_size' must be between 1 and 100")
        
        return errors


class ISerializer(ABC):
    """Interface for data serialization"""
    
    @abstractmethod
    def serialize_detection_result(self, result) -> Dict[str, Any]:
        """Serialize detection result for API response"""
        pass
    
    @abstractmethod
    def serialize_training_result(self, result) -> Dict[str, Any]:
        """Serialize training result for API response"""
        pass
    
    @abstractmethod
    def serialize_statistics(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize statistics for API response"""
        pass


class Serializer(ISerializer):
    """Implementation of data serializer"""
    
    def serialize_detection_result(self, result) -> Dict[str, Any]:
        """Serialize detection result for API response"""
        return {
            "id": getattr(result, 'result_id', None),
            "text": result.text[:200] + "..." if len(result.text) > 200 else result.text,
            "is_ai": result.is_ai,
            "confidence": round(result.confidence, 3),
            "mode": result.mode.value if hasattr(result.mode, 'value') else str(result.mode),
            "indicators": result.indicators,
            "timestamp": result.timestamp.isoformat() if result.timestamp else None,
            "processing_time": result.processing_time,
            "metadata": {
                "text_length": len(result.text),
                "indicator_count": len(result.indicators),
                **result.metadata
            }
        }
    
    def serialize_training_result(self, result) -> Dict[str, Any]:
        """Serialize training result for API response"""
        return {
            "model_id": result.get("model_id"),
            "model_type": result.get("model_type"),
            "metrics": {
                k: round(v, 3) if isinstance(v, (int, float)) else v
                for k, v in result.get("metrics", {}).items()
            },
            "training_time": result.get("training_time"),
            "sample_count": result.get("sample_count"),
            "timestamp": result.get("timestamp")
        }
    
    def serialize_statistics(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize statistics for API response"""
        return {
            "total_detections": stats.get("total_detections", 0),
            "ai_detected": stats.get("ai_detected", 0),
            "human_detected": stats.get("human_detected", 0),
            "ai_percentage": round(stats.get("ai_percentage", 0), 1),
            "average_confidence": round(stats.get("average_confidence", 0), 3),
            "recent_24h": stats.get("recent_24h", 0),
            "confidence_distribution": stats.get("confidence_distribution", {}),
            "statistics_by_mode": stats.get("statistics_by_mode", {})
        }


class IDetectionController(ABC):
    """Interface for detection controller"""
    
    @abstractmethod
    async def detect_text(self, request_data: Dict[str, Any]) -> APIResponse:
        """Handle text detection request"""
        pass
    
    @abstractmethod
    async def batch_detect(self, request_data: Dict[str, Any]) -> APIResponse:
        """Handle batch detection request"""
        pass
    
    @abstractmethod
    async def get_detection_history(self, user_id: str, pagination: PaginationRequest) -> APIResponse:
        """Get user's detection history"""
        pass
    
    @abstractmethod
    async def get_statistics(self, user_id: Optional[str] = None) -> APIResponse:
        """Get detection statistics"""
        pass


class DetectionController(IDetectionController):
    """Implementation of detection controller"""
    
    def __init__(self, business_logic: IDetectionBusinessLogic,
                 validator: IRequestValidator, serializer: ISerializer):
        self.business_logic = business_logic
        self.validator = validator
        self.serializer = serializer
    
    async def detect_text(self, request_data: Dict[str, Any]) -> APIResponse:
        """Handle text detection request"""
        try:
            # Validate request
            validation_errors = self.validator.validate_detection_request(request_data)
            if validation_errors:
                return APIResponse.validation_error(validation_errors)
            
            # Create business logic request
            detection_request = DetectionRequest(
                text=request_data['text'],
                mode=request_data.get('mode', 'hybrid'),
                user_id=request_data.get('user_id'),
                source=request_data.get('source', 'api'),
                metadata=request_data.get('metadata', {})
            )
            
            # Convert mode string to enum if needed
            if isinstance(detection_request.mode, str):
                from src.core.repositories.detection_repository import DetectionMode
                mode_map = {
                    'pattern': DetectionMode.PATTERN,
                    'llm': DetectionMode.LLM,
                    'hybrid': DetectionMode.HYBRID
                }
                detection_request.mode = mode_map.get(detection_request.mode, DetectionMode.HYBRID)
            
            # Execute business logic
            result = await self.business_logic.detect_text(detection_request)
            
            # Serialize response
            serialized_result = self.serializer.serialize_detection_result(result)
            
            return APIResponse.success(
                data=serialized_result,
                message="Text detection completed successfully"
            )
            
        except Exception as e:
            return APIResponse.error(
                message="Detection failed",
                errors=[str(e)],
                status=ResponseStatus.INTERNAL_ERROR
            )
    
    async def batch_detect(self, request_data: Dict[str, Any]) -> APIResponse:
        """Handle batch detection request"""
        try:
            # Validate batch request
            if 'requests' not in request_data or not isinstance(request_data['requests'], list):
                return APIResponse.validation_error(["Field 'requests' must be a list"])
            
            if len(request_data['requests']) > 50:  # Limit batch size
                return APIResponse.validation_error(["Batch size limited to 50 requests"])
            
            # Validate each request
            requests = []
            validation_errors = []
            
            for i, req_data in enumerate(request_data['requests']):
                errors = self.validator.validate_detection_request(req_data)
                if errors:
                    validation_errors.extend([f"Request {i}: {error}" for error in errors])
                else:
                    detection_request = DetectionRequest(
                        text=req_data['text'],
                        mode=req_data.get('mode', 'hybrid'),
                        user_id=req_data.get('user_id'),
                        source=req_data.get('source', 'api'),
                        metadata=req_data.get('metadata', {})
                    )
                    requests.append(detection_request)
            
            if validation_errors:
                return APIResponse.validation_error(validation_errors)
            
            # Execute batch detection
            results = await self.business_logic.batch_detect(requests)
            
            # Serialize results
            serialized_results = [
                self.serializer.serialize_detection_result(result)
                for result in results
            ]
            
            return APIResponse.success(
                data={
                    "results": serialized_results,
                    "total_processed": len(serialized_results)
                },
                message=f"Batch detection completed for {len(serialized_results)} texts"
            )
            
        except Exception as e:
            return APIResponse.error(
                message="Batch detection failed",
                errors=[str(e)],
                status=ResponseStatus.INTERNAL_ERROR
            )
    
    async def get_detection_history(self, user_id: str, pagination: PaginationRequest) -> APIResponse:
        """Get user's detection history"""
        try:
            # Get detection history
            results = await self.business_logic.get_detection_history(
                user_id, 
                limit=pagination.page_size
            )
            
            # Apply pagination (simplified - in real implementation would use data layer pagination)
            start_idx = pagination.offset
            end_idx = start_idx + pagination.page_size
            paginated_results = results[start_idx:end_idx]
            
            # Serialize results
            serialized_results = [
                self.serializer.serialize_detection_result(result)
                for result in paginated_results
            ]
            
            # Create paginated response
            paginated_response = PaginatedResponse(
                items=serialized_results,
                total_count=len(results),
                page=pagination.page,
                page_size=pagination.page_size,
                total_pages=(len(results) + pagination.page_size - 1) // pagination.page_size,
                has_next=end_idx < len(results),
                has_previous=pagination.page > 1
            )
            
            return APIResponse.success(
                data=asdict(paginated_response),
                message="Detection history retrieved successfully"
            )
            
        except Exception as e:
            return APIResponse.error(
                message="Failed to retrieve detection history",
                errors=[str(e)],
                status=ResponseStatus.INTERNAL_ERROR
            )
    
    async def get_statistics(self, user_id: Optional[str] = None) -> APIResponse:
        """Get detection statistics"""
        try:
            # Get statistics from business logic
            stats = await self.business_logic.get_detection_statistics(user_id)
            
            # Serialize statistics
            serialized_stats = self.serializer.serialize_statistics(stats)
            
            return APIResponse.success(
                data=serialized_stats,
                message="Statistics retrieved successfully"
            )
            
        except Exception as e:
            return APIResponse.error(
                message="Failed to retrieve statistics",
                errors=[str(e)],
                status=ResponseStatus.INTERNAL_ERROR
            )


class ControllerRegistry:
    """Registry for managing controllers"""
    
    def __init__(self):
        self._controllers = {}
    
    def register(self, name: str, controller) -> None:
        """Register controller"""
        self._controllers[name] = controller
    
    def get(self, name: str):
        """Get controller by name"""
        return self._controllers.get(name)
    
    def get_all(self) -> Dict[str, Any]:
        """Get all controllers"""
        return self._controllers.copy()


# Global controller registry
controller_registry = ControllerRegistry()


__all__ = [
    'ResponseStatus', 'APIResponse', 'PaginationRequest', 'PaginatedResponse',
    'IRequestValidator', 'RequestValidator',
    'ISerializer', 'Serializer',
    'IDetectionController', 'DetectionController',
    'ControllerRegistry', 'controller_registry'
]