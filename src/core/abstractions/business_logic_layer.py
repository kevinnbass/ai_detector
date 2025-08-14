"""
Business Logic Layer Abstraction
Provides abstraction between API/UI and core business logic
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from src.core.repositories.detection_repository import DetectionResult, DetectionMode
from src.core.abstractions.data_access_layer import IDetectionDataAccess, Query, PagedResult


class BusinessRuleError(Exception):
    """Business rule violation error"""
    def __init__(self, rule: str, message: str, details: Dict[str, Any] = None):
        self.rule = rule
        self.message = message
        self.details = details or {}
        super().__init__(message)


class ValidationResult:
    """Result of business validation"""
    
    def __init__(self, is_valid: bool, errors: List[str] = None, warnings: List[str] = None):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
    
    def add_error(self, error: str):
        """Add validation error"""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """Add validation warning"""
        self.warnings.append(warning)


@dataclass
class DetectionRequest:
    """Detection request from API/UI layer"""
    text: str
    mode: DetectionMode = DetectionMode.HYBRID
    user_id: Optional[str] = None
    source: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TrainingRequest:
    """Training request from API/UI layer"""
    training_data: List[Dict[str, Any]]
    model_type: str = "ensemble"
    hyperparameters: Dict[str, Any] = None
    validation_split: float = 0.2
    user_id: Optional[str] = None
    
    def __post_init__(self):
        if self.hyperparameters is None:
            self.hyperparameters = {}


class IBusinessRuleEngine(ABC):
    """Interface for business rule engine"""
    
    @abstractmethod
    async def validate_detection_request(self, request: DetectionRequest) -> ValidationResult:
        """Validate detection request"""
        pass
    
    @abstractmethod
    async def validate_training_request(self, request: TrainingRequest) -> ValidationResult:
        """Validate training request"""
        pass
    
    @abstractmethod
    async def can_user_perform_action(self, user_id: str, action: str, resource: str) -> bool:
        """Check if user can perform action"""
        pass
    
    @abstractmethod
    async def apply_rate_limits(self, user_id: str, action: str) -> bool:
        """Apply rate limiting rules"""
        pass


class BusinessRuleEngine(IBusinessRuleEngine):
    """Implementation of business rule engine"""
    
    def __init__(self):
        self.rules = {}
        self.rate_limits = {}
        self.user_actions = {}  # Track user actions for rate limiting
        
    def register_rule(self, rule_name: str, rule_func: Callable) -> None:
        """Register business rule"""
        self.rules[rule_name] = rule_func
        
    async def validate_detection_request(self, request: DetectionRequest) -> ValidationResult:
        """Validate detection request"""
        result = ValidationResult(True)
        
        # Text validation
        if not request.text or not request.text.strip():
            result.add_error("Text cannot be empty")
        
        if len(request.text) > 10000:
            result.add_error("Text length exceeds maximum of 10,000 characters")
        
        if len(request.text) < 10:
            result.add_warning("Text is very short, detection may be unreliable")
        
        # Mode validation
        if not isinstance(request.mode, DetectionMode):
            result.add_error("Invalid detection mode")
        
        # User validation
        if request.user_id:
            # Check user permissions
            can_detect = await self.can_user_perform_action(request.user_id, "detect", "text")
            if not can_detect:
                result.add_error("User not authorized to perform detection")
            
            # Apply rate limits
            rate_ok = await self.apply_rate_limits(request.user_id, "detection")
            if not rate_ok:
                result.add_error("Rate limit exceeded for user")
        
        # Apply custom rules
        for rule_name, rule_func in self.rules.items():
            if rule_name.startswith("detection_"):
                try:
                    rule_result = await rule_func(request)
                    if not rule_result:
                        result.add_error(f"Business rule violated: {rule_name}")
                except Exception as e:
                    result.add_warning(f"Rule {rule_name} failed to execute: {e}")
        
        return result
    
    async def validate_training_request(self, request: TrainingRequest) -> ValidationResult:
        """Validate training request"""
        result = ValidationResult(True)
        
        # Training data validation
        if not request.training_data:
            result.add_error("Training data cannot be empty")
        
        if len(request.training_data) < 10:
            result.add_error("Minimum 10 training samples required")
        
        if len(request.training_data) > 10000:
            result.add_warning("Large training dataset may take significant time")
        
        # Validate data format
        for i, sample in enumerate(request.training_data[:5]):  # Check first 5 samples
            if not isinstance(sample, dict):
                result.add_error(f"Training sample {i} must be a dictionary")
                continue
                
            if 'text' not in sample or 'label' not in sample:
                result.add_error(f"Training sample {i} missing 'text' or 'label' field")
            
            if sample.get('label') not in [0, 1]:
                result.add_error(f"Training sample {i} label must be 0 or 1")
        
        # Validation split
        if not 0.1 <= request.validation_split <= 0.5:
            result.add_error("Validation split must be between 0.1 and 0.5")
        
        # User permissions
        if request.user_id:
            can_train = await self.can_user_perform_action(request.user_id, "train", "model")
            if not can_train:
                result.add_error("User not authorized to train models")
        
        return result
    
    async def can_user_perform_action(self, user_id: str, action: str, resource: str) -> bool:
        """Check if user can perform action"""
        # Simple permission system - in real implementation, this would
        # check against a proper authorization system
        
        # For now, allow all actions but could be extended
        blocked_users = []  # Could load from config
        
        if user_id in blocked_users:
            return False
        
        # Check resource-specific permissions
        if action == "train" and resource == "model":
            # Training might require special permissions
            # For now, allow all users
            pass
        
        return True
    
    async def apply_rate_limits(self, user_id: str, action: str) -> bool:
        """Apply rate limiting rules"""
        current_time = datetime.utcnow()
        
        # Initialize user tracking if not exists
        if user_id not in self.user_actions:
            self.user_actions[user_id] = {}
        
        if action not in self.user_actions[user_id]:
            self.user_actions[user_id][action] = []
        
        # Clean old actions (older than 1 hour)
        hour_ago = current_time.timestamp() - 3600
        self.user_actions[user_id][action] = [
            timestamp for timestamp in self.user_actions[user_id][action]
            if timestamp > hour_ago
        ]
        
        # Define rate limits
        limits = {
            "detection": 1000,  # 1000 detections per hour
            "training": 10,     # 10 training sessions per hour
        }
        
        limit = limits.get(action, 100)  # Default limit
        
        # Check if limit exceeded
        if len(self.user_actions[user_id][action]) >= limit:
            return False
        
        # Record this action
        self.user_actions[user_id][action].append(current_time.timestamp())
        return True


class IDetectionBusinessLogic(ABC):
    """Interface for detection business logic"""
    
    @abstractmethod
    async def detect_text(self, request: DetectionRequest) -> DetectionResult:
        """Detect AI content in text"""
        pass
    
    @abstractmethod
    async def batch_detect(self, requests: List[DetectionRequest]) -> List[DetectionResult]:
        """Batch detect multiple texts"""
        pass
    
    @abstractmethod
    async def get_detection_history(self, user_id: str, limit: Optional[int] = None) -> List[DetectionResult]:
        """Get user's detection history"""
        pass
    
    @abstractmethod
    async def get_detection_statistics(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get detection statistics"""
        pass


class DetectionBusinessLogic(IDetectionBusinessLogic):
    """Implementation of detection business logic"""
    
    def __init__(self, data_access: IDetectionDataAccess, 
                 detector_factory, rule_engine: IBusinessRuleEngine):
        self.data_access = data_access
        self.detector_factory = detector_factory
        self.rule_engine = rule_engine
        self.active_detectors = {}  # Cache for detectors
        
    async def _get_detector(self, mode: DetectionMode):
        """Get or create detector for mode"""
        if mode not in self.active_detectors:
            detector = self.detector_factory.create_from_mode(mode)
            await detector.initialize()
            self.active_detectors[mode] = detector
        
        return self.active_detectors[mode]
    
    async def detect_text(self, request: DetectionRequest) -> DetectionResult:
        """Detect AI content in text"""
        # Validate request
        validation = await self.rule_engine.validate_detection_request(request)
        if not validation.is_valid:
            raise BusinessRuleError(
                "validation_failed",
                f"Validation failed: {'; '.join(validation.errors)}",
                {"errors": validation.errors, "warnings": validation.warnings}
            )
        
        # Get appropriate detector
        detector = await self._get_detector(request.mode)
        
        # Perform detection
        detection_data = await detector.detect(request.text)
        
        # Create detection result
        result = DetectionResult(
            text=request.text,
            is_ai=detection_data["is_ai"],
            confidence=detection_data["confidence"],
            mode=request.mode,
            indicators=detection_data.get("indicators", []),
            metadata={
                **detection_data.get("metadata", {}),
                **request.metadata,
                "source": request.source,
                "request_timestamp": datetime.utcnow().isoformat()
            },
            user_id=request.user_id,
            processing_time=detection_data.get("processing_time")
        )
        
        # Save result
        result_id = await self.data_access.create(result)
        
        # Add result ID to metadata
        if hasattr(result, 'result_id'):
            result.result_id = result_id
        
        return result
    
    async def batch_detect(self, requests: List[DetectionRequest]) -> List[DetectionResult]:
        """Batch detect multiple texts"""
        results = []
        
        # Group requests by mode for efficiency
        mode_groups = {}
        for i, request in enumerate(requests):
            if request.mode not in mode_groups:
                mode_groups[request.mode] = []
            mode_groups[request.mode].append((i, request))
        
        # Process each mode group
        for mode, mode_requests in mode_groups.items():
            detector = await self._get_detector(mode)
            
            # Extract texts for batch processing
            texts = [req.text for _, req in mode_requests]
            
            # Perform batch detection
            batch_results = await detector.batch_detect(texts)
            
            # Create detection results
            for (original_index, request), detection_data in zip(mode_requests, batch_results):
                result = DetectionResult(
                    text=request.text,
                    is_ai=detection_data["is_ai"],
                    confidence=detection_data["confidence"],
                    mode=request.mode,
                    indicators=detection_data.get("indicators", []),
                    metadata={
                        **detection_data.get("metadata", {}),
                        **request.metadata,
                        "batch_index": original_index,
                        "source": request.source
                    },
                    user_id=request.user_id
                )
                
                # Insert at original position
                while len(results) <= original_index:
                    results.append(None)
                results[original_index] = result
        
        # Save all results
        for result in results:
            if result:
                await self.data_access.create(result)
        
        return results
    
    async def get_detection_history(self, user_id: str, limit: Optional[int] = None) -> List[DetectionResult]:
        """Get user's detection history"""
        return await self.data_access.get_by_user(user_id, limit)
    
    async def get_detection_statistics(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get detection statistics"""
        stats = await self.data_access.get_statistics(user_id)
        
        # Add business logic statistics
        if user_id:
            # Get user-specific insights
            recent_results = await self.data_access.get_recent(hours=24)
            user_recent = [r for r in recent_results if r.user_id == user_id]
            
            stats["recent_24h"] = len(user_recent)
            stats["recent_ai_detected"] = sum(1 for r in user_recent if r.is_ai)
            
            # Calculate trends
            if len(user_recent) > 1:
                avg_confidence = sum(r.confidence for r in user_recent) / len(user_recent)
                stats["average_confidence_24h"] = avg_confidence
        
        return stats
    
    async def cleanup_detectors(self):
        """Cleanup active detectors"""
        for detector in self.active_detectors.values():
            await detector.cleanup()
        self.active_detectors.clear()


class ITrainingBusinessLogic(ABC):
    """Interface for training business logic"""
    
    @abstractmethod
    async def train_model(self, request: TrainingRequest) -> Dict[str, Any]:
        """Train new model"""
        pass
    
    @abstractmethod
    async def get_training_history(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get training history"""
        pass
    
    @abstractmethod
    async def get_model_performance(self, model_id: str) -> Dict[str, Any]:
        """Get model performance metrics"""
        pass


class BusinessLogicFacade:
    """Facade for all business logic operations"""
    
    def __init__(self, detection_logic: IDetectionBusinessLogic,
                 training_logic: ITrainingBusinessLogic,
                 rule_engine: IBusinessRuleEngine):
        self.detection = detection_logic
        self.training = training_logic
        self.rules = rule_engine
    
    async def initialize(self):
        """Initialize all business logic components"""
        # Initialize components if needed
        pass
    
    async def cleanup(self):
        """Cleanup all business logic components"""
        if hasattr(self.detection, 'cleanup_detectors'):
            await self.detection.cleanup_detectors()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of business logic layer"""
        return {
            "detection_logic": "healthy",
            "training_logic": "healthy", 
            "rule_engine": "healthy",
            "timestamp": datetime.utcnow().isoformat()
        }


__all__ = [
    'BusinessRuleError', 'ValidationResult', 'DetectionRequest', 'TrainingRequest',
    'IBusinessRuleEngine', 'BusinessRuleEngine',
    'IDetectionBusinessLogic', 'DetectionBusinessLogic',
    'ITrainingBusinessLogic', 'BusinessLogicFacade'
]