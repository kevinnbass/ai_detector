"""
Detector Interface Definitions
Interfaces for AI text detection components
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

from .base_interfaces import IInitializable, IConfigurable, IMetricsProvider, IValidatable


class DetectionMethod(Enum):
    """Detection method enumeration"""
    PATTERN_BASED = "pattern_based"
    LLM_BASED = "llm_based"
    ML_BASED = "ml_based"
    ENSEMBLE = "ensemble"
    HYBRID = "hybrid"


class ConfidenceLevel(Enum):
    """Confidence level enumeration"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class DetectionScore:
    """Standardized detection score"""
    ai_probability: float  # 0.0 to 1.0
    prediction: str  # 'ai' or 'human'
    confidence: float  # 0.0 to 1.0
    confidence_level: ConfidenceLevel


@dataclass
class DetectionEvidence:
    """Evidence supporting detection result"""
    indicator_type: str
    description: str
    weight: float
    location: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class IDetectionResult(ABC):
    """Interface for detection results"""
    
    @abstractmethod
    def get_score(self) -> DetectionScore:
        """Get detection score"""
        pass
    
    @abstractmethod
    def get_evidence(self) -> List[DetectionEvidence]:
        """Get supporting evidence"""
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get result metadata"""
        pass
    
    @abstractmethod
    def get_processing_time(self) -> float:
        """Get processing time in seconds"""
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        pass
    
    @abstractmethod
    def to_json(self) -> str:
        """Convert to JSON string"""
        pass


class IDetectionConfig(ABC):
    """Interface for detection configuration"""
    
    @abstractmethod
    def get_threshold(self) -> float:
        """Get detection threshold"""
        pass
    
    @abstractmethod
    def get_method(self) -> DetectionMethod:
        """Get detection method"""
        pass
    
    @abstractmethod
    def get_options(self) -> Dict[str, Any]:
        """Get detection options"""
        pass
    
    @abstractmethod
    def is_quick_mode(self) -> bool:
        """Check if quick mode is enabled"""
        pass
    
    @abstractmethod
    def validate(self) -> tuple[bool, List[str]]:
        """Validate configuration"""
        pass


class IDetector(IInitializable, IConfigurable, IMetricsProvider, IValidatable, ABC):
    """Base interface for all detectors"""
    
    @abstractmethod
    async def detect(self, text: str, config: Optional[IDetectionConfig] = None) -> IDetectionResult:
        """
        Detect AI-generated text
        
        Args:
            text: Text to analyze
            config: Optional detection configuration
            
        Returns:
            Detection result
        """
        pass
    
    @abstractmethod
    async def detect_batch(self, texts: List[str], config: Optional[IDetectionConfig] = None) -> List[IDetectionResult]:
        """
        Detect AI-generated text in batch
        
        Args:
            texts: List of texts to analyze
            config: Optional detection configuration
            
        Returns:
            List of detection results
        """
        pass
    
    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        pass
    
    @abstractmethod
    def get_detection_method(self) -> DetectionMethod:
        """Get detection method"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        pass
    
    @abstractmethod
    def can_explain(self) -> bool:
        """Check if detector can provide explanations"""
        pass


class IPatternDetector(IDetector):
    """Interface for pattern-based detectors"""
    
    @abstractmethod
    def get_patterns(self) -> Dict[str, Any]:
        """Get detection patterns"""
        pass
    
    @abstractmethod
    def add_pattern(self, name: str, pattern: Any) -> None:
        """Add new detection pattern"""
        pass
    
    @abstractmethod
    def remove_pattern(self, name: str) -> bool:
        """Remove detection pattern"""
        pass
    
    @abstractmethod
    def update_pattern(self, name: str, pattern: Any) -> bool:
        """Update existing pattern"""
        pass


class ILLMDetector(IDetector):
    """Interface for LLM-based detectors"""
    
    @abstractmethod
    async def set_api_key(self, api_key: str) -> bool:
        """Set API key for LLM service"""
        pass
    
    @abstractmethod
    def get_llm_provider(self) -> str:
        """Get LLM provider name"""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get model name"""
        pass
    
    @abstractmethod
    async def test_connection(self) -> bool:
        """Test connection to LLM service"""
        pass
    
    @abstractmethod
    def get_token_usage(self) -> Dict[str, int]:
        """Get token usage statistics"""
        pass
    
    @abstractmethod
    def estimate_cost(self, text: str) -> float:
        """Estimate analysis cost"""
        pass


class IMLDetector(IDetector):
    """Interface for ML-based detectors"""
    
    @abstractmethod
    async def load_model(self, model_path: str) -> bool:
        """Load ML model"""
        pass
    
    @abstractmethod
    def get_model_path(self) -> Optional[str]:
        """Get current model path"""
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get feature names used by model"""
        pass
    
    @abstractmethod
    def extract_features(self, text: str) -> Dict[str, float]:
        """Extract features from text"""
        pass
    
    @abstractmethod
    def get_model_accuracy(self) -> Optional[float]:
        """Get model accuracy score"""
        pass


class IEnsembleDetector(IDetector):
    """Interface for ensemble detectors"""
    
    @abstractmethod
    def add_detector(self, detector: IDetector, weight: float = 1.0) -> None:
        """Add detector to ensemble"""
        pass
    
    @abstractmethod
    def remove_detector(self, detector: IDetector) -> bool:
        """Remove detector from ensemble"""
        pass
    
    @abstractmethod
    def get_detectors(self) -> List[tuple[IDetector, float]]:
        """Get list of detectors and their weights"""
        pass
    
    @abstractmethod
    def set_combination_method(self, method: str) -> None:
        """Set result combination method"""
        pass
    
    @abstractmethod
    def get_individual_results(self, text: str) -> Dict[str, IDetectionResult]:
        """Get results from individual detectors"""
        pass


class IStreamingDetector(IDetector):
    """Interface for streaming detection"""
    
    @abstractmethod
    async def detect_stream(self, text_stream: AsyncGenerator[str, None], 
                          config: Optional[IDetectionConfig] = None) -> AsyncGenerator[IDetectionResult, None]:
        """
        Detect AI-generated text from stream
        
        Args:
            text_stream: Async generator of text chunks
            config: Optional detection configuration
            
        Yields:
            Detection results as they become available
        """
        pass
    
    @abstractmethod
    def get_buffer_size(self) -> int:
        """Get buffer size for streaming"""
        pass
    
    @abstractmethod
    def set_buffer_size(self, size: int) -> None:
        """Set buffer size for streaming"""
        pass


class IAdaptiveDetector(IDetector):
    """Interface for adaptive detectors that learn from feedback"""
    
    @abstractmethod
    async def provide_feedback(self, text: str, correct_label: str, 
                             original_result: IDetectionResult) -> None:
        """
        Provide feedback for adaptive learning
        
        Args:
            text: Original text
            correct_label: Correct label ('ai' or 'human')
            original_result: Original detection result
        """
        pass
    
    @abstractmethod
    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Get adaptation statistics"""
        pass
    
    @abstractmethod
    def enable_adaptation(self, enabled: bool) -> None:
        """Enable or disable adaptation"""
        pass
    
    @abstractmethod
    def is_adaptation_enabled(self) -> bool:
        """Check if adaptation is enabled"""
        pass


class IExplainableDetector(IDetector):
    """Interface for detectors that can explain their decisions"""
    
    @abstractmethod
    async def explain(self, text: str, result: IDetectionResult) -> Dict[str, Any]:
        """
        Explain detection decision
        
        Args:
            text: Original text
            result: Detection result to explain
            
        Returns:
            Explanation data
        """
        pass
    
    @abstractmethod
    def get_explanation_types(self) -> List[str]:
        """Get available explanation types"""
        pass
    
    @abstractmethod
    async def generate_examples(self, prediction: str, count: int = 5) -> List[str]:
        """Generate example texts for given prediction"""
        pass


class IDetectorFactory(ABC):
    """Interface for detector factory"""
    
    @abstractmethod
    def create_detector(self, detector_type: DetectionMethod, **kwargs) -> IDetector:
        """Create detector instance"""
        pass
    
    @abstractmethod
    def get_available_types(self) -> List[DetectionMethod]:
        """Get available detector types"""
        pass
    
    @abstractmethod
    def register_detector_class(self, detector_type: DetectionMethod, detector_class: type) -> None:
        """Register new detector class"""
        pass