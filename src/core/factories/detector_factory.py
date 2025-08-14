"""
Factory Pattern Implementation for Detectors
Creates detector instances based on configuration and strategy
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type, List
from enum import Enum
import logging

from src.core.repositories.detection_repository import DetectionMode
from src.utils.common import Config

logger = logging.getLogger(__name__)


class DetectorType(Enum):
    PATTERN_BASED = "pattern_based"
    LLM_BASED = "llm_based" 
    ML_BASED = "ml_based"
    HYBRID = "hybrid"
    ENSEMBLE = "ensemble"


class DetectorConfig:
    """Configuration for detector creation"""
    
    def __init__(self, detector_type: DetectorType, **kwargs):
        self.detector_type = detector_type
        self.config = kwargs
        
        # Common configuration
        self.confidence_threshold = kwargs.get('confidence_threshold', 0.7)
        self.max_text_length = kwargs.get('max_text_length', 10000)
        self.cache_enabled = kwargs.get('cache_enabled', True)
        
        # Pattern-based specific
        self.pattern_file = kwargs.get('pattern_file', 'patterns.json')
        self.pattern_weights = kwargs.get('pattern_weights', {})
        
        # LLM-based specific
        self.llm_provider = kwargs.get('llm_provider', 'gemini')
        self.llm_model = kwargs.get('llm_model', 'gemini-pro')
        self.llm_api_key = kwargs.get('llm_api_key')
        self.llm_temperature = kwargs.get('llm_temperature', 0.1)
        self.llm_max_tokens = kwargs.get('llm_max_tokens', 150)
        
        # ML-based specific
        self.model_path = kwargs.get('model_path')
        self.model_type = kwargs.get('model_type', 'random_forest')
        self.feature_extractors = kwargs.get('feature_extractors', ['basic', 'ngram'])
        
        # Hybrid specific
        self.hybrid_weights = kwargs.get('hybrid_weights', {
            'pattern': 0.3,
            'llm': 0.4,
            'ml': 0.3
        })
        
        # Ensemble specific
        self.ensemble_methods = kwargs.get('ensemble_methods', ['pattern_based', 'llm_based'])
        self.ensemble_voting = kwargs.get('ensemble_voting', 'weighted')


class IDetector(ABC):
    """Base detector interface"""
    
    @abstractmethod
    async def detect(self, text: str) -> Dict[str, Any]:
        """Detect AI-generated content in text"""
        pass
    
    @abstractmethod
    async def batch_detect(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Batch detection for multiple texts"""
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Get detector information"""
        pass
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize detector resources"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup detector resources"""
        pass


class BaseDetector(IDetector):
    """Base detector implementation"""
    
    def __init__(self, config: DetectorConfig):
        self.config = config
        self.detector_type = config.detector_type
        self.initialized = False
        self.detection_count = 0
        
    async def initialize(self) -> None:
        """Initialize detector resources"""
        logger.info(f"Initializing {self.detector_type.value} detector")
        self.initialized = True
    
    async def cleanup(self) -> None:
        """Cleanup detector resources"""
        logger.info(f"Cleaning up {self.detector_type.value} detector")
        self.initialized = False
    
    def get_info(self) -> Dict[str, Any]:
        """Get detector information"""
        return {
            "type": self.detector_type.value,
            "initialized": self.initialized,
            "detection_count": self.detection_count,
            "confidence_threshold": self.config.confidence_threshold
        }
    
    def _validate_text(self, text: str) -> None:
        """Validate input text"""
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")
        
        if len(text) > self.config.max_text_length:
            raise ValueError(f"Text length exceeds maximum of {self.config.max_text_length} characters")
    
    async def batch_detect(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Default batch detection implementation"""
        results = []
        for text in texts:
            result = await self.detect(text)
            results.append(result)
        return results


class PatternBasedDetector(BaseDetector):
    """Pattern-based AI detection"""
    
    def __init__(self, config: DetectorConfig):
        super().__init__(config)
        self.patterns = None
        
    async def initialize(self) -> None:
        """Initialize pattern-based detector"""
        await super().initialize()
        
        # Load patterns
        from src.core.patterns.pattern_registry import get_registry
        self.patterns = get_registry()
        
        logger.info(f"Loaded {len(self.patterns.get_all_patterns())} patterns")
    
    async def detect(self, text: str) -> Dict[str, Any]:
        """Detect using pattern matching"""
        self._validate_text(text)
        
        if not self.initialized:
            raise RuntimeError("Detector not initialized")
        
        # Apply patterns
        pattern_results = self.patterns.apply_patterns(text)
        
        # Calculate confidence based on pattern matches
        total_weight = sum(result['weight'] * result['count'] for result in pattern_results.values())
        max_possible_weight = sum(pattern.weight for pattern in self.patterns.get_all_patterns())
        
        confidence = min(total_weight / max_possible_weight, 1.0) if max_possible_weight > 0 else 0.0
        is_ai = confidence >= self.config.confidence_threshold
        
        # Extract indicators
        indicators = [
            pattern_id for pattern_id, result in pattern_results.items()
            if result['count'] > 0
        ]
        
        self.detection_count += 1
        
        return {
            "is_ai": is_ai,
            "confidence": confidence,
            "indicators": indicators,
            "metadata": {
                "detector_type": "pattern_based",
                "pattern_matches": pattern_results,
                "total_weight": total_weight
            }
        }


class LLMBasedDetector(BaseDetector):
    """LLM-based AI detection"""
    
    def __init__(self, config: DetectorConfig):
        super().__init__(config)
        self.llm_client = None
        
    async def initialize(self) -> None:
        """Initialize LLM-based detector"""
        await super().initialize()
        
        # Initialize LLM client based on provider
        if self.config.llm_provider == 'gemini':
            from src.integrations.unified_analyzer import GeminiProvider
            self.llm_client = GeminiProvider(self.config.llm_api_key)
        elif self.config.llm_provider == 'openai':
            # Would initialize OpenAI client
            pass
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.llm_provider}")
        
        logger.info(f"Initialized {self.config.llm_provider} LLM client")
    
    async def detect(self, text: str) -> Dict[str, Any]:
        """Detect using LLM analysis"""
        self._validate_text(text)
        
        if not self.initialized:
            raise RuntimeError("Detector not initialized")
        
        # Prepare prompt for AI detection
        prompt = f"""
        Analyze this text to determine if it was written by AI or a human.
        
        Text to analyze: "{text}"
        
        Consider these indicators of AI-generated text:
        - Overly formal or perfect grammar
        - Lack of personal opinions or experiences
        - Repetitive phrasing patterns
        - Generic or vague statements
        - Balanced arguments without taking sides
        
        Respond with a JSON object containing:
        - is_ai: boolean indicating if text is AI-generated
        - confidence: float between 0.0 and 1.0 indicating confidence level
        - reasoning: brief explanation of the decision
        - indicators: list of specific indicators found
        """
        
        try:
            response = await self.llm_client.analyze_text(text, prompt)
            
            # Parse LLM response (assuming structured response)
            result = {
                "is_ai": response.get("is_ai", False),
                "confidence": max(0.0, min(1.0, response.get("confidence", 0.5))),
                "indicators": response.get("indicators", []),
                "metadata": {
                    "detector_type": "llm_based",
                    "llm_provider": self.config.llm_provider,
                    "llm_model": self.config.llm_model,
                    "reasoning": response.get("reasoning", ""),
                    "raw_response": response
                }
            }
            
        except Exception as e:
            logger.error(f"LLM detection failed: {e}")
            # Fallback to low confidence
            result = {
                "is_ai": False,
                "confidence": 0.0,
                "indicators": [],
                "metadata": {
                    "detector_type": "llm_based",
                    "error": str(e)
                }
            }
        
        self.detection_count += 1
        return result


class HybridDetector(BaseDetector):
    """Hybrid detector combining multiple detection methods"""
    
    def __init__(self, config: DetectorConfig):
        super().__init__(config)
        self.detectors = {}
        
    async def initialize(self) -> None:
        """Initialize hybrid detector"""
        await super().initialize()
        
        # Initialize sub-detectors
        weights = self.config.hybrid_weights
        
        if weights.get('pattern', 0) > 0:
            pattern_config = DetectorConfig(DetectorType.PATTERN_BASED, **self.config.config)
            self.detectors['pattern'] = PatternBasedDetector(pattern_config)
            await self.detectors['pattern'].initialize()
        
        if weights.get('llm', 0) > 0:
            llm_config = DetectorConfig(DetectorType.LLM_BASED, **self.config.config)
            self.detectors['llm'] = LLMBasedDetector(llm_config)
            await self.detectors['llm'].initialize()
        
        logger.info(f"Initialized hybrid detector with {len(self.detectors)} sub-detectors")
    
    async def cleanup(self) -> None:
        """Cleanup hybrid detector"""
        for detector in self.detectors.values():
            await detector.cleanup()
        await super().cleanup()
    
    async def detect(self, text: str) -> Dict[str, Any]:
        """Detect using hybrid approach"""
        self._validate_text(text)
        
        if not self.initialized:
            raise RuntimeError("Detector not initialized")
        
        # Run all sub-detectors
        results = {}
        weights = self.config.hybrid_weights
        
        for detector_type, detector in self.detectors.items():
            try:
                result = await detector.detect(text)
                results[detector_type] = result
            except Exception as e:
                logger.error(f"Error in {detector_type} detector: {e}")
                results[detector_type] = {
                    "is_ai": False,
                    "confidence": 0.0,
                    "error": str(e)
                }
        
        # Combine results using weighted average
        total_confidence = 0.0
        total_weight = 0.0
        all_indicators = []
        
        for detector_type, result in results.items():
            weight = weights.get(detector_type, 0)
            if weight > 0:
                total_confidence += result.get("confidence", 0) * weight
                total_weight += weight
                all_indicators.extend(result.get("indicators", []))
        
        final_confidence = total_confidence / total_weight if total_weight > 0 else 0.0
        is_ai = final_confidence >= self.config.confidence_threshold
        
        # Remove duplicate indicators
        unique_indicators = list(set(all_indicators))
        
        self.detection_count += 1
        
        return {
            "is_ai": is_ai,
            "confidence": final_confidence,
            "indicators": unique_indicators,
            "metadata": {
                "detector_type": "hybrid",
                "sub_results": results,
                "weights_used": weights,
                "total_weight": total_weight
            }
        }


class IDetectorFactory(ABC):
    """Factory interface for creating detectors"""
    
    @abstractmethod
    def create_detector(self, detector_type: DetectorType, **kwargs) -> IDetector:
        """Create detector instance"""
        pass
    
    @abstractmethod
    def get_supported_types(self) -> List[DetectorType]:
        """Get supported detector types"""
        pass


class DetectorFactory(IDetectorFactory):
    """Main detector factory"""
    
    def __init__(self):
        self._detector_classes = {
            DetectorType.PATTERN_BASED: PatternBasedDetector,
            DetectorType.LLM_BASED: LLMBasedDetector,
            DetectorType.HYBRID: HybridDetector
        }
        
    def register_detector(self, detector_type: DetectorType, detector_class: Type[IDetector]) -> None:
        """Register custom detector class"""
        self._detector_classes[detector_type] = detector_class
        logger.info(f"Registered detector type: {detector_type.value}")
    
    def create_detector(self, detector_type: DetectorType, **kwargs) -> IDetector:
        """Create detector instance"""
        if detector_type not in self._detector_classes:
            raise ValueError(f"Unsupported detector type: {detector_type.value}")
        
        config = DetectorConfig(detector_type, **kwargs)
        detector_class = self._detector_classes[detector_type]
        
        detector = detector_class(config)
        logger.info(f"Created {detector_type.value} detector")
        
        return detector
    
    def create_from_config(self, config_dict: Dict[str, Any]) -> IDetector:
        """Create detector from configuration dictionary"""
        detector_type_str = config_dict.pop('type', 'pattern_based')
        detector_type = DetectorType(detector_type_str)
        
        return self.create_detector(detector_type, **config_dict)
    
    def create_from_mode(self, mode: DetectionMode, **kwargs) -> IDetector:
        """Create detector from detection mode"""
        mode_mapping = {
            DetectionMode.PATTERN: DetectorType.PATTERN_BASED,
            DetectionMode.LLM: DetectorType.LLM_BASED,
            DetectionMode.HYBRID: DetectorType.HYBRID
        }
        
        detector_type = mode_mapping.get(mode, DetectorType.HYBRID)
        return self.create_detector(detector_type, **kwargs)
    
    def get_supported_types(self) -> List[DetectorType]:
        """Get supported detector types"""
        return list(self._detector_classes.keys())


# Global factory instance
detector_factory = DetectorFactory()


class DetectorPool:
    """Pool of pre-initialized detectors for performance"""
    
    def __init__(self, factory: DetectorFactory):
        self.factory = factory
        self.pool = {}
        self.active_detectors = {}
    
    async def get_detector(self, detector_type: DetectorType, **kwargs) -> IDetector:
        """Get detector from pool or create new one"""
        pool_key = f"{detector_type.value}_{hash(frozenset(kwargs.items()))}"
        
        if pool_key not in self.pool:
            detector = self.factory.create_detector(detector_type, **kwargs)
            await detector.initialize()
            self.pool[pool_key] = detector
            logger.info(f"Added new detector to pool: {pool_key}")
        
        detector = self.pool[pool_key]
        self.active_detectors[pool_key] = detector
        return detector
    
    async def cleanup_all(self) -> None:
        """Cleanup all detectors in pool"""
        for detector in self.pool.values():
            await detector.cleanup()
        
        self.pool.clear()
        self.active_detectors.clear()
        logger.info("Cleaned up detector pool")
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get pool status"""
        return {
            "total_detectors": len(self.pool),
            "active_detectors": len(self.active_detectors),
            "detector_types": [key.split('_')[0] for key in self.pool.keys()]
        }


# Global detector pool
detector_pool = DetectorPool(detector_factory)


__all__ = [
    'DetectorType', 'DetectorConfig', 'IDetector', 'BaseDetector',
    'PatternBasedDetector', 'LLMBasedDetector', 'HybridDetector',
    'IDetectorFactory', 'DetectorFactory', 'DetectorPool',
    'detector_factory', 'detector_pool'
]