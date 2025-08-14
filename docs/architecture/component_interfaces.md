# Component Interfaces and Contracts

## Overview

This document defines the interfaces and contracts between all major components of the AI Detector system. These interfaces ensure loose coupling, testability, and maintainability.

## Core Interfaces

### 1. Detection Interfaces

#### IDetector
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from enum import Enum

class DetectionMode(Enum):
    PATTERN = "pattern"
    LLM = "llm"
    HYBRID = "hybrid"

class DetectionResult:
    def __init__(self, is_ai: bool, confidence: float, 
                 indicators: List[str], metadata: Dict[str, Any]):
        self.is_ai = is_ai
        self.confidence = confidence
        self.indicators = indicators
        self.metadata = metadata
        self.timestamp = datetime.utcnow()

class IDetector(ABC):
    """Core detection interface"""
    
    @abstractmethod
    async def detect(self, text: str, mode: DetectionMode = DetectionMode.HYBRID) -> DetectionResult:
        """Detect AI-generated content in text"""
        pass
    
    @abstractmethod
    async def batch_detect(self, texts: List[str], mode: DetectionMode = DetectionMode.HYBRID) -> List[DetectionResult]:
        """Batch detection for multiple texts"""
        pass
    
    @abstractmethod
    def get_supported_modes(self) -> List[DetectionMode]:
        """Get list of supported detection modes"""
        pass
```

#### IDetectionStrategy
```python
class IDetectionStrategy(ABC):
    """Strategy interface for different detection methods"""
    
    @abstractmethod
    async def detect(self, text: str) -> DetectionResult:
        """Execute detection strategy"""
        pass
    
    @abstractmethod
    def get_confidence_threshold(self) -> float:
        """Get minimum confidence threshold"""
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get strategy metadata"""
        pass
```

### 2. Data Access Interfaces

#### IRepository
```python
class IRepository(ABC):
    """Generic repository interface"""
    
    @abstractmethod
    async def save(self, entity: Any) -> str:
        """Save entity and return ID"""
        pass
    
    @abstractmethod
    async def get_by_id(self, id: str) -> Optional[Any]:
        """Get entity by ID"""
        pass
    
    @abstractmethod
    async def get_all(self, filters: Optional[Dict[str, Any]] = None) -> List[Any]:
        """Get all entities with optional filters"""
        pass
    
    @abstractmethod
    async def update(self, id: str, updates: Dict[str, Any]) -> bool:
        """Update entity"""
        pass
    
    @abstractmethod
    async def delete(self, id: str) -> bool:
        """Delete entity"""
        pass
```

#### IDataCollectionRepository
```python
class IDataCollectionRepository(IRepository):
    """Repository for data collection samples"""
    
    @abstractmethod
    async def save_sample(self, sample: CollectionSample) -> str:
        """Save collection sample"""
        pass
    
    @abstractmethod
    async def get_samples_by_label(self, label: str) -> List[CollectionSample]:
        """Get samples by label (ai/human)"""
        pass
    
    @abstractmethod
    async def get_balanced_dataset(self, max_samples: int) -> Dict[str, List[CollectionSample]]:
        """Get balanced dataset"""
        pass
    
    @abstractmethod
    async def get_statistics(self) -> Dict[str, Any]:
        """Get collection statistics"""
        pass
```

#### IModelRepository
```python
class IModelRepository(IRepository):
    """Repository for ML models"""
    
    @abstractmethod
    async def save_model(self, model: Any, metadata: Dict[str, Any]) -> str:
        """Save trained model with metadata"""
        pass
    
    @abstractmethod
    async def load_model(self, model_id: str) -> Tuple[Any, Dict[str, Any]]:
        """Load model with metadata"""
        pass
    
    @abstractmethod
    async def get_best_model(self, metric: str = "f1_score") -> Tuple[str, Any, Dict[str, Any]]:
        """Get best performing model"""
        pass
    
    @abstractmethod
    async def get_model_versions(self) -> List[Dict[str, Any]]:
        """Get all model versions"""
        pass
```

### 3. Service Layer Interfaces

#### IDetectionService
```python
class IDetectionService(ABC):
    """High-level detection service"""
    
    @abstractmethod
    async def detect_text(self, text: str, user_id: Optional[str] = None, 
                         mode: DetectionMode = DetectionMode.HYBRID) -> DetectionResult:
        """Detect AI content with user context"""
        pass
    
    @abstractmethod
    async def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get user detection statistics"""
        pass
    
    @abstractmethod
    async def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        """Update user detection preferences"""
        pass
```

#### ITrainingService
```python
class TrainingRequest:
    def __init__(self, training_data: List[Tuple[str, int]], 
                 model_type: str, hyperparameters: Dict[str, Any]):
        self.training_data = training_data
        self.model_type = model_type
        self.hyperparameters = hyperparameters

class TrainingResult:
    def __init__(self, model_id: str, metrics: Dict[str, float], 
                 model_path: str, training_time: float):
        self.model_id = model_id
        self.metrics = metrics
        self.model_path = model_path
        self.training_time = training_time

class ITrainingService(ABC):
    """Machine learning training service"""
    
    @abstractmethod
    async def train_model(self, request: TrainingRequest) -> TrainingResult:
        """Train new model"""
        pass
    
    @abstractmethod
    async def evaluate_model(self, model_id: str, test_data: List[Tuple[str, int]]) -> Dict[str, float]:
        """Evaluate model performance"""
        pass
    
    @abstractmethod
    async def compare_models(self, model_ids: List[str]) -> Dict[str, Dict[str, float]]:
        """Compare multiple models"""
        pass
    
    @abstractmethod
    async def get_training_history(self) -> List[TrainingResult]:
        """Get training history"""
        pass
```

#### IAnalysisService
```python
class AnalysisRequest:
    def __init__(self, text: str, analysis_type: str, options: Dict[str, Any]):
        self.text = text
        self.analysis_type = analysis_type
        self.options = options

class AnalysisResult:
    def __init__(self, analysis_type: str, results: Dict[str, Any], 
                 confidence: float, processing_time: float):
        self.analysis_type = analysis_type
        self.results = results
        self.confidence = confidence
        self.processing_time = processing_time

class IAnalysisService(ABC):
    """Advanced text analysis service"""
    
    @abstractmethod
    async def analyze_text(self, request: AnalysisRequest) -> AnalysisResult:
        """Perform text analysis"""
        pass
    
    @abstractmethod
    async def get_supported_analysis_types(self) -> List[str]:
        """Get supported analysis types"""
        pass
    
    @abstractmethod
    async def batch_analyze(self, requests: List[AnalysisRequest]) -> List[AnalysisResult]:
        """Batch analysis"""
        pass
```

### 4. External Integration Interfaces

#### ILLMProvider
```python
class LLMRequest:
    def __init__(self, prompt: str, model: str, parameters: Dict[str, Any]):
        self.prompt = prompt
        self.model = model
        self.parameters = parameters

class LLMResponse:
    def __init__(self, text: str, confidence: float, 
                 usage: Dict[str, int], metadata: Dict[str, Any]):
        self.text = text
        self.confidence = confidence
        self.usage = usage
        self.metadata = metadata

class ILLMProvider(ABC):
    """LLM service provider interface"""
    
    @abstractmethod
    async def generate_text(self, request: LLMRequest) -> LLMResponse:
        """Generate text using LLM"""
        pass
    
    @abstractmethod
    async def analyze_text(self, text: str, prompt: str) -> LLMResponse:
        """Analyze text using LLM"""
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Get available models"""
        pass
    
    @abstractmethod
    def get_rate_limits(self) -> Dict[str, int]:
        """Get rate limit information"""
        pass
```

### 5. Configuration Interfaces

#### IConfigurationService
```python
class IConfigurationService(ABC):
    """Configuration management service"""
    
    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """Set configuration value"""
        pass
    
    @abstractmethod
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get configuration section"""
        pass
    
    @abstractmethod
    def reload(self) -> None:
        """Reload configuration"""
        pass
    
    @abstractmethod
    def validate(self) -> List[str]:
        """Validate configuration and return errors"""
        pass
```

### 6. Event System Interfaces

#### IEventBus
```python
class Event:
    def __init__(self, event_type: str, data: Any, source: str):
        self.event_type = event_type
        self.data = data
        self.source = source
        self.timestamp = datetime.utcnow()
        self.id = str(uuid.uuid4())

class IEventHandler(ABC):
    @abstractmethod
    async def handle(self, event: Event) -> None:
        """Handle event"""
        pass

class IEventBus(ABC):
    """Event bus for decoupled communication"""
    
    @abstractmethod
    def subscribe(self, event_type: str, handler: IEventHandler) -> str:
        """Subscribe to event type, returns subscription ID"""
        pass
    
    @abstractmethod
    def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from event"""
        pass
    
    @abstractmethod
    async def publish(self, event: Event) -> None:
        """Publish event"""
        pass
    
    @abstractmethod
    async def publish_async(self, event: Event) -> None:
        """Publish event asynchronously"""
        pass
```

## Chrome Extension Interfaces

### 1. Message Protocol

#### Message Types
```typescript
interface BaseMessage {
    type: string;
    requestId?: string;
    timestamp: number;
}

interface DetectionRequest extends BaseMessage {
    type: 'DETECT_TEXT';
    data: {
        text: string;
        mode?: 'pattern' | 'llm' | 'hybrid';
        options?: Record<string, any>;
    };
}

interface DetectionResponse extends BaseMessage {
    type: 'DETECTION_RESULT';
    data: {
        isAi: boolean;
        confidence: number;
        indicators: string[];
        metadata: Record<string, any>;
    };
}

interface ErrorMessage extends BaseMessage {
    type: 'ERROR';
    data: {
        message: string;
        code?: string;
        details?: any;
    };
}
```

### 2. Content Script Interface

#### IContentScriptHandler
```typescript
interface IContentScriptHandler {
    /**
     * Initialize content script functionality
     */
    initialize(): Promise<void>;
    
    /**
     * Extract text from page elements
     */
    extractText(selector?: string): string[];
    
    /**
     * Highlight detected AI text
     */
    highlightText(elements: HighlightData[]): void;
    
    /**
     * Remove highlights
     */
    clearHighlights(): void;
    
    /**
     * Handle incoming messages
     */
    handleMessage(message: BaseMessage): Promise<BaseMessage>;
}

interface HighlightData {
    text: string;
    confidence: number;
    indicators: string[];
    className: string;
}
```

### 3. Background Service Worker Interface

#### IBackgroundService
```typescript
interface IBackgroundService {
    /**
     * Initialize background service
     */
    initialize(): Promise<void>;
    
    /**
     * Handle detection requests
     */
    handleDetectionRequest(request: DetectionRequest): Promise<DetectionResponse>;
    
    /**
     * Manage API communication
     */
    callPythonAPI(endpoint: string, data: any): Promise<any>;
    
    /**
     * Manage extension state
     */
    getState(): ExtensionState;
    updateState(updates: Partial<ExtensionState>): void;
}

interface ExtensionState {
    isEnabled: boolean;
    detectionMode: 'pattern' | 'llm' | 'hybrid';
    apiKey?: string;
    settings: Record<string, any>;
    statistics: {
        totalDetections: number;
        aiDetected: number;
        humanDetected: number;
    };
}
```

## Data Transfer Objects (DTOs)

### 1. Detection DTOs
```python
@dataclass
class DetectionRequestDTO:
    text: str
    mode: str = "hybrid"
    user_id: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DetectionResponseDTO:
    is_ai: bool
    confidence: float
    indicators: List[str]
    metadata: Dict[str, Any]
    processing_time: float
    model_version: str
```

### 2. Training DTOs
```python
@dataclass
class TrainingDataDTO:
    texts: List[str]
    labels: List[int]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelMetricsDTO:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: Optional[float]
    confusion_matrix: List[List[int]]
```

### 3. Configuration DTOs
```python
@dataclass
class APIConfigDTO:
    base_url: str
    api_key: Optional[str]
    timeout: int = 30
    max_retries: int = 3
    rate_limit: int = 100

@dataclass
class DetectionConfigDTO:
    confidence_threshold: float = 0.7
    enabled_modes: List[str] = field(default_factory=lambda: ["hybrid"])
    cache_enabled: bool = True
    batch_size: int = 10
```

## Interface Implementation Guidelines

### 1. Error Handling
All interface implementations must:
- Use typed exceptions for different error categories
- Include error codes for programmatic handling
- Provide detailed error messages for debugging
- Log errors appropriately

### 2. Async/Await Support
- All I/O operations must be async
- Use proper async context managers
- Handle async exceptions correctly
- Implement timeout handling

### 3. Validation
- Validate all input parameters
- Use type hints and runtime type checking
- Implement schema validation for complex data
- Provide clear validation error messages

### 4. Testing
- All interfaces must be mockable
- Provide test doubles for integration testing
- Include contract tests for implementations
- Implement property-based testing where appropriate

### 5. Documentation
- Use docstrings for all interface methods
- Include usage examples
- Document error conditions
- Specify performance characteristics

This interface definition ensures that all components can be developed, tested, and maintained independently while maintaining system cohesion and reliability.