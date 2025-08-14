# System Interfaces Documentation

This document describes the comprehensive interface system that defines contracts between all modules in the AI Detector system.

## Overview

The interface system provides:

- **Clear contracts** between components
- **Loose coupling** for better maintainability  
- **Standardized communication** across modules
- **Easy testing** with interface mocking
- **Flexible implementation** swapping

## Interface Categories

### 1. Base Interfaces (`base_interfaces.py`)

Fundamental interfaces that other interfaces build upon:

- `IInitializable` - Components requiring initialization
- `IConfigurable` - Configurable components
- `IHealthCheckable` - Health monitoring support
- `IMetricsProvider` - Metrics collection
- `ILoggable` - Logging capabilities
- `IDisposable` - Resource cleanup
- `IValidatable` - Validation support
- `IMonitorable` - Monitoring capabilities

### 2. Detector Interfaces (`detector_interfaces.py`)

AI detection system contracts:

- `IDetector` - Base detector interface
- `IPatternDetector` - Pattern-based detection
- `ILLMDetector` - LLM-based detection  
- `IMLDetector` - Machine learning detection
- `IEnsembleDetector` - Multiple detector combination
- `IDetectionResult` - Standardized results
- `IDetectionConfig` - Configuration contracts

### 3. Data Interfaces (`data_interfaces.py`)

Data processing and management contracts:

- `IDataSource` / `IDataSink` - Data input/output
- `IDataCollector` - Data collection
- `IDataProcessor` - Data processing
- `IDataValidator` - Data validation
- `IDataTransformer` - Data transformation
- `IDataExporter` / `IDataImporter` - Data exchange
- `IDataRepository` - Data storage

### 4. Service Interfaces (`service_interfaces.py`)

Service layer contracts:

- `IBaseService` - Common service functionality
- `IAnalysisService` - Text analysis services
- `IDetectionService` - Detection orchestration
- `ITrainingService` - Model training
- `ICacheService` - Caching functionality
- `IAuthenticationService` - Authentication
- `IConfigurationService` - Configuration management

### 5. API Interfaces (`api_interfaces.py`)

API layer contracts:

- `IAPIClient` - API client functionality
- `IAPIHandler` - Request handling
- `IRequestValidator` - Request validation
- `IResponseFormatter` - Response formatting
- `IMiddleware` - Middleware processing
- `IWebSocketHandler` - WebSocket support
- `IAPIGateway` - API routing

### 6. Extension Interfaces (`extension_interfaces.py`)

Chrome extension contracts:

- `IMessageHandler` - Message processing
- `IMessageBus` - Inter-component communication
- `IExtensionStorage` - Storage management
- `IBackgroundScript` - Background functionality
- `IContentScript` - Content script operations
- `IPopupHandler` - Popup interface

### 7. ML Interfaces (`ml_interfaces.py`)

Machine learning contracts:

- `IFeatureExtractor` - Feature extraction
- `IModelTrainer` - Model training
- `IModelEvaluator` - Model evaluation
- `IModelPredictor` - Prediction functionality
- `IDatasetHandler` - Dataset management
- `IModelRegistry` - Model storage/retrieval

### 8. Pipeline Interfaces (`pipeline_interfaces.py`)

Processing pipeline contracts:

- `IPipelineStage` - Individual pipeline stages
- `IPipelineOrchestrator` - Pipeline coordination
- `IPipelineMonitor` - Pipeline monitoring
- `IPipelineBuilder` - Pipeline construction
- `IWorkflowEngine` - Workflow management

## Interface Design Principles

### 1. Single Responsibility
Each interface defines a single, focused concern:

```python
# Good: Focused on detection only
class IDetector(ABC):
    @abstractmethod
    async def detect(self, text: str) -> IDetectionResult:
        pass

# Bad: Mixed concerns
class IDetectorAndTrainer(ABC):
    @abstractmethod
    async def detect(self, text: str) -> IDetectionResult:
        pass
    
    @abstractmethod
    async def train(self, data: List[Sample]) -> str:
        pass
```

### 2. Interface Segregation
Clients depend only on interfaces they use:

```python
# Segregated interfaces
class IDetector(ABC):
    async def detect(self, text: str) -> IDetectionResult: pass

class IExplainableDetector(IDetector):
    async def explain(self, text: str, result: IDetectionResult) -> Dict: pass

# Usage - only implement what you need
class SimpleDetector(IDetector):  # Doesn't need explanation
    pass

class AdvancedDetector(IExplainableDetector):  # Needs both
    pass
```

### 3. Dependency Inversion
Depend on abstractions, not concretions:

```python
# Good: Depends on interface
class AnalysisService:
    def __init__(self, detector: IDetector):
        self.detector = detector

# Bad: Depends on concrete class  
class AnalysisService:
    def __init__(self, detector: PatternDetector):
        self.detector = detector
```

### 4. Liskov Substitution
Implementations are substitutable:

```python
# All implementations work the same way
detector1: IDetector = PatternDetector()
detector2: IDetector = LLMDetector()  
detector3: IDetector = MLDetector()

# Can be used interchangeably
async def analyze(detector: IDetector, text: str):
    return await detector.detect(text)
```

## Integration Patterns

### 1. Layered Architecture
Interfaces enable clean layers:

```
┌─────────────────┐
│   API Layer     │ ← IAPIHandler, IMiddleware
├─────────────────┤
│ Service Layer   │ ← IAnalysisService, IDetectionService  
├─────────────────┤
│ Domain Layer    │ ← IDetector, IDataProcessor
├─────────────────┤
│ Data Layer      │ ← IRepository, IDataSource
└─────────────────┘
```

### 2. Plugin Architecture
Easy component swapping:

```python
# Register different implementations
detector_factory.register("pattern", PatternDetector)
detector_factory.register("llm", LLMDetector)
detector_factory.register("ml", MLDetector)

# Create based on configuration
detector = detector_factory.create(config.detector_type)
```

### 3. Pipeline Composition
Chain processing stages:

```python
pipeline = PipelineBuilder() \
    .add_stage(DataValidationStage()) \
    .add_stage(PreprocessingStage()) \
    .add_stage(DetectionStage()) \
    .add_stage(PostprocessingStage()) \
    .build()
```

## Testing with Interfaces

### 1. Mock Implementations
Easy testing with mocks:

```python
class MockDetector(IDetector):
    async def detect(self, text: str) -> IDetectionResult:
        return MockDetectionResult(ai_probability=0.8)

# Test with mock
service = AnalysisService(MockDetector())
result = await service.analyze("test")
assert result.success
```

### 2. Interface Contracts
Test that implementations satisfy contracts:

```python
@pytest.fixture(params=[PatternDetector, LLMDetector, MLDetector])
def detector(request):
    return request.param()

async def test_detector_interface(detector: IDetector):
    result = await detector.detect("test text")
    assert isinstance(result, IDetectionResult)
    assert hasattr(result, 'get_score')
```

## Best Practices

### 1. Interface Design
- Keep interfaces small and focused
- Use descriptive method names
- Include comprehensive docstrings
- Define clear return types
- Handle errors consistently

### 2. Implementation Guidelines
- Implement all interface methods
- Follow interface contracts exactly
- Provide meaningful error messages
- Include proper logging
- Validate input parameters

### 3. Documentation
- Document all interfaces thoroughly
- Provide usage examples
- Explain integration patterns
- Include performance considerations
- Document error conditions

## Usage Examples

See `integration_examples.py` for comprehensive examples of:

- Service orchestration
- Pipeline integration
- API to service communication
- Extension to backend integration
- Multi-detector composition

## Interface Validation

The system includes tools to validate interface compliance:

```python
from src.core.interfaces import validate_implementation

# Validate that implementation satisfies interface
is_valid, errors = validate_implementation(
    implementation=MyDetector(),
    interface=IDetector
)

if not is_valid:
    print(f"Interface validation failed: {errors}")
```

## Migration Guide

When updating interfaces:

1. **Extend, don't break** - Add new methods as optional
2. **Deprecate gradually** - Mark old methods as deprecated
3. **Update implementations** - Ensure all implementations comply
4. **Test thoroughly** - Run interface validation tests
5. **Document changes** - Update interface documentation

## Performance Considerations

- Interface calls have minimal overhead
- Use async interfaces for I/O operations
- Consider caching for expensive operations
- Monitor interface call patterns
- Optimize hot paths while maintaining contracts

This interface system provides a solid foundation for building maintainable, testable, and flexible AI detection systems.