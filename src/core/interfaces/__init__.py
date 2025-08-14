"""
Core Interface Definitions
Standardized contracts for all system modules
"""

from .base_interfaces import *
from .detector_interfaces import *
from .data_interfaces import *
from .service_interfaces import *
from .api_interfaces import *
from .extension_interfaces import *
from .ml_interfaces import *
from .pipeline_interfaces import *

__all__ = [
    # Base interfaces
    'IInitializable',
    'IConfigurable',
    'IHealthCheckable',
    'IMetricsProvider',
    'ILoggable',
    'IDisposable',
    
    # Detection interfaces
    'IDetector',
    'IPatternDetector',
    'ILLMDetector',
    'IMLDetector',
    'IEnsembleDetector',
    'IDetectionResult',
    'IDetectionConfig',
    
    # Data interfaces
    'IDataCollector',
    'IDataProcessor',
    'IDataValidator',
    'IDataExporter',
    'IDataImporter',
    'IDataTransformer',
    'IDataSource',
    'IDataSink',
    
    # Service interfaces
    'IAnalysisService',
    'IDetectionService',
    'ITrainingService',
    'ICacheService',
    'IAuthenticationService',
    'IConfigurationService',
    
    # API interfaces
    'IAPIClient',
    'IAPIHandler',
    'IRequestValidator',
    'IResponseFormatter',
    'IWebSocketHandler',
    'IMiddleware',
    
    # Extension interfaces
    'IContentScript',
    'IBackgroundScript',
    'IPopupHandler',
    'IExtensionStorage',
    'IMessageHandler',
    
    # ML interfaces
    'IFeatureExtractor',
    'IModelTrainer',
    'IModelEvaluator',
    'IModelPredictor',
    'IDatasetHandler',
    
    # Pipeline interfaces
    'IPipelineStage',
    'IPipelineOrchestrator',
    'IPipelineMonitor',
    'IPipelineConfig'
]