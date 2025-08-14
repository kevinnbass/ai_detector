"""
Machine Learning Interface Definitions
Interfaces for ML components and model management
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple, Iterator
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
import numpy as np

from .base_interfaces import IInitializable, IConfigurable, IValidatable, IMetricsProvider
from .data_interfaces import DataSample, DataBatch


class ModelType(Enum):
    """Model type enumeration"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    NEURAL_NETWORK = "neural_network"
    TRANSFORMER = "transformer"
    ENSEMBLE = "ensemble"


class TrainingStatus(Enum):
    """Training status enumeration"""
    NOT_STARTED = "not_started"
    INITIALIZING = "initializing"
    TRAINING = "training"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OptimizationObjective(Enum):
    """Optimization objective enumeration"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    AUC_ROC = "auc_roc"
    LOSS = "loss"


@dataclass
class ModelMetrics:
    """Model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: Optional[float] = None
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: Optional[Dict[str, Any]] = None
    training_time: Optional[float] = None
    inference_time: Optional[float] = None


@dataclass
class TrainingConfig:
    """Training configuration"""
    model_type: ModelType
    hyperparameters: Dict[str, Any]
    optimization_objective: OptimizationObjective
    max_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_split: float = 0.2
    early_stopping: bool = True
    save_checkpoints: bool = True


class IFeatureExtractor(IInitializable, IConfigurable, ABC):
    """Interface for feature extractors"""
    
    @abstractmethod
    def extract_features(self, text: str) -> Dict[str, float]:
        """Extract features from text"""
        pass
    
    @abstractmethod
    def extract_batch_features(self, texts: List[str]) -> List[Dict[str, float]]:
        """Extract features from batch of texts"""
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get feature names"""
        pass
    
    @abstractmethod
    def get_feature_dimension(self) -> int:
        """Get feature vector dimension"""
        pass
    
    @abstractmethod
    def fit(self, texts: List[str]) -> None:
        """Fit feature extractor to data"""
        pass
    
    @abstractmethod
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to feature matrix"""
        pass
    
    @abstractmethod
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit and transform texts"""
        pass


class IModelTrainer(IInitializable, IConfigurable, IValidatable, ABC):
    """Interface for model trainers"""
    
    @abstractmethod
    async def train(self, training_data: List[DataSample], 
                   config: TrainingConfig) -> str:
        """Train model and return model ID"""
        pass
    
    @abstractmethod
    def get_training_status(self, training_id: str) -> TrainingStatus:
        """Get training status"""
        pass
    
    @abstractmethod
    def get_training_progress(self, training_id: str) -> Dict[str, Any]:
        """Get training progress"""
        pass
    
    @abstractmethod
    def cancel_training(self, training_id: str) -> bool:
        """Cancel training"""
        pass
    
    @abstractmethod
    def get_training_logs(self, training_id: str) -> List[str]:
        """Get training logs"""
        pass
    
    @abstractmethod
    def save_model(self, model: Any, model_path: str) -> bool:
        """Save trained model"""
        pass
    
    @abstractmethod
    def load_model(self, model_path: str) -> Any:
        """Load trained model"""
        pass


class IModelEvaluator(IInitializable, ABC):
    """Interface for model evaluators"""
    
    @abstractmethod
    def evaluate(self, model: Any, test_data: List[DataSample]) -> ModelMetrics:
        """Evaluate model performance"""
        pass
    
    @abstractmethod
    def cross_validate(self, model: Any, data: List[DataSample], 
                      folds: int = 5) -> Dict[str, List[float]]:
        """Perform cross-validation"""
        pass
    
    @abstractmethod
    def compare_models(self, models: List[Any], test_data: List[DataSample]) -> Dict[str, ModelMetrics]:
        """Compare multiple models"""
        pass
    
    @abstractmethod
    def generate_confusion_matrix(self, y_true: List[str], y_pred: List[str]) -> np.ndarray:
        """Generate confusion matrix"""
        pass
    
    @abstractmethod
    def generate_classification_report(self, y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
        """Generate classification report"""
        pass


class IModelPredictor(IInitializable, ABC):
    """Interface for model predictors"""
    
    @abstractmethod
    def predict(self, text: str) -> Dict[str, Any]:
        """Make single prediction"""
        pass
    
    @abstractmethod
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Make batch predictions"""
        pass
    
    @abstractmethod
    def predict_proba(self, text: str) -> Dict[str, float]:
        """Get prediction probabilities"""
        pass
    
    @abstractmethod
    def predict_proba_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """Get batch prediction probabilities"""
        pass
    
    @abstractmethod
    def explain_prediction(self, text: str) -> Dict[str, Any]:
        """Explain prediction"""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        pass


class IHyperparameterOptimizer(IInitializable, IConfigurable, ABC):
    """Interface for hyperparameter optimization"""
    
    @abstractmethod
    def optimize(self, model_class: type, training_data: List[DataSample],
                validation_data: List[DataSample], 
                param_space: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize hyperparameters"""
        pass
    
    @abstractmethod
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization history"""
        pass
    
    @abstractmethod
    def suggest_parameters(self, param_space: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest next parameters to try"""
        pass
    
    @abstractmethod
    def update_result(self, params: Dict[str, Any], score: float) -> None:
        """Update optimization with result"""
        pass


class IDatasetHandler(IInitializable, ABC):
    """Interface for dataset handling"""
    
    @abstractmethod
    def load_dataset(self, dataset_path: str) -> List[DataSample]:
        """Load dataset from file"""
        pass
    
    @abstractmethod
    def save_dataset(self, samples: List[DataSample], dataset_path: str) -> bool:
        """Save dataset to file"""
        pass
    
    @abstractmethod
    def split_dataset(self, samples: List[DataSample], 
                     train_ratio: float, val_ratio: float, 
                     test_ratio: float) -> Tuple[List[DataSample], List[DataSample], List[DataSample]]:
        """Split dataset into train/val/test"""
        pass
    
    @abstractmethod
    def balance_dataset(self, samples: List[DataSample]) -> List[DataSample]:
        """Balance dataset classes"""
        pass
    
    @abstractmethod
    def get_dataset_stats(self, samples: List[DataSample]) -> Dict[str, Any]:
        """Get dataset statistics"""
        pass
    
    @abstractmethod
    def validate_dataset(self, samples: List[DataSample]) -> tuple[bool, List[str]]:
        """Validate dataset"""
        pass


class IModelRegistry(ABC):
    """Interface for model registry"""
    
    @abstractmethod
    def register_model(self, model_id: str, model: Any, 
                      metadata: Dict[str, Any]) -> bool:
        """Register model in registry"""
        pass
    
    @abstractmethod
    def get_model(self, model_id: str) -> Optional[Any]:
        """Get model from registry"""
        pass
    
    @abstractmethod
    def list_models(self, model_type: Optional[ModelType] = None) -> List[Dict[str, Any]]:
        """List registered models"""
        pass
    
    @abstractmethod
    def delete_model(self, model_id: str) -> bool:
        """Delete model from registry"""
        pass
    
    @abstractmethod
    def update_model_metadata(self, model_id: str, metadata: Dict[str, Any]) -> bool:
        """Update model metadata"""
        pass
    
    @abstractmethod
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model information"""
        pass


class IModelVersioning(ABC):
    """Interface for model versioning"""
    
    @abstractmethod
    def create_version(self, model_id: str, model: Any, 
                      version_info: Dict[str, Any]) -> str:
        """Create new model version"""
        pass
    
    @abstractmethod
    def get_version(self, model_id: str, version: str) -> Optional[Any]:
        """Get specific model version"""
        pass
    
    @abstractmethod
    def list_versions(self, model_id: str) -> List[Dict[str, Any]]:
        """List model versions"""
        pass
    
    @abstractmethod
    def set_active_version(self, model_id: str, version: str) -> bool:
        """Set active model version"""
        pass
    
    @abstractmethod
    def compare_versions(self, model_id: str, version1: str, version2: str) -> Dict[str, Any]:
        """Compare model versions"""
        pass


class IAutoML(IInitializable, IConfigurable, ABC):
    """Interface for AutoML systems"""
    
    @abstractmethod
    async def auto_train(self, training_data: List[DataSample], 
                        target_metric: str = "f1_score",
                        time_budget: int = 3600) -> Dict[str, Any]:
        """Automatically train best model"""
        pass
    
    @abstractmethod
    def get_model_recommendations(self, data_stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get model recommendations"""
        pass
    
    @abstractmethod
    def auto_feature_engineering(self, samples: List[DataSample]) -> List[DataSample]:
        """Automatic feature engineering"""
        pass
    
    @abstractmethod
    def get_training_pipeline(self, model_id: str) -> Dict[str, Any]:
        """Get training pipeline configuration"""
        pass


class IModelMonitor(IInitializable, IMetricsProvider, ABC):
    """Interface for model monitoring"""
    
    @abstractmethod
    def monitor_model(self, model_id: str, predictions: List[Dict[str, Any]]) -> None:
        """Monitor model predictions"""
        pass
    
    @abstractmethod
    def detect_drift(self, model_id: str, new_data: List[DataSample]) -> Dict[str, Any]:
        """Detect data/concept drift"""
        pass
    
    @abstractmethod
    def get_model_performance(self, model_id: str, 
                             time_window: int = 3600) -> Dict[str, Any]:
        """Get model performance metrics"""
        pass
    
    @abstractmethod
    def set_performance_threshold(self, model_id: str, metric: str, 
                                 threshold: float) -> None:
        """Set performance threshold"""
        pass
    
    @abstractmethod
    def get_alerts(self, model_id: str) -> List[Dict[str, Any]]:
        """Get model alerts"""
        pass


class IExperimentTracker(ABC):
    """Interface for experiment tracking"""
    
    @abstractmethod
    def start_experiment(self, name: str, config: Dict[str, Any]) -> str:
        """Start new experiment"""
        pass
    
    @abstractmethod
    def log_metric(self, experiment_id: str, metric_name: str, 
                  value: float, step: Optional[int] = None) -> None:
        """Log experiment metric"""
        pass
    
    @abstractmethod
    def log_parameter(self, experiment_id: str, param_name: str, 
                     value: Any) -> None:
        """Log experiment parameter"""
        pass
    
    @abstractmethod
    def log_artifact(self, experiment_id: str, artifact_path: str) -> None:
        """Log experiment artifact"""
        pass
    
    @abstractmethod
    def end_experiment(self, experiment_id: str, 
                      status: str = "completed") -> None:
        """End experiment"""
        pass
    
    @abstractmethod
    def compare_experiments(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """Compare experiments"""
        pass