"""
Unified Trainer Module
Consolidates functionality from trainer.py, enhanced_trainer.py, and active_learner.py
"""

import json
import os
import pickle
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
from datetime import datetime
from pathlib import Path

# ML imports
try:
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report
    )
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingMode(Enum):
    """Training modes available"""
    BASIC = "basic"
    ENHANCED = "enhanced"
    ACTIVE_LEARNING = "active_learning"
    TRANSFER_LEARNING = "transfer_learning"
    ENSEMBLE = "ensemble"


class ModelType(Enum):
    """Supported model types"""
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    SVM = "svm"
    ENSEMBLE = "ensemble"
    CUSTOM = "custom"


@dataclass
class TrainingConfig:
    """Configuration for training"""
    mode: TrainingMode = TrainingMode.BASIC
    model_type: ModelType = ModelType.RANDOM_FOREST
    test_size: float = 0.2
    random_state: int = 42
    max_iterations: int = 100
    early_stopping: bool = True
    patience: int = 10
    validation_split: float = 0.1
    batch_size: int = 32
    learning_rate: float = 0.001
    n_estimators: int = 100
    max_depth: Optional[int] = None
    cross_validation: bool = True
    cv_folds: int = 5
    save_path: str = "models/"
    features: Dict[str, Any] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingResult:
    """Training results container"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: np.ndarray
    classification_report: str
    training_time: float
    model_path: str
    feature_importance: Optional[Dict[str, float]] = None
    cross_val_scores: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TrainingStrategy(ABC):
    """Abstract strategy for different training approaches"""
    
    @abstractmethod
    def train(self, X_train, y_train, X_val, y_val, config: TrainingConfig) -> Any:
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, model, X) -> np.ndarray:
        """Make predictions"""
        pass
    
    @abstractmethod
    def get_feature_importance(self, model) -> Dict[str, float]:
        """Get feature importance if available"""
        pass


class BasicTraining(TrainingStrategy):
    """Basic training strategy"""
    
    def train(self, X_train, y_train, X_val, y_val, config: TrainingConfig):
        """Basic training implementation"""
        if config.model_type == ModelType.RANDOM_FOREST:
            model = RandomForestClassifier(
                n_estimators=config.n_estimators,
                max_depth=config.max_depth,
                random_state=config.random_state
            )
        elif config.model_type == ModelType.GRADIENT_BOOSTING:
            model = GradientBoostingClassifier(
                n_estimators=config.n_estimators,
                learning_rate=config.learning_rate,
                max_depth=config.max_depth or 3,
                random_state=config.random_state
            )
        elif config.model_type == ModelType.SVM:
            model = SVC(
                kernel='rbf',
                random_state=config.random_state,
                probability=True
            )
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")
        
        model.fit(X_train, y_train)
        return model
    
    def predict(self, model, X) -> np.ndarray:
        """Make predictions"""
        return model.predict(X)
    
    def get_feature_importance(self, model) -> Dict[str, float]:
        """Get feature importance"""
        if hasattr(model, 'feature_importances_'):
            return {f"feature_{i}": imp for i, imp in enumerate(model.feature_importances_)}
        return {}


class EnhancedTraining(TrainingStrategy):
    """Enhanced training with advanced features"""
    
    def __init__(self):
        self.best_model = None
        self.best_score = 0
        self.training_history = []
    
    def train(self, X_train, y_train, X_val, y_val, config: TrainingConfig):
        """Enhanced training with early stopping and validation"""
        models = []
        
        # Train multiple models
        if config.model_type == ModelType.ENSEMBLE:
            model_configs = [
                (RandomForestClassifier(n_estimators=100), "RF"),
                (GradientBoostingClassifier(n_estimators=100), "GB"),
                (SVC(probability=True), "SVM")
            ]
        else:
            model_configs = [(self._create_model(config), config.model_type.value)]
        
        for model, name in model_configs:
            logger.info(f"Training {name}...")
            
            # Training with early stopping simulation
            best_val_score = 0
            patience_counter = 0
            
            for epoch in range(config.max_iterations):
                model.fit(X_train, y_train)
                val_score = model.score(X_val, y_val)
                
                self.training_history.append({
                    'epoch': epoch,
                    'model': name,
                    'val_score': val_score
                })
                
                if val_score > best_val_score:
                    best_val_score = val_score
                    patience_counter = 0
                    if val_score > self.best_score:
                        self.best_score = val_score
                        self.best_model = model
                else:
                    patience_counter += 1
                
                if config.early_stopping and patience_counter >= config.patience:
                    logger.info(f"Early stopping triggered for {name} at epoch {epoch}")
                    break
            
            models.append((model, name, best_val_score))
        
        # Return best model or ensemble
        if config.model_type == ModelType.ENSEMBLE:
            return EnsembleModel(models)
        return self.best_model
    
    def predict(self, model, X) -> np.ndarray:
        """Make predictions"""
        if isinstance(model, EnsembleModel):
            return model.predict(X)
        return model.predict(X)
    
    def get_feature_importance(self, model) -> Dict[str, float]:
        """Get aggregated feature importance"""
        if isinstance(model, EnsembleModel):
            return model.get_feature_importance()
        elif hasattr(model, 'feature_importances_'):
            return {f"feature_{i}": imp for i, imp in enumerate(model.feature_importances_)}
        return {}
    
    def _create_model(self, config: TrainingConfig):
        """Create model based on config"""
        if config.model_type == ModelType.RANDOM_FOREST:
            return RandomForestClassifier(
                n_estimators=config.n_estimators,
                max_depth=config.max_depth,
                random_state=config.random_state
            )
        elif config.model_type == ModelType.GRADIENT_BOOSTING:
            return GradientBoostingClassifier(
                n_estimators=config.n_estimators,
                learning_rate=config.learning_rate,
                random_state=config.random_state
            )
        elif config.model_type == ModelType.SVM:
            return SVC(probability=True, random_state=config.random_state)
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")


class ActiveLearningTraining(TrainingStrategy):
    """Active learning strategy"""
    
    def __init__(self, query_strategy: str = "uncertainty"):
        self.query_strategy = query_strategy
        self.labeled_indices = []
        self.unlabeled_indices = []
        self.query_history = []
    
    def train(self, X_train, y_train, X_val, y_val, config: TrainingConfig):
        """Active learning training"""
        # Start with small labeled set
        n_initial = min(10, len(X_train) // 10)
        self.labeled_indices = list(range(n_initial))
        self.unlabeled_indices = list(range(n_initial, len(X_train)))
        
        model = RandomForestClassifier(
            n_estimators=config.n_estimators,
            random_state=config.random_state
        )
        
        for iteration in range(config.max_iterations):
            # Train on labeled data
            X_labeled = X_train[self.labeled_indices]
            y_labeled = y_train[self.labeled_indices]
            
            if len(X_labeled) > 0:
                model.fit(X_labeled, y_labeled)
            
            if not self.unlabeled_indices:
                break
            
            # Query next samples
            X_unlabeled = X_train[self.unlabeled_indices]
            query_indices = self._query_samples(model, X_unlabeled, n_samples=5)
            
            # Move queried samples to labeled set
            for idx in query_indices:
                actual_idx = self.unlabeled_indices[idx]
                self.labeled_indices.append(actual_idx)
                self.query_history.append({
                    'iteration': iteration,
                    'sample_idx': actual_idx,
                    'n_labeled': len(self.labeled_indices)
                })
            
            # Remove from unlabeled
            for idx in sorted(query_indices, reverse=True):
                self.unlabeled_indices.pop(idx)
            
            # Check performance
            if len(X_labeled) % 10 == 0:
                val_score = model.score(X_val, y_val)
                logger.info(f"Active learning iteration {iteration}, "
                          f"labeled samples: {len(self.labeled_indices)}, "
                          f"val score: {val_score:.3f}")
        
        return model
    
    def _query_samples(self, model, X_unlabeled, n_samples: int = 1) -> List[int]:
        """Query samples based on strategy"""
        if self.query_strategy == "uncertainty":
            # Get prediction probabilities
            proba = model.predict_proba(X_unlabeled)
            # Calculate uncertainty (entropy)
            uncertainty = -np.sum(proba * np.log(proba + 1e-10), axis=1)
            # Select most uncertain samples
            return np.argsort(uncertainty)[-n_samples:].tolist()
        elif self.query_strategy == "random":
            return np.random.choice(len(X_unlabeled), n_samples, replace=False).tolist()
        else:
            raise ValueError(f"Unknown query strategy: {self.query_strategy}")
    
    def predict(self, model, X) -> np.ndarray:
        """Make predictions"""
        return model.predict(X)
    
    def get_feature_importance(self, model) -> Dict[str, float]:
        """Get feature importance"""
        if hasattr(model, 'feature_importances_'):
            return {f"feature_{i}": imp for i, imp in enumerate(model.feature_importances_)}
        return {}


class EnsembleModel:
    """Ensemble of multiple models"""
    
    def __init__(self, models: List[Tuple[Any, str, float]]):
        self.models = models
        self.weights = self._calculate_weights()
    
    def _calculate_weights(self) -> np.ndarray:
        """Calculate model weights based on validation scores"""
        scores = [score for _, _, score in self.models]
        total = sum(scores)
        return np.array([s / total for s in scores])
    
    def predict(self, X) -> np.ndarray:
        """Ensemble prediction"""
        predictions = []
        for model, _, _ in self.models:
            pred = model.predict_proba(X) if hasattr(model, 'predict_proba') else model.predict(X)
            predictions.append(pred)
        
        # Weighted average
        if len(predictions[0].shape) > 1:  # Probability predictions
            weighted_pred = np.average(predictions, axis=0, weights=self.weights)
            return np.argmax(weighted_pred, axis=1)
        else:  # Direct predictions
            weighted_pred = np.average(predictions, axis=0, weights=self.weights)
            return (weighted_pred > 0.5).astype(int)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get aggregated feature importance"""
        importance_dict = {}
        for model, name, weight in self.models:
            if hasattr(model, 'feature_importances_'):
                for i, imp in enumerate(model.feature_importances_):
                    key = f"feature_{i}"
                    if key not in importance_dict:
                        importance_dict[key] = 0
                    importance_dict[key] += imp * weight
        return importance_dict


class UnifiedTrainer:
    """
    Unified trainer that consolidates all training functionality
    Replaces: trainer.py, enhanced_trainer.py, active_learner.py
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for training")
        
        self.config = config or TrainingConfig()
        self.strategy = self._get_strategy()
        self.model = None
        self.scaler = StandardScaler()
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.feature_names = []
        self.training_result = None
        
        # Create model directory
        Path(self.config.save_path).mkdir(parents=True, exist_ok=True)
    
    def _get_strategy(self) -> TrainingStrategy:
        """Get training strategy based on mode"""
        if self.config.mode == TrainingMode.BASIC:
            return BasicTraining()
        elif self.config.mode == TrainingMode.ENHANCED:
            return EnhancedTraining()
        elif self.config.mode == TrainingMode.ACTIVE_LEARNING:
            return ActiveLearningTraining()
        else:
            raise ValueError(f"Unsupported training mode: {self.config.mode}")
    
    def prepare_features(self, texts: List[str], labels: Optional[List[int]] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepare features from text data
        
        Args:
            texts: List of text samples
            labels: Optional labels
            
        Returns:
            Feature matrix and labels
        """
        # Text vectorization
        if labels is not None:  # Training mode
            X = self.vectorizer.fit_transform(texts)
            self.feature_names = self.vectorizer.get_feature_names_out().tolist()
        else:  # Prediction mode
            X = self.vectorizer.transform(texts)
        
        # Convert to array
        X = X.toarray()
        
        # Scale features
        if labels is not None:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        
        # Convert labels
        if labels is not None:
            y = np.array(labels)
            return X, y
        
        return X, None
    
    def train(self, texts: List[str], labels: List[int], validation_texts: Optional[List[str]] = None,
              validation_labels: Optional[List[int]] = None) -> TrainingResult:
        """
        Train the model
        
        Args:
            texts: Training texts
            labels: Training labels
            validation_texts: Optional validation texts
            validation_labels: Optional validation labels
            
        Returns:
            Training results
        """
        start_time = datetime.now()
        logger.info(f"Starting training with mode: {self.config.mode}")
        
        # Prepare features
        X, y = self.prepare_features(texts, labels)
        
        # Split data if no validation provided
        if validation_texts is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.config.test_size, random_state=self.config.random_state
            )
        else:
            X_train, y_train = X, y
            X_val, y_val = self.prepare_features(validation_texts, validation_labels)
        
        # Train model
        self.model = self.strategy.train(X_train, y_train, X_val, y_val, self.config)
        
        # Evaluate
        y_pred = self.strategy.predict(self.model, X_val)
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='weighted')
        recall = recall_score(y_val, y_pred, average='weighted')
        f1 = f1_score(y_val, y_pred, average='weighted')
        cm = confusion_matrix(y_val, y_pred)
        report = classification_report(y_val, y_pred)
        
        # Cross-validation if enabled
        cv_scores = None
        if self.config.cross_validation:
            cv_scores = cross_val_score(
                self.model, X, y, cv=self.config.cv_folds, scoring='accuracy'
            ).tolist()
            logger.info(f"Cross-validation scores: {cv_scores}")
            logger.info(f"Mean CV score: {np.mean(cv_scores):.3f}")
        
        # Get feature importance
        feature_importance = self.strategy.get_feature_importance(self.model)
        
        # Save model
        model_path = self._save_model()
        
        # Training time
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Create result
        self.training_result = TrainingResult(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            confusion_matrix=cm,
            classification_report=report,
            training_time=training_time,
            model_path=model_path,
            feature_importance=feature_importance,
            cross_val_scores=cv_scores,
            metadata={
                'mode': self.config.mode.value,
                'model_type': self.config.model_type.value,
                'n_samples': len(X),
                'n_features': X.shape[1],
                'timestamp': datetime.now().isoformat()
            }
        )
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, "
                   f"Recall: {recall:.3f}, F1: {f1:.3f}")
        
        return self.training_result
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Make predictions
        
        Args:
            texts: Texts to predict
            
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X, _ = self.prepare_features(texts, None)
        return self.strategy.predict(self.model, X)
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Get prediction probabilities
        
        Args:
            texts: Texts to predict
            
        Returns:
            Probability array
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X, _ = self.prepare_features(texts, None)
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # Fallback to binary predictions
            preds = self.strategy.predict(self.model, X)
            proba = np.zeros((len(preds), 2))
            proba[range(len(preds)), preds] = 1.0
            return proba
    
    def _save_model(self) -> str:
        """Save trained model"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"model_{self.config.mode.value}_{timestamp}.pkl"
        filepath = Path(self.config.save_path) / filename
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'vectorizer': self.vectorizer,
            'config': self.config,
            'feature_names': self.feature_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
        return str(filepath)
    
    def load_model(self, model_path: str) -> None:
        """Load a saved model"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.vectorizer = model_data['vectorizer']
        self.config = model_data['config']
        self.feature_names = model_data.get('feature_names', [])
        
        # Recreate strategy
        self.strategy = self._get_strategy()
        
        logger.info(f"Model loaded from {model_path}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get training metrics"""
        if self.training_result is None:
            return {}
        
        return {
            'accuracy': self.training_result.accuracy,
            'precision': self.training_result.precision,
            'recall': self.training_result.recall,
            'f1_score': self.training_result.f1_score,
            'training_time': self.training_result.training_time,
            'cross_val_scores': self.training_result.cross_val_scores
        }
    
    def get_feature_importance(self, top_n: int = 20) -> List[Tuple[str, float]]:
        """Get top feature importance"""
        if self.training_result is None or not self.training_result.feature_importance:
            return []
        
        # Map feature indices to names
        importance_with_names = []
        for key, value in self.training_result.feature_importance.items():
            if key.startswith('feature_'):
                idx = int(key.split('_')[1])
                if idx < len(self.feature_names):
                    importance_with_names.append((self.feature_names[idx], value))
            else:
                importance_with_names.append((key, value))
        
        # Sort by importance
        importance_with_names.sort(key=lambda x: x[1], reverse=True)
        
        return importance_with_names[:top_n]


# Factory function for backward compatibility
def create_trainer(mode: str = 'basic', **kwargs) -> UnifiedTrainer:
    """Create a trainer instance"""
    training_mode = TrainingMode(mode)
    config = TrainingConfig(mode=training_mode, **kwargs)
    return UnifiedTrainer(config)


if __name__ == "__main__":
    # Example usage
    trainer = create_trainer(mode='enhanced', model_type=ModelType.ENSEMBLE)
    
    # Sample data
    texts = ["AI text example"] * 50 + ["Human text example"] * 50
    labels = [1] * 50 + [0] * 50
    
    # Train
    result = trainer.train(texts, labels)
    print(f"Training result: {result.accuracy:.3f}")
    
    # Get feature importance
    importance = trainer.get_feature_importance(top_n=10)
    print(f"Top features: {importance}")