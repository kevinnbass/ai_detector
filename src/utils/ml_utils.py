"""
Machine Learning Utilities - Shared ML Functions
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

logger = logging.getLogger(__name__)


# ============================================
# Data Preparation
# ============================================

def prepare_train_test_split(X: np.ndarray, y: np.ndarray, 
                           test_size: float = 0.2,
                           random_state: int = 42,
                           stratify: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare train/test split with stratification"""
    stratify_param = y if stratify else None
    return train_test_split(X, y, test_size=test_size, 
                          random_state=random_state, 
                          stratify=stratify_param)


def balance_dataset(X: np.ndarray, y: np.ndarray, 
                   strategy: str = 'undersample') -> Tuple[np.ndarray, np.ndarray]:
    """Balance dataset by undersampling or oversampling"""
    unique, counts = np.unique(y, return_counts=True)
    
    if strategy == 'undersample':
        min_count = min(counts)
        indices = []
        for label in unique:
            label_indices = np.where(y == label)[0]
            selected = np.random.choice(label_indices, min_count, replace=False)
            indices.extend(selected)
    else:  # oversample
        max_count = max(counts)
        indices = []
        for label in unique:
            label_indices = np.where(y == label)[0]
            if len(label_indices) < max_count:
                selected = np.random.choice(label_indices, max_count, replace=True)
            else:
                selected = label_indices
            indices.extend(selected)
    
    indices = np.array(indices)
    np.random.shuffle(indices)
    return X[indices], y[indices]


def normalize_features(X: np.ndarray, method: str = 'standard') -> Tuple[np.ndarray, Any]:
    """Normalize features using standard or minmax scaling"""
    if method == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


# ============================================
# Feature Engineering
# ============================================

def extract_text_features(texts: List[str]) -> np.ndarray:
    """Extract basic text features"""
    features = []
    for text in texts:
        feat = [
            len(text),  # Length
            len(text.split()),  # Word count
            text.count('.'),  # Sentence count approximation
            text.count(','),  # Comma count
            text.count('!'),  # Exclamation count
            text.count('?'),  # Question count
            sum(1 for c in text if c.isupper()) / max(1, len(text)),  # Uppercase ratio
            len(set(text.split())) / max(1, len(text.split()))  # Vocabulary diversity
        ]
        features.append(feat)
    return np.array(features)


def create_ngram_features(texts: List[str], n: int = 2) -> Dict[str, List[float]]:
    """Create n-gram features from texts"""
    from collections import Counter
    
    ngram_counts = {}
    for text in texts:
        words = text.lower().split()
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            if ngram not in ngram_counts:
                ngram_counts[ngram] = []
    
    # Count occurrences
    for ngram in ngram_counts:
        counts = []
        for text in texts:
            count = text.lower().count(ngram)
            counts.append(count)
        ngram_counts[ngram] = counts
    
    return ngram_counts


# ============================================
# Model Evaluation
# ============================================

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, 
                  y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Comprehensive model evaluation"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_proba[:, 1])
        except:
            metrics['auc_roc'] = 0.5
    
    return metrics


def get_confusion_matrix_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Get confusion matrix and derived metrics"""
    cm = confusion_matrix(y_true, y_pred)
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        return {
            'confusion_matrix': cm.tolist(),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0
        }
    
    return {'confusion_matrix': cm.tolist()}


def cross_validate_model(model, X: np.ndarray, y: np.ndarray, 
                        cv: int = 5, scoring: str = 'accuracy') -> Dict[str, float]:
    """Perform cross-validation"""
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    return {
        'mean_score': scores.mean(),
        'std_score': scores.std(),
        'scores': scores.tolist()
    }


# ============================================
# Model Selection
# ============================================

def get_best_threshold(y_true: np.ndarray, y_proba: np.ndarray, 
                      metric: str = 'f1') -> float:
    """Find best classification threshold"""
    best_threshold = 0.5
    best_score = 0
    
    for threshold in np.arange(0.1, 0.9, 0.05):
        y_pred = (y_proba >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred)
        else:
            score = accuracy_score(y_true, y_pred)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold


def ensemble_predictions(predictions: List[np.ndarray], 
                        weights: Optional[List[float]] = None,
                        method: str = 'voting') -> np.ndarray:
    """Ensemble multiple model predictions"""
    if weights is None:
        weights = [1.0] * len(predictions)
    
    if method == 'voting':
        # Hard voting
        votes = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, weights):
            votes += pred * weight
        return (votes >= (sum(weights) / 2)).astype(int)
    
    elif method == 'averaging':
        # Soft voting / averaging
        avg = np.zeros_like(predictions[0], dtype=float)
        total_weight = sum(weights)
        for pred, weight in zip(predictions, weights):
            avg += pred * weight / total_weight
        return avg
    
    else:
        raise ValueError(f"Unknown ensemble method: {method}")


# ============================================
# Feature Selection
# ============================================

def select_top_features(X: np.ndarray, y: np.ndarray, 
                       feature_names: List[str],
                       k: int = 10) -> Tuple[np.ndarray, List[str]]:
    """Select top k features using chi-squared test"""
    from sklearn.feature_selection import SelectKBest, chi2
    
    selector = SelectKBest(chi2, k=min(k, X.shape[1]))
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    selected_indices = selector.get_support(indices=True)
    selected_names = [feature_names[i] for i in selected_indices]
    
    return X_selected, selected_names


def calculate_feature_importance(model, feature_names: List[str]) -> Dict[str, float]:
    """Calculate feature importance from tree-based models"""
    importance_dict = {}
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        for name, importance in zip(feature_names, importances):
            importance_dict[name] = float(importance)
    
    elif hasattr(model, 'coef_'):
        # For linear models
        coef = model.coef_
        if len(coef.shape) > 1:
            coef = coef[0]
        for name, importance in zip(feature_names, np.abs(coef)):
            importance_dict[name] = float(importance)
    
    return importance_dict


# ============================================
# Hyperparameter Tuning
# ============================================

def grid_search_cv(model_class, param_grid: Dict[str, List[Any]], 
                  X: np.ndarray, y: np.ndarray,
                  cv: int = 5, scoring: str = 'f1') -> Tuple[Any, Dict[str, Any]]:
    """Simple grid search for hyperparameter tuning"""
    from sklearn.model_selection import GridSearchCV
    
    grid_search = GridSearchCV(
        model_class(),
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )
    
    grid_search.fit(X, y)
    
    return grid_search.best_estimator_, grid_search.best_params_


# ============================================
# Model Persistence
# ============================================

def save_model(model, filepath: str) -> bool:
    """Save model to file"""
    import joblib
    try:
        joblib.dump(model, filepath)
        return True
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return False


def load_model(filepath: str) -> Optional[Any]:
    """Load model from file"""
    import joblib
    try:
        return joblib.load(filepath)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None


# ============================================
# Validation Utilities
# ============================================

def validate_predictions(y_pred: np.ndarray, num_classes: int = 2) -> bool:
    """Validate prediction array"""
    if len(y_pred.shape) != 1:
        return False
    
    unique_values = np.unique(y_pred)
    if not all(val in range(num_classes) for val in unique_values):
        return False
    
    return True


def validate_probabilities(y_proba: np.ndarray) -> bool:
    """Validate probability array"""
    if len(y_proba.shape) != 2:
        return False
    
    # Check if probabilities sum to 1
    sums = np.sum(y_proba, axis=1)
    if not np.allclose(sums, 1.0):
        return False
    
    # Check if all values are between 0 and 1
    if np.any(y_proba < 0) or np.any(y_proba > 1):
        return False
    
    return True


# Export all utilities
__all__ = [
    # Data Preparation
    'prepare_train_test_split', 'balance_dataset', 'normalize_features',
    
    # Feature Engineering
    'extract_text_features', 'create_ngram_features',
    
    # Model Evaluation
    'evaluate_model', 'get_confusion_matrix_metrics', 'cross_validate_model',
    
    # Model Selection
    'get_best_threshold', 'ensemble_predictions',
    
    # Feature Selection
    'select_top_features', 'calculate_feature_importance',
    
    # Hyperparameter Tuning
    'grid_search_cv',
    
    # Model Persistence
    'save_model', 'load_model',
    
    # Validation
    'validate_predictions', 'validate_probabilities'
]