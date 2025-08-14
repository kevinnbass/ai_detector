import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
import os
import re
from collections import Counter

from data_collector import DataCollector
from detector import GPT4oDetector

class GPT4oTrainer:
    def __init__(self, data_file: str = "../data/labeled_dataset.json"):
        self.data_collector = DataCollector(data_file)
        self.models = {}
        self.feature_extractors = {}
        self.training_history = []
        
    def extract_features(self, texts: List[str]) -> np.ndarray:
        """Extract comprehensive features from text samples"""
        features = []
        
        for text in texts:
            text_features = self._extract_single_text_features(text)
            features.append(text_features)
        
        return np.array(features)
    
    def _extract_single_text_features(self, text: str) -> List[float]:
        """Extract features from a single text sample"""
        features = []
        
        # Basic statistics
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        features.extend([
            len(text),  # Character count
            len(words),  # Word count
            len(sentences),  # Sentence count
            len(words) / len(sentences) if sentences else 0,  # Avg words per sentence
            len(set(words)) / len(words) if words else 0,  # Lexical diversity
        ])
        
        # GPT-4o specific patterns
        pattern_features = self._extract_pattern_features(text)
        features.extend(pattern_features)
        
        # Statistical features
        stat_features = self._extract_statistical_features(text, words, sentences)
        features.extend(stat_features)
        
        # Stylistic features
        style_features = self._extract_stylistic_features(text)
        features.extend(style_features)
        
        return features
    
    def _extract_pattern_features(self, text: str) -> List[float]:
        """Extract GPT-4o specific pattern features"""
        text_lower = text.lower()
        
        # Hedging patterns
        hedge_words = ['perhaps', 'maybe', 'possibly', 'might', 'could', 'seems', 'appears', 
                      'likely', 'probably', 'generally', 'typically', 'often', 'sometimes']
        hedge_count = sum(1 for word in text_lower.split() if word in hedge_words)
        
        # Contrast patterns
        contrast_patterns = [
            r'not\s+\w+,?\s+but\s+\w+',
            r'while\s+.*,\s+',
            r'although\s+.*,\s+',
            r'however,?\s+',
            r'on\s+the\s+other\s+hand',
        ]
        contrast_count = sum(1 for pattern in contrast_patterns if re.search(pattern, text_lower))
        
        # Formal language
        formal_words = ['furthermore', 'moreover', 'consequently', 'therefore', 'thus', 
                       'hence', 'accordingly', 'nevertheless', 'nonetheless']
        formal_count = sum(1 for word in text_lower.split() if word in formal_words)
        
        # List indicators
        list_patterns = [r'firstly', r'secondly', r'thirdly', r'first,', r'second,', r'third,']
        list_count = sum(1 for pattern in list_patterns if re.search(pattern, text_lower))
        
        # Qualifier phrases
        qualifiers = ['important to note', 'worth noting', 'keep in mind', 'bear in mind', 
                     'remember that', 'consider that']
        qualifier_count = sum(1 for phrase in qualifiers if phrase in text_lower)
        
        # Balanced presentation
        balance_patterns = ['advantages.*disadvantages', 'pros.*cons', 'benefits.*drawbacks']
        balance_count = sum(1 for pattern in balance_patterns if re.search(pattern, text_lower))
        
        # Explanatory language
        explain_words = ['essentially', 'basically', 'in other words', 'simply put', 'in essence']
        explain_count = sum(1 for phrase in explain_words if phrase in text_lower)
        
        # Caveats
        caveat_phrases = ['that said', 'having said that', 'with that in mind', 'to be fair']
        caveat_count = sum(1 for phrase in caveat_phrases if phrase in text_lower)
        
        return [
            hedge_count / len(text.split()) if text.split() else 0,
            contrast_count,
            formal_count / len(text.split()) if text.split() else 0,
            list_count,
            qualifier_count,
            balance_count,
            explain_count,
            caveat_count
        ]
    
    def _extract_statistical_features(self, text: str, words: List[str], sentences: List[str]) -> List[float]:
        """Extract statistical text features"""
        if not sentences or not words:
            return [0] * 8
        
        # Sentence length statistics
        sentence_lengths = [len(s.split()) for s in sentences]
        
        # Punctuation analysis
        punctuation_count = sum(1 for char in text if char in '.,;:!?()[]{}"\'-')
        punctuation_ratio = punctuation_count / len(text) if text else 0
        
        # Capitalization patterns
        capital_ratio = sum(1 for char in text if char.isupper()) / len(text) if text else 0
        
        # Word length statistics
        word_lengths = [len(word) for word in words]
        
        return [
            np.mean(sentence_lengths),
            np.var(sentence_lengths),
            np.max(sentence_lengths),
            np.min(sentence_lengths),
            punctuation_ratio,
            capital_ratio,
            np.mean(word_lengths),
            np.var(word_lengths)
        ]
    
    def _extract_stylistic_features(self, text: str) -> List[float]:
        """Extract stylistic features"""
        # Question marks and exclamation points
        question_count = text.count('?')
        exclamation_count = text.count('!')
        
        # Parentheses usage
        paren_count = text.count('(') + text.count('[')
        
        # Quotation usage
        quote_count = text.count('"') + text.count("'")
        
        # Dash usage
        dash_count = text.count('-') + text.count('—')
        
        # Ellipsis usage
        ellipsis_count = text.count('...')
        
        # Emoji/emoticon detection (basic)
        emoji_pattern = r'[😀-🙏🌀-🗿]'
        emoji_count = len(re.findall(emoji_pattern, text))
        
        return [
            question_count,
            exclamation_count, 
            paren_count,
            quote_count,
            dash_count,
            ellipsis_count,
            emoji_count
        ]
    
    def prepare_training_data(self, min_samples: int = 50) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features and labels for training"""
        dataset = self.data_collector.dataset
        
        if len(dataset) < min_samples:
            raise ValueError(f"Need at least {min_samples} samples, got {len(dataset)}")
        
        # Check class balance
        gpt4o_count = sum(1 for s in dataset if s['label'] == 'gpt4o')
        human_count = sum(1 for s in dataset if s['label'] == 'human')
        
        if min(gpt4o_count, human_count) < min_samples // 4:
            raise ValueError(f"Need at least {min_samples//4} samples per class")
        
        texts = [sample['text'] for sample in dataset]
        labels = [1 if sample['label'] == 'gpt4o' else 0 for sample in dataset]
        
        # Extract features
        print("Extracting features...")
        X = self.extract_features(texts)
        y = np.array(labels)
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Class distribution - GPT-4o: {gpt4o_count}, Human: {human_count}")
        
        return X, y, texts
    
    def train_models(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Dict[str, Any]:
        """Train multiple models and compare performance"""
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        
        # Define models to try
        models_to_train = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'svm': SVC(random_state=42, probability=True)
        }
        
        results = {}
        
        for name, model in models_to_train.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            # Calculate metrics
            results[name] = {
                'model': model,
                'accuracy': np.mean(y_pred == y_test),
                'cv_accuracy': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'roc_auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None,
                'predictions': {
                    'y_test': y_test.tolist(),
                    'y_pred': y_pred.tolist(),
                    'y_pred_proba': y_pred_proba.tolist() if y_pred_proba is not None else None
                }
            }
            
            print(f"  Accuracy: {results[name]['accuracy']:.3f}")
            print(f"  CV Accuracy: {results[name]['cv_accuracy']:.3f} ± {results[name]['cv_std']:.3f}")
            if results[name]['roc_auc']:
                print(f"  ROC AUC: {results[name]['roc_auc']:.3f}")
        
        # Save best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['cv_accuracy'])
        self.models['best'] = results[best_model_name]['model']
        
        print(f"\nBest model: {best_model_name}")
        
        return results
    
    def evaluate_current_detector(self, X: np.ndarray, y: np.ndarray, texts: List[str]) -> Dict[str, Any]:
        """Evaluate the current rule-based detector against labeled data"""
        detector = GPT4oDetector()
        
        predictions = []
        confidences = []
        
        for text in texts:
            result = detector.detect(text)
            predictions.append(1 if result.is_gpt4o else 0)
            confidences.append(result.confidence)
        
        y_pred = np.array(predictions)
        accuracy = np.mean(y_pred == y)
        
        results = {
            'accuracy': accuracy,
            'classification_report': classification_report(y, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
            'roc_auc': roc_auc_score(y, confidences),
            'avg_confidence': np.mean(confidences)
        }
        
        print(f"\nCurrent Rule-Based Detector Performance:")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  ROC AUC: {results['roc_auc']:.3f}")
        
        return results
    
    def generate_training_report(self, model_results: Dict[str, Any], 
                               current_detector_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive training report"""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_stats': self.data_collector.get_statistics(),
            'model_comparison': {},
            'current_detector_performance': current_detector_results,
            'recommendations': []
        }
        
        # Compare models
        for name, results in model_results.items():
            report['model_comparison'][name] = {
                'accuracy': results['accuracy'],
                'cv_accuracy': results['cv_accuracy'],
                'cv_std': results['cv_std'],
                'roc_auc': results['roc_auc'],
                'precision': results['classification_report']['1']['precision'],
                'recall': results['classification_report']['1']['recall'],
                'f1_score': results['classification_report']['1']['f1-score']
            }
        
        # Generate recommendations
        best_ml_accuracy = max(r['cv_accuracy'] for r in model_results.values())
        current_accuracy = current_detector_results['accuracy']
        
        if best_ml_accuracy > current_accuracy + 0.1:
            report['recommendations'].append(
                f"ML model shows {(best_ml_accuracy - current_accuracy)*100:.1f}% improvement over current detector"
            )
        
        if report['dataset_stats']['total_samples'] < 200:
            report['recommendations'].append(
                "Consider collecting more training data (target: 200+ samples per class)"
            )
        
        balance_ratio = report['dataset_stats']['balance_ratio']
        if balance_ratio < 0.5 or balance_ratio > 2.0:
            report['recommendations'].append(
                f"Dataset is imbalanced (ratio: {balance_ratio:.2f}). Consider balancing classes."
            )
        
        return report
    
    def save_model(self, model_name: str = 'best', output_dir: str = "../models/"):
        """Save trained model and feature extractor"""
        os.makedirs(output_dir, exist_ok=True)
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        # Save model
        model_path = os.path.join(output_dir, f"{model_name}_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(self.models[model_name], f)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load a saved model"""
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        self.models['loaded'] = model
        print(f"Model loaded from {model_path}")
        return model
    
    def predict(self, texts: List[str], model_name: str = 'best') -> List[Dict[str, Any]]:
        """Make predictions using trained model"""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]
        X = self.extract_features(texts)
        
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
        
        results = []
        for i, text in enumerate(texts):
            results.append({
                'text': text,
                'is_gpt4o': bool(predictions[i]),
                'confidence': probabilities[i] if probabilities is not None else predictions[i],
                'label': 'gpt4o' if predictions[i] else 'human'
            })
        
        return results

def main():
    """Main training pipeline"""
    print("GPT-4o Detector Training Pipeline")
    print("="*50)
    
    # Initialize trainer
    trainer = GPT4oTrainer()
    
    # Check dataset size
    stats = trainer.data_collector.get_statistics()
    print(f"Dataset: {stats['total_samples']} samples")
    print(f"  GPT-4o: {stats['gpt4o_samples']}")
    print(f"  Human: {stats['human_samples']}")
    
    if stats['total_samples'] < 50:
        print("\n⚠️  WARNING: Dataset too small for reliable training!")
        print("   Recommendation: Collect at least 100 samples (50 per class)")
        print("   Run: python data_collector.py")
        return
    
    try:
        # Prepare training data
        X, y, texts = trainer.prepare_training_data()
        
        # Evaluate current detector
        print("\nEvaluating current rule-based detector...")
        current_results = trainer.evaluate_current_detector(X, y, texts)
        
        # Train ML models
        print("\nTraining machine learning models...")
        model_results = trainer.train_models(X, y)
        
        # Generate report
        report = trainer.generate_training_report(model_results, current_results)
        
        # Save report
        os.makedirs("../reports", exist_ok=True)
        report_path = f"../reports/training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nTraining report saved to: {report_path}")
        
        # Save best model
        trainer.save_model()
        
        # Print recommendations
        print("\n" + "="*50)
        print("RECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"  • {rec}")
        
    except ValueError as e:
        print(f"\nError: {e}")
        print("\nTo collect more data, run:")
        print("  python data_collector.py")

if __name__ == "__main__":
    main()