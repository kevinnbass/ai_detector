import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

from data_collector import DataCollector
from trainer import GPT4oTrainer
from detector import GPT4oDetector

class ValidationSystem:
    """
    Comprehensive validation system for GPT-4o detection models
    """
    
    def __init__(self, data_file: str = "../data/labeled_dataset.json"):
        self.data_collector = DataCollector(data_file)
        self.trainer = GPT4oTrainer(data_file)
        self.rule_detector = GPT4oDetector()
        
    def cross_validate_model(self, model, X: np.ndarray, y: np.ndarray, 
                           cv_folds: int = 5) -> Dict[str, Any]:
        """
        Perform k-fold cross-validation on a model
        """
        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        cv_results = cross_validate(
            model, X, y, 
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
            scoring=scoring,
            return_train_score=True
        )
        
        results = {
            'cv_folds': cv_folds,
            'metrics': {}
        }
        
        for metric in scoring:
            test_scores = cv_results[f'test_{metric}']
            train_scores = cv_results[f'train_{metric}']
            
            results['metrics'][metric] = {
                'test_mean': test_scores.mean(),
                'test_std': test_scores.std(),
                'train_mean': train_scores.mean(),
                'train_std': train_scores.std(),
                'test_scores': test_scores.tolist(),
                'train_scores': train_scores.tolist()
            }
        
        return results
    
    def validate_rule_based_detector(self, texts: List[str], labels: List[int]) -> Dict[str, Any]:
        """
        Validate the current rule-based detector
        """
        predictions = []
        confidences = []
        pattern_matches = []
        
        for text in texts:
            result = self.rule_detector.detect(text)
            predictions.append(1 if result.is_gpt4o else 0)
            confidences.append(result.confidence)
            pattern_matches.append(len(result.matched_patterns))
        
        y_pred = np.array(predictions)
        y_true = np.array(labels)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, confidences),
            'avg_confidence': np.mean(confidences),
            'avg_patterns_matched': np.mean(pattern_matches)
        }
        
        # Detailed analysis
        results = {
            'metrics': metrics,
            'predictions': y_pred.tolist(),
            'confidences': confidences,
            'pattern_matches': pattern_matches,
            'classification_report': classification_report(y_true, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        return results
    
    def analyze_errors(self, texts: List[str], y_true: List[int], 
                      y_pred: List[int], confidences: List[float]) -> Dict[str, Any]:
        """
        Analyze prediction errors in detail
        """
        errors = {
            'false_positives': [],  # Human labeled as GPT-4o
            'false_negatives': [],  # GPT-4o labeled as Human
            'high_confidence_errors': [],
            'low_confidence_correct': []
        }
        
        for i, (text, true_label, pred_label, conf) in enumerate(
            zip(texts, y_true, y_pred, confidences)
        ):
            
            if true_label != pred_label:
                error_info = {
                    'index': i,
                    'text': text,
                    'true_label': 'gpt4o' if true_label == 1 else 'human',
                    'predicted_label': 'gpt4o' if pred_label == 1 else 'human',
                    'confidence': conf,
                    'text_length': len(text),
                    'word_count': len(text.split())
                }
                
                if true_label == 0 and pred_label == 1:
                    errors['false_positives'].append(error_info)
                elif true_label == 1 and pred_label == 0:
                    errors['false_negatives'].append(error_info)
                
                if conf > 0.8:
                    errors['high_confidence_errors'].append(error_info)
            
            elif conf < 0.6:  # Correct but low confidence
                errors['low_confidence_correct'].append({
                    'index': i,
                    'text': text,
                    'label': 'gpt4o' if true_label == 1 else 'human',
                    'confidence': conf
                })
        
        # Sort by confidence
        errors['false_positives'].sort(key=lambda x: x['confidence'], reverse=True)
        errors['false_negatives'].sort(key=lambda x: x['confidence'], reverse=True)
        
        return errors
    
    def length_based_analysis(self, texts: List[str], y_true: List[int], 
                            y_pred: List[int]) -> Dict[str, Any]:
        """
        Analyze performance by text length
        """
        length_buckets = {
            'very_short': (0, 50),
            'short': (50, 100), 
            'medium': (100, 200),
            'long': (200, float('inf'))
        }
        
        analysis = {}
        
        for bucket_name, (min_len, max_len) in length_buckets.items():
            bucket_indices = [
                i for i, text in enumerate(texts) 
                if min_len <= len(text) < max_len
            ]
            
            if bucket_indices:
                bucket_true = [y_true[i] for i in bucket_indices]
                bucket_pred = [y_pred[i] for i in bucket_indices]
                
                analysis[bucket_name] = {
                    'count': len(bucket_indices),
                    'accuracy': accuracy_score(bucket_true, bucket_pred),
                    'precision': precision_score(bucket_true, bucket_pred, zero_division=0),
                    'recall': recall_score(bucket_true, bucket_pred, zero_division=0),
                    'f1': f1_score(bucket_true, bucket_pred, zero_division=0),
                    'avg_length': np.mean([len(texts[i]) for i in bucket_indices])
                }
        
        return analysis
    
    def confidence_calibration_analysis(self, y_true: List[int], 
                                      confidences: List[float]) -> Dict[str, Any]:
        """
        Analyze how well confidence scores match actual accuracy
        """
        # Bin predictions by confidence
        confidence_bins = [(i/10, (i+1)/10) for i in range(10)]
        calibration = {}
        
        for i, (min_conf, max_conf) in enumerate(confidence_bins):
            bin_indices = [
                j for j, conf in enumerate(confidences)
                if min_conf <= conf < max_conf
            ]
            
            if bin_indices:
                bin_true = [y_true[j] for j in bin_indices]
                bin_predictions = [1 if confidences[j] >= 0.5 else 0 for j in bin_indices]
                
                calibration[f'bin_{i}'] = {
                    'confidence_range': (min_conf, max_conf),
                    'count': len(bin_indices),
                    'accuracy': accuracy_score(bin_true, bin_predictions),
                    'avg_confidence': np.mean([confidences[j] for j in bin_indices]),
                    'expected_accuracy': np.mean([confidences[j] for j in bin_indices])
                }
        
        return calibration
    
    def comprehensive_validation_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive validation report
        """
        stats = self.data_collector.get_statistics()
        
        if stats['total_samples'] < 50:
            raise ValueError("Need at least 50 samples for comprehensive validation")
        
        print("Generating comprehensive validation report...")
        
        # Prepare data
        X, y, texts = self.trainer.prepare_training_data(min_samples=50)
        
        # Rule-based detector validation
        print("Validating rule-based detector...")
        rule_results = self.validate_rule_based_detector(texts, y.tolist())
        
        # Error analysis
        print("Analyzing errors...")
        error_analysis = self.analyze_errors(
            texts, 
            y.tolist(), 
            rule_results['predictions'],
            rule_results['confidences']
        )
        
        # Length-based analysis
        print("Analyzing by text length...")
        length_analysis = self.length_based_analysis(texts, y.tolist(), rule_results['predictions'])
        
        # Confidence calibration
        print("Analyzing confidence calibration...")
        calibration_analysis = self.confidence_calibration_analysis(y.tolist(), rule_results['confidences'])
        
        # Try ML models if enough data
        ml_results = {}
        if stats['total_samples'] >= 100:
            print("Training and validating ML models...")
            try:
                model_results = self.trainer.train_models(X, y)
                
                for name, result in model_results.items():
                    # Cross-validation
                    cv_results = self.cross_validate_model(result['model'], X, y)
                    ml_results[name] = {
                        'holdout_performance': {
                            'accuracy': result['accuracy'],
                            'roc_auc': result['roc_auc']
                        },
                        'cross_validation': cv_results
                    }
            except Exception as e:
                print(f"ML validation failed: {e}")
        
        # Compile comprehensive report
        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'total_samples': stats['total_samples'],
                'gpt4o_samples': stats['gpt4o_samples'], 
                'human_samples': stats['human_samples'],
                'balance_ratio': stats['balance_ratio'],
                'sources': stats.get('sources', [])
            },
            'rule_based_detector': {
                'performance': rule_results['metrics'],
                'classification_report': rule_results['classification_report'],
                'confusion_matrix': rule_results['confusion_matrix']
            },
            'error_analysis': {
                'false_positives_count': len(error_analysis['false_positives']),
                'false_negatives_count': len(error_analysis['false_negatives']),
                'high_confidence_errors_count': len(error_analysis['high_confidence_errors']),
                'top_false_positives': error_analysis['false_positives'][:5],
                'top_false_negatives': error_analysis['false_negatives'][:5]
            },
            'length_analysis': length_analysis,
            'confidence_calibration': calibration_analysis,
            'ml_models': ml_results,
            'recommendations': []
        }
        
        # Generate recommendations
        rule_accuracy = rule_results['metrics']['accuracy']
        
        if rule_accuracy < 0.7:
            report['recommendations'].append("Rule-based detector accuracy is low. Consider collecting more training data.")
        
        if len(error_analysis['false_positives']) > len(error_analysis['false_negatives']) * 2:
            report['recommendations'].append("High false positive rate. Consider increasing detection threshold.")
        
        if stats['balance_ratio'] < 0.5 or stats['balance_ratio'] > 2.0:
            report['recommendations'].append(f"Dataset imbalanced (ratio: {stats['balance_ratio']:.2f}). Collect more minority class samples.")
        
        if stats['total_samples'] < 200:
            report['recommendations'].append("Small dataset. Consider collecting more samples for better performance.")
        
        # Compare ML vs rule-based if available
        if ml_results:
            best_ml_acc = max(r['cross_validation']['metrics']['accuracy']['test_mean'] 
                             for r in ml_results.values())
            if best_ml_acc > rule_accuracy + 0.05:
                report['recommendations'].append(
                    f"ML models show {(best_ml_acc - rule_accuracy)*100:.1f}% improvement. Consider switching to ML approach."
                )
        
        return report
    
    def save_validation_report(self, report: Dict[str, Any], 
                             output_dir: str = "../reports/") -> str:
        """Save validation report to file"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"validation_report_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        return filepath
    
    def print_summary(self, report: Dict[str, Any]):
        """Print a human-readable summary of the validation report"""
        print("\n" + "="*80)
        print("VALIDATION REPORT SUMMARY")
        print("="*80)
        
        # Dataset info
        dataset = report['dataset_info']
        print(f"📊 Dataset: {dataset['total_samples']} samples ({dataset['gpt4o_samples']} GPT-4o, {dataset['human_samples']} Human)")
        print(f"   Balance ratio: {dataset['balance_ratio']:.2f}")
        
        # Rule-based performance
        rule_perf = report['rule_based_detector']['performance']
        print(f"\n🤖 Rule-Based Detector:")
        print(f"   Accuracy:  {rule_perf['accuracy']:.1%}")
        print(f"   Precision: {rule_perf['precision']:.1%}")
        print(f"   Recall:    {rule_perf['recall']:.1%}")
        print(f"   F1-Score:  {rule_perf['f1']:.1%}")
        print(f"   ROC AUC:   {rule_perf['roc_auc']:.1%}")
        
        # Error analysis
        errors = report['error_analysis']
        print(f"\n❌ Error Analysis:")
        print(f"   False Positives: {errors['false_positives_count']}")
        print(f"   False Negatives: {errors['false_negatives_count']}")
        print(f"   High Confidence Errors: {errors['high_confidence_errors_count']}")
        
        # Length analysis
        length = report['length_analysis']
        print(f"\n📏 Performance by Length:")
        for bucket, metrics in length.items():
            print(f"   {bucket.replace('_', ' ').title()}: {metrics['accuracy']:.1%} ({metrics['count']} samples)")
        
        # ML comparison
        if report['ml_models']:
            print(f"\n🧠 ML Models:")
            for name, results in report['ml_models'].items():
                cv_acc = results['cross_validation']['metrics']['accuracy']['test_mean']
                print(f"   {name.replace('_', ' ').title()}: {cv_acc:.1%}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        for rec in report['recommendations']:
            print(f"   • {rec}")
        
        print("="*80)

def main():
    """Main validation interface"""
    print("GPT-4o Detector Validation System")
    print("="*50)
    
    validator = ValidationSystem()
    
    # Check if enough data
    stats = validator.data_collector.get_statistics()
    if stats['total_samples'] < 50:
        print(f"⚠️  Only {stats['total_samples']} samples available.")
        print("Need at least 50 samples for validation.")
        print("Run data collection first: python data_collector.py")
        return
    
    print(f"Dataset ready: {stats['total_samples']} samples")
    
    # Generate comprehensive report
    print("\nGenerating validation report...")
    report = validator.comprehensive_validation_report()
    
    # Save report
    report_path = validator.save_validation_report(report)
    print(f"Report saved to: {report_path}")
    
    # Print summary
    validator.print_summary(report)

if __name__ == "__main__":
    main()