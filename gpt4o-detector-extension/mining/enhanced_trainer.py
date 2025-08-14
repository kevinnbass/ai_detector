import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
import os
from datetime import datetime

from trainer import GPT4oTrainer
from llm_analyzer import LLMAnalyzer
from data_collector import DataCollector

class EnhancedTrainer(GPT4oTrainer):
    """
    Enhanced trainer combining traditional ML with LLM-powered analysis
    """
    
    def __init__(self, data_file: str = "../data/labeled_dataset.json", 
                 openrouter_api_key: str = None):
        super().__init__(data_file)
        self.llm_analyzer = None
        
        if openrouter_api_key:
            self.llm_analyzer = LLMAnalyzer(openrouter_api_key)
            print("✅ LLM analyzer initialized with Gemini 2.5 Flash")
        else:
            print("⚠️ No OpenRouter API key - LLM features disabled")
    
    def extract_llm_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract features using LLM analysis
        """
        if not self.llm_analyzer:
            return np.array([])
        
        print("🧠 Extracting LLM-powered features...")
        
        llm_features = []
        
        for i, text in enumerate(texts):
            print(f"  Analyzing text {i+1}/{len(texts)}...")
            
            try:
                # Get comprehensive analysis
                result = self.llm_analyzer.analyze_text_patterns(text, 'comprehensive')
                
                # Extract numerical features from LLM analysis
                features = self._llm_result_to_features(result)
                llm_features.append(features)
                
                # Rate limiting
                if i < len(texts) - 1:
                    import time
                    time.sleep(0.5)
                
            except Exception as e:
                print(f"    Error analyzing text {i+1}: {e}")
                # Use zero features for failed analysis
                llm_features.append([0.0] * 15)  # 15 LLM features
        
        return np.array(llm_features)
    
    def _llm_result_to_features(self, result: Dict[str, Any]) -> List[float]:
        """
        Convert LLM analysis result to numerical features
        """
        features = []
        
        if 'analysis' in result and isinstance(result['analysis'], dict):
            analysis = result['analysis']
            
            # Primary probability and confidence
            features.append(analysis.get('gpt4o_probability', 0.5))
            features.append(analysis.get('confidence', 0.5))
            
            # Pattern detection features
            patterns = analysis.get('detected_patterns', [])
            pattern_counts = {
                'hedging': 0,
                'contrast': 0, 
                'formal': 0,
                'structured': 0,
                'qualifier': 0,
                'balanced': 0
            }
            
            total_pattern_strength = 0
            for pattern in patterns:
                if isinstance(pattern, dict):
                    pattern_name = pattern.get('pattern', '').lower()
                    strength = pattern.get('strength', 0.5)
                    
                    for key in pattern_counts:
                        if key in pattern_name:
                            pattern_counts[key] += strength
                    
                    total_pattern_strength += strength
            
            # Add pattern features
            for key in pattern_counts:
                features.append(pattern_counts[key])
            
            features.append(total_pattern_strength)  # Total pattern strength
            features.append(len(patterns))  # Number of detected patterns
            
            # Human vs AI indicators
            human_indicators = analysis.get('human_indicators', [])
            ai_indicators = analysis.get('ai_indicators', [])
            
            features.append(len(human_indicators))
            features.append(len(ai_indicators))
            features.append(len(ai_indicators) - len(human_indicators))  # AI vs human balance
            
        else:
            # Default features if analysis failed
            features = [0.5] + [0.0] * 14
        
        # Ensure we have exactly 15 features
        while len(features) < 15:
            features.append(0.0)
        
        return features[:15]
    
    def train_enhanced_models(self, X_traditional: np.ndarray, X_llm: np.ndarray, 
                             y: np.ndarray, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train models combining traditional ML features with LLM features
        """
        
        from sklearn.model_selection import train_test_split
        
        # Combine features
        if X_llm.size > 0:
            X_combined = np.hstack([X_traditional, X_llm])
            feature_names = ['traditional'] * X_traditional.shape[1] + ['llm'] * X_llm.shape[1]
        else:
            X_combined = X_traditional
            feature_names = ['traditional'] * X_traditional.shape[1]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y, test_size=test_size, stratify=y, random_state=42
        )
        
        results = {}
        
        # Traditional ML only
        print("\n🔧 Training traditional ML model...")
        X_trad_train = X_train[:, :X_traditional.shape[1]]
        X_trad_test = X_test[:, :X_traditional.shape[1]]
        
        trad_model = RandomForestClassifier(random_state=42, n_estimators=100)
        trad_model.fit(X_trad_train, y_train)
        trad_pred = trad_model.predict(X_trad_test)
        trad_accuracy = accuracy_score(y_test, trad_pred)
        
        results['traditional_only'] = {
            'model': trad_model,
            'accuracy': trad_accuracy,
            'features_used': X_traditional.shape[1],
            'cv_scores': cross_val_score(trad_model, X_trad_train, y_train, cv=5)
        }
        
        # LLM-enhanced model (if LLM features available)
        if X_llm.size > 0:
            print("🧠 Training LLM-enhanced model...")
            
            enhanced_model = RandomForestClassifier(random_state=42, n_estimators=150)
            enhanced_model.fit(X_train, y_train)
            enhanced_pred = enhanced_model.predict(X_test)
            enhanced_accuracy = accuracy_score(y_test, enhanced_pred)
            
            results['llm_enhanced'] = {
                'model': enhanced_model,
                'accuracy': enhanced_accuracy,
                'features_used': X_combined.shape[1],
                'cv_scores': cross_val_score(enhanced_model, X_train, y_train, cv=5),
                'feature_importance': self._analyze_feature_importance(
                    enhanced_model, feature_names, X_traditional.shape[1]
                )
            }
            
            # LLM-only model
            print("🎯 Training LLM-only model...")
            X_llm_train = X_train[:, X_traditional.shape[1]:]
            X_llm_test = X_test[:, X_traditional.shape[1]:]
            
            llm_model = LogisticRegression(random_state=42)
            llm_model.fit(X_llm_train, y_train)
            llm_pred = llm_model.predict(X_llm_test)
            llm_accuracy = accuracy_score(y_test, llm_pred)
            
            results['llm_only'] = {
                'model': llm_model,
                'accuracy': llm_accuracy,
                'features_used': X_llm.shape[1],
                'cv_scores': cross_val_score(llm_model, X_llm_train, y_train, cv=5)
            }
            
            # Ensemble model
            print("🎭 Training ensemble model...")
            ensemble = VotingClassifier([
                ('traditional', trad_model),
                ('llm_enhanced', enhanced_model),
                ('llm_only', llm_model)
            ], voting='soft')
            
            ensemble.fit(X_train, y_train)
            ensemble_pred = ensemble.predict(X_test)
            ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
            
            results['ensemble'] = {
                'model': ensemble,
                'accuracy': ensemble_accuracy,
                'features_used': X_combined.shape[1],
                'cv_scores': cross_val_score(ensemble, X_train, y_train, cv=5)
            }
        
        return results
    
    def _analyze_feature_importance(self, model, feature_names: List[str], 
                                  n_traditional: int) -> Dict[str, Any]:
        """
        Analyze which features (traditional vs LLM) are most important
        """
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            traditional_importance = np.sum(importances[:n_traditional])
            llm_importance = np.sum(importances[n_traditional:]) if len(importances) > n_traditional else 0
            
            # Top features overall
            top_indices = np.argsort(importances)[-10:][::-1]
            top_features = [(i, importances[i], 'traditional' if i < n_traditional else 'llm') 
                           for i in top_indices]
            
            return {
                'traditional_total_importance': traditional_importance,
                'llm_total_importance': llm_importance,
                'llm_advantage': llm_importance - traditional_importance,
                'top_features': top_features,
                'importance_ratio': llm_importance / traditional_importance if traditional_importance > 0 else float('inf')
            }
        
        return {}
    
    def comprehensive_evaluation(self, min_samples: int = 30) -> Dict[str, Any]:
        """
        Run comprehensive evaluation combining traditional and LLM approaches
        """
        
        stats = self.data_collector.get_statistics()
        
        if stats['total_samples'] < min_samples:
            raise ValueError(f"Need at least {min_samples} samples, got {stats['total_samples']}")
        
        print(f"🚀 Starting comprehensive evaluation with {stats['total_samples']} samples")
        
        # Extract traditional features
        print("🔧 Extracting traditional ML features...")
        X_traditional, y, texts = self.prepare_training_data(min_samples)
        
        # Extract LLM features (if available)
        X_llm = np.array([])
        if self.llm_analyzer:
            try:
                X_llm = self.extract_llm_features(texts)
                print(f"✅ Extracted {X_llm.shape[1]} LLM features")
            except Exception as e:
                print(f"❌ LLM feature extraction failed: {e}")
                X_llm = np.array([])
        
        # Train all models
        print("\n🎯 Training enhanced models...")
        model_results = self.train_enhanced_models(X_traditional, X_llm, y)
        
        # Evaluate current rule-based detector
        print("\n📊 Evaluating baseline detector...")
        baseline_results = self.evaluate_current_detector(X_traditional, y, texts)
        
        # Generate comprehensive report
        report = self._generate_enhanced_report(
            model_results, baseline_results, stats, 
            X_traditional.shape[1], X_llm.shape[1] if X_llm.size > 0 else 0
        )
        
        return report
    
    def _generate_enhanced_report(self, model_results: Dict[str, Any], 
                                 baseline_results: Dict[str, Any], 
                                 dataset_stats: Dict[str, Any],
                                 n_traditional_features: int,
                                 n_llm_features: int) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report
        """
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_stats': dataset_stats,
            'feature_summary': {
                'traditional_features': n_traditional_features,
                'llm_features': n_llm_features,
                'total_features': n_traditional_features + n_llm_features
            },
            'baseline_performance': {
                'rule_based_accuracy': baseline_results['accuracy'],
                'rule_based_roc_auc': baseline_results['roc_auc']
            },
            'model_comparison': {},
            'performance_gains': {},
            'recommendations': []
        }
        
        baseline_accuracy = baseline_results['accuracy']
        
        for model_name, results in model_results.items():
            accuracy = results['accuracy']
            cv_mean = results['cv_scores'].mean()
            cv_std = results['cv_scores'].std()
            
            report['model_comparison'][model_name] = {
                'accuracy': accuracy,
                'cv_accuracy': cv_mean,
                'cv_std': cv_std,
                'features_used': results['features_used'],
                'improvement_over_baseline': accuracy - baseline_accuracy
            }
            
            if 'feature_importance' in results:
                report['model_comparison'][model_name]['feature_analysis'] = results['feature_importance']
        
        # Calculate performance gains
        if 'llm_enhanced' in model_results and 'traditional_only' in model_results:
            llm_gain = model_results['llm_enhanced']['accuracy'] - model_results['traditional_only']['accuracy']
            report['performance_gains']['llm_enhancement'] = llm_gain
            
            if llm_gain > 0.02:  # 2% improvement
                report['recommendations'].append(
                    f"LLM features provide {llm_gain*100:.1f}% accuracy improvement - recommended for production"
                )
        
        if 'ensemble' in model_results:
            best_single = max(model_results[k]['accuracy'] for k in model_results if k != 'ensemble')
            ensemble_gain = model_results['ensemble']['accuracy'] - best_single
            report['performance_gains']['ensemble_improvement'] = ensemble_gain
            
            if ensemble_gain > 0.01:
                report['recommendations'].append(
                    f"Ensemble provides additional {ensemble_gain*100:.1f}% improvement"
                )
        
        # Best model recommendation
        best_model = max(model_results.keys(), key=lambda k: model_results[k]['accuracy'])
        best_accuracy = model_results[best_model]['accuracy']
        
        report['best_model'] = {
            'name': best_model,
            'accuracy': best_accuracy,
            'improvement_over_baseline': best_accuracy - baseline_accuracy
        }
        
        # Feature importance insights
        if n_llm_features > 0 and 'llm_enhanced' in model_results:
            importance_data = model_results['llm_enhanced'].get('feature_importance', {})
            if importance_data:
                llm_ratio = importance_data.get('importance_ratio', 1.0)
                if llm_ratio > 1.5:
                    report['recommendations'].append(
                        f"LLM features are {llm_ratio:.1f}x more important than traditional features"
                    )
        
        # Dataset recommendations
        if dataset_stats['total_samples'] < 100:
            report['recommendations'].append(
                "Consider collecting more data for better LLM feature extraction"
            )
        
        return report
    
    def save_enhanced_models(self, model_results: Dict[str, Any], 
                           output_dir: str = "../models/enhanced/"):
        """
        Save all enhanced models
        """
        import pickle
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name, results in model_results.items():
            model_path = os.path.join(output_dir, f"{model_name}_model.pkl")
            
            with open(model_path, 'wb') as f:
                pickle.dump(results['model'], f)
            
            print(f"💾 Saved {model_name} model to {model_path}")
        
        # Save metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'models': list(model_results.keys()),
            'best_model': max(model_results.keys(), key=lambda k: model_results[k]['accuracy']),
            'accuracies': {k: v['accuracy'] for k, v in model_results.items()}
        }
        
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

def main():
    """
    Main enhanced training pipeline
    """
    
    print("🚀 Enhanced GPT-4o Detector Training (Traditional + LLM)")
    print("=" * 70)
    
    # Check for API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if api_key:
        print("✅ OpenRouter API key found - LLM analysis enabled")
    else:
        print("⚠️ No OpenRouter API key - using traditional ML only")
        print("  Set OPENROUTER_API_KEY to enable LLM features")
    
    # Initialize enhanced trainer
    trainer = EnhancedTrainer(openrouter_api_key=api_key)
    
    # Check dataset
    stats = trainer.data_collector.get_statistics()
    print(f"\n📊 Dataset: {stats['total_samples']} samples ({stats['gpt4o_samples']} GPT-4o, {stats['human_samples']} human)")
    
    if stats['total_samples'] < 30:
        print("❌ Need at least 30 samples for enhanced training")
        print("   Run data collection first: python mining/data_collector.py")
        return
    
    try:
        # Run comprehensive evaluation
        report = trainer.comprehensive_evaluation()
        
        # Save results
        os.makedirs("../reports/enhanced", exist_ok=True)
        report_path = f"../reports/enhanced/enhanced_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Report saved: {report_path}")
        
        # Print summary
        print("\n" + "=" * 70)
        print("🎯 ENHANCED TRAINING RESULTS")
        print("=" * 70)
        
        print(f"📊 Dataset: {report['dataset_stats']['total_samples']} samples")
        print(f"🔧 Features: {report['feature_summary']['traditional_features']} traditional + {report['feature_summary']['llm_features']} LLM")
        
        print(f"\n📈 Performance Comparison:")
        baseline_acc = report['baseline_performance']['rule_based_accuracy']
        print(f"   Baseline (rules): {baseline_acc:.1%}")
        
        for model_name, results in report['model_comparison'].items():
            improvement = results['improvement_over_baseline']
            print(f"   {model_name.replace('_', ' ').title()}: {results['accuracy']:.1%} (+{improvement*100:.1f}%)")
        
        print(f"\n🏆 Best Model: {report['best_model']['name']} ({report['best_model']['accuracy']:.1%})")
        
        print(f"\n💡 Recommendations:")
        for rec in report['recommendations']:
            print(f"   • {rec}")
        
        print(f"\n✅ Enhanced training complete!")
        
    except Exception as e:
        print(f"\n❌ Error during enhanced training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()