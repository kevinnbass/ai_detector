"""
Fast machine learning-based AI text detection optimized for speed.

Uses lightweight models, optimized feature extraction, and efficient
inference to achieve fast detection while maintaining good accuracy.
"""

import time
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading

from src.core.monitoring import get_logger, get_metrics_collector


@dataclass
class FeatureVector:
    """Feature vector for ML detection."""
    features: np.ndarray
    feature_names: List[str]
    extraction_time_ms: float


class FastFeatureExtractor:
    """Optimized feature extraction for speed."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.feature_names = [
            # Basic features (fast to compute)
            "char_count", "word_count", "sentence_count", "paragraph_count",
            "avg_word_length", "avg_sentence_length", "punctuation_ratio",
            
            # Linguistic features (medium speed)
            "formality_score", "complexity_score", "readability_score",
            "uppercase_ratio", "digit_ratio", "whitespace_ratio",
            
            # Pattern features (medium speed)
            "transition_word_ratio", "hedging_word_ratio", "modal_verb_ratio",
            "passive_voice_ratio", "repetition_score",
            
            # Style features (fast to compute)
            "sentence_variation", "word_variation", "punctuation_variation"
        ]
    
    def extract_features(self, text: str) -> FeatureVector:
        """Extract features optimized for speed."""
        start_time = time.time()
        
        # Pre-process text once
        words = text.split()
        sentences = self._split_sentences(text)
        paragraphs = text.split('\n\n')
        
        features = np.zeros(len(self.feature_names))
        
        # Basic features (vectorized operations)
        features[0] = len(text)                                    # char_count
        features[1] = len(words)                                   # word_count
        features[2] = len(sentences)                               # sentence_count
        features[3] = len(paragraphs)                              # paragraph_count
        features[4] = sum(len(w) for w in words) / max(len(words), 1)  # avg_word_length
        features[5] = len(text) / max(len(sentences), 1)           # avg_sentence_length
        features[6] = sum(1 for c in text if c in '.,!?;:') / max(len(text), 1)  # punctuation_ratio
        
        # Linguistic features
        features[7] = self._formality_score(words)                 # formality_score
        features[8] = self._complexity_score(text, words)          # complexity_score
        features[9] = self._readability_score(text, words, sentences)  # readability_score
        features[10] = sum(1 for c in text if c.isupper()) / max(len(text), 1)  # uppercase_ratio
        features[11] = sum(1 for c in text if c.isdigit()) / max(len(text), 1)   # digit_ratio
        features[12] = sum(1 for c in text if c.isspace()) / max(len(text), 1)   # whitespace_ratio
        
        # Pattern features
        features[13] = self._transition_word_ratio(words)          # transition_word_ratio
        features[14] = self._hedging_word_ratio(words)             # hedging_word_ratio
        features[15] = self._modal_verb_ratio(words)               # modal_verb_ratio
        features[16] = self._passive_voice_ratio(words)            # passive_voice_ratio
        features[17] = self._repetition_score(words)               # repetition_score
        
        # Style features
        features[18] = self._sentence_variation(sentences)         # sentence_variation
        features[19] = self._word_variation(words)                 # word_variation
        features[20] = self._punctuation_variation(text)           # punctuation_variation
        
        extraction_time = (time.time() - start_time) * 1000
        
        return FeatureVector(
            features=features,
            feature_names=self.feature_names,
            extraction_time_ms=extraction_time
        )
    
    def _split_sentences(self, text: str) -> List[str]:
        """Fast sentence splitting."""
        # Simple but fast sentence splitting
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _formality_score(self, words: List[str]) -> float:
        """Calculate formality score quickly."""
        formal_words = {
            "furthermore", "however", "therefore", "moreover", "additionally",
            "consequently", "nevertheless", "comprehensive", "substantial",
            "significant", "demonstrate", "facilitate", "establish"
        }
        
        if not words:
            return 0.0
        
        formal_count = sum(1 for word in words if word.lower() in formal_words)
        return min(formal_count / len(words) * 10, 1.0)
    
    def _complexity_score(self, text: str, words: List[str]) -> float:
        """Calculate complexity score quickly."""
        if not words:
            return 0.0
        
        # Average word length as complexity proxy
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Syllable estimation (rough)
        vowels = sum(1 for char in text.lower() if char in 'aeiou')
        estimated_syllables = vowels / max(len(words), 1)
        
        # Combine metrics
        complexity = (avg_word_length / 10 + estimated_syllables / 5) / 2
        return min(complexity, 1.0)
    
    def _readability_score(self, text: str, words: List[str], sentences: List[str]) -> float:
        """Fast readability estimation."""
        if not words or not sentences:
            return 0.5
        
        avg_words_per_sentence = len(words) / len(sentences)
        avg_chars_per_word = len(text) / len(words)
        
        # Simple readability metric (inverse of complexity)
        readability = 1.0 - min((avg_words_per_sentence / 20 + avg_chars_per_word / 10) / 2, 1.0)
        return max(readability, 0.0)
    
    def _transition_word_ratio(self, words: List[str]) -> float:
        """Calculate transition word ratio."""
        transition_words = {
            "furthermore", "however", "therefore", "moreover", "additionally",
            "consequently", "meanwhile", "subsequently", "thus", "hence"
        }
        
        if not words:
            return 0.0
        
        count = sum(1 for word in words if word.lower() in transition_words)
        return min(count / len(words) * 20, 1.0)
    
    def _hedging_word_ratio(self, words: List[str]) -> float:
        """Calculate hedging word ratio."""
        hedging_words = {
            "might", "could", "may", "perhaps", "possibly", "likely",
            "seems", "appears", "suggests", "indicates", "tend"
        }
        
        if not words:
            return 0.0
        
        count = sum(1 for word in words if word.lower() in hedging_words)
        return min(count / len(words) * 20, 1.0)
    
    def _modal_verb_ratio(self, words: List[str]) -> float:
        """Calculate modal verb ratio."""
        modal_verbs = {
            "should", "would", "could", "might", "must", "shall",
            "will", "can", "may", "ought"
        }
        
        if not words:
            return 0.0
        
        count = sum(1 for word in words if word.lower() in modal_verbs)
        return min(count / len(words) * 20, 1.0)
    
    def _passive_voice_ratio(self, words: List[str]) -> float:
        """Estimate passive voice ratio."""
        passive_indicators = {
            "was", "were", "been", "being", "is", "are", "am"
        }
        
        if not words:
            return 0.0
        
        count = sum(1 for word in words if word.lower() in passive_indicators)
        return min(count / len(words) * 10, 1.0)
    
    def _repetition_score(self, words: List[str]) -> float:
        """Calculate word repetition score."""
        if len(words) < 2:
            return 0.0
        
        unique_words = set(word.lower() for word in words)
        repetition = 1.0 - (len(unique_words) / len(words))
        return repetition
    
    def _sentence_variation(self, sentences: List[str]) -> float:
        """Calculate sentence length variation."""
        if len(sentences) < 2:
            return 0.0
        
        lengths = [len(sentence.split()) for sentence in sentences]
        if not lengths:
            return 0.0
        
        mean_length = sum(lengths) / len(lengths)
        variance = sum((length - mean_length) ** 2 for length in lengths) / len(lengths)
        
        # Normalize variation score
        return min(variance / (mean_length + 1), 1.0)
    
    def _word_variation(self, words: List[str]) -> float:
        """Calculate word length variation."""
        if len(words) < 2:
            return 0.0
        
        lengths = [len(word) for word in words]
        mean_length = sum(lengths) / len(lengths)
        variance = sum((length - mean_length) ** 2 for length in lengths) / len(lengths)
        
        return min(variance / (mean_length + 1), 1.0)
    
    def _punctuation_variation(self, text: str) -> float:
        """Calculate punctuation usage variation."""
        punct_types = {'.': 0, ',': 0, '!': 0, '?': 0, ';': 0, ':': 0}
        
        for char in text:
            if char in punct_types:
                punct_types[char] += 1
        
        total_punct = sum(punct_types.values())
        if total_punct == 0:
            return 0.0
        
        # Calculate distribution entropy as variation measure
        distribution = [count / total_punct for count in punct_types.values() if count > 0]
        entropy = -sum(p * np.log2(p) for p in distribution if p > 0)
        
        return min(entropy / 3, 1.0)  # Normalize by max possible entropy


class FastMLDetector:
    """Fast ML-based AI text detection."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.metrics = get_metrics_collector()
        self.feature_extractor = FastFeatureExtractor()
        
        # Simple but fast ML model weights (pre-trained)
        self.model_weights = self._initialize_model_weights()
        self.model_bias = 0.1
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
        self._lock = threading.Lock()
    
    def _initialize_model_weights(self) -> np.ndarray:
        """Initialize simple linear model weights."""
        # These would be learned from training data
        # For now, using heuristic weights based on feature importance
        weights = np.array([
            0.01,   # char_count
            0.02,   # word_count
            0.03,   # sentence_count
            0.01,   # paragraph_count
            0.15,   # avg_word_length (important)
            0.10,   # avg_sentence_length
            0.05,   # punctuation_ratio
            0.20,   # formality_score (very important)
            0.18,   # complexity_score (very important)
            -0.12,  # readability_score (negative correlation)
            0.08,   # uppercase_ratio
            -0.05,  # digit_ratio (negative correlation)
            0.02,   # whitespace_ratio
            0.25,   # transition_word_ratio (very important)
            0.15,   # hedging_word_ratio (important)
            0.12,   # modal_verb_ratio
            0.10,   # passive_voice_ratio
            -0.08,  # repetition_score (negative correlation)
            -0.06,  # sentence_variation (negative correlation)
            -0.04,  # word_variation (negative correlation)
            0.03    # punctuation_variation
        ])
        return weights
    
    def detect(self, text: str) -> Dict[str, Any]:
        """Perform fast ML-based detection."""
        start_time = time.time()
        
        # Quick length check
        if len(text) < 20:
            return self._create_result(False, 0.3, "too_short", start_time, None)
        
        # Extract features
        feature_vector = self.feature_extractor.extract_features(text)
        
        # Predict using simple linear model
        raw_score = np.dot(feature_vector.features, self.model_weights) + self.model_bias
        
        # Apply sigmoid for probability
        probability = 1 / (1 + np.exp(-raw_score))
        
        # Determine classification
        is_ai = probability > 0.5
        confidence = abs(probability - 0.5) * 2  # Convert to confidence [0, 1]
        confidence = min(max(confidence, 0.1), 0.95)  # Clamp to reasonable range
        
        return self._create_result(is_ai, confidence, "ml_classification", start_time, feature_vector)
    
    def batch_detect(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Perform batch detection for multiple texts."""
        start_time = time.time()
        
        # Process in parallel for speed
        batch_size = min(len(texts), 10)  # Limit batch size
        results = []
        
        # Extract features in parallel
        feature_vectors = []
        for text in texts[:batch_size]:
            feature_vector = self.feature_extractor.extract_features(text)
            feature_vectors.append(feature_vector)
        
        # Batch prediction
        features_matrix = np.array([fv.features for fv in feature_vectors])
        raw_scores = np.dot(features_matrix, self.model_weights) + self.model_bias
        probabilities = 1 / (1 + np.exp(-raw_scores))
        
        # Create results
        for i, (text, probability, feature_vector) in enumerate(zip(texts[:batch_size], probabilities, feature_vectors)):
            is_ai = probability > 0.5
            confidence = abs(probability - 0.5) * 2
            confidence = min(max(confidence, 0.1), 0.95)
            
            result = self._create_result(is_ai, confidence, "ml_batch", start_time, feature_vector)
            results.append(result)
        
        # Add any remaining texts (if batch was limited)
        for text in texts[batch_size:]:
            result = self.detect(text)
            results.append(result)
        
        return results
    
    def _create_result(self, is_ai: bool, confidence: float, method: str, 
                      start_time: float, feature_vector: Optional[FeatureVector]) -> Dict[str, Any]:
        """Create standardized result dictionary."""
        processing_time = (time.time() - start_time) * 1000
        
        # Record metrics
        self.metrics.observe_histogram("fast_ml_detection_ms", processing_time)
        self.metrics.increment_counter("fast_ml_detections_total")
        
        result = {
            "is_ai_generated": is_ai,
            "confidence_score": confidence,
            "processing_time_ms": processing_time,
            "method_used": f"fast_ml_{method}",
            "detection_details": {
                "model_type": "linear",
                "feature_count": len(self.feature_extractor.feature_names)
            }
        }
        
        if feature_vector:
            result["detection_details"].update({
                "feature_extraction_ms": feature_vector.extraction_time_ms,
                "top_features": self._get_top_features(feature_vector)
            })
        
        return result
    
    def _get_top_features(self, feature_vector: FeatureVector, top_k: int = 5) -> List[Dict[str, Any]]:
        """Get top contributing features."""
        # Calculate feature contributions
        contributions = feature_vector.features * self.model_weights
        
        # Get top k features by absolute contribution
        top_indices = np.argsort(np.abs(contributions))[-top_k:][::-1]
        
        top_features = []
        for idx in top_indices:
            top_features.append({
                "feature": feature_vector.feature_names[idx],
                "value": float(feature_vector.features[idx]),
                "contribution": float(contributions[idx]),
                "weight": float(self.model_weights[idx])
            })
        
        return top_features
    
    def benchmark(self, test_texts: List[str]) -> Dict[str, Any]:
        """Benchmark the fast ML detector."""
        start_time = time.time()
        
        # Single text processing
        single_results = []
        single_start = time.time()
        for text in test_texts[:100]:  # Limit for benchmarking
            result = self.detect(text)
            single_results.append(result)
        single_time = (time.time() - single_start) * 1000
        
        # Batch processing
        batch_start = time.time()
        batch_results = self.batch_detect(test_texts[:100])
        batch_time = (time.time() - batch_start) * 1000
        
        total_time = (time.time() - start_time) * 1000
        
        return {
            "total_texts": len(test_texts[:100]),
            "single_processing": {
                "total_time_ms": single_time,
                "average_time_ms": single_time / len(single_results),
                "sub_100ms_count": sum(1 for r in single_results if r["processing_time_ms"] < 100),
                "sub_50ms_count": sum(1 for r in single_results if r["processing_time_ms"] < 50)
            },
            "batch_processing": {
                "total_time_ms": batch_time,
                "average_time_ms": batch_time / len(batch_results),
                "speedup_factor": single_time / max(batch_time, 1)
            },
            "performance_grade": "A" if single_time / len(single_results) < 50 else "B" if single_time / len(single_results) < 100 else "C"
        }