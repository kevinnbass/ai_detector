"""
Performance optimizer for the detection engine.

Implements various optimization techniques to achieve sub-100ms detection times
including caching, pre-computation, parallel processing, and method selection.
"""

import asyncio
import time
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum

from src.core.monitoring import get_logger, get_metrics_collector


class OptimizationLevel(Enum):
    """Performance optimization levels."""
    MINIMAL = "minimal"      # Basic optimizations only
    STANDARD = "standard"    # Balanced performance/accuracy
    AGGRESSIVE = "aggressive"  # Maximum performance


@dataclass
class DetectionProfile:
    """Performance profile for different detection scenarios."""
    name: str
    max_time_ms: float
    preferred_methods: List[str]
    cache_ttl: int
    parallel_threshold: int
    accuracy_threshold: float


class PerformanceOptimizer:
    """Optimizes detection performance across different scenarios."""
    
    def __init__(self, level: OptimizationLevel = OptimizationLevel.STANDARD):
        self.level = level
        self.logger = get_logger(__name__)
        self.metrics = get_metrics_collector()
        
        # Performance profiles
        self.profiles = {
            "realtime": DetectionProfile(
                name="realtime",
                max_time_ms=50.0,
                preferred_methods=["pattern", "heuristic"],
                cache_ttl=300,  # 5 minutes
                parallel_threshold=1,
                accuracy_threshold=0.6
            ),
            "interactive": DetectionProfile(
                name="interactive", 
                max_time_ms=100.0,
                preferred_methods=["pattern", "ml_fast"],
                cache_ttl=600,  # 10 minutes
                parallel_threshold=3,
                accuracy_threshold=0.7
            ),
            "batch": DetectionProfile(
                name="batch",
                max_time_ms=500.0,
                preferred_methods=["ml", "ensemble_light"],
                cache_ttl=1800,  # 30 minutes
                parallel_threshold=10,
                accuracy_threshold=0.8
            ),
            "accuracy": DetectionProfile(
                name="accuracy",
                max_time_ms=2000.0,
                preferred_methods=["llm", "ensemble_full"],
                cache_ttl=3600,  # 1 hour
                parallel_threshold=5,
                accuracy_threshold=0.9
            )
        }
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Method timings cache
        self.method_timings = {}
        
        # Pre-computed features cache
        self.feature_cache = {}
    
    def select_optimal_method(self, text: str, target_time_ms: float = 100.0) -> str:
        """Select the optimal detection method based on performance constraints."""
        text_length = len(text)
        
        # Use historical timing data to predict performance
        if target_time_ms <= 50:
            if text_length < 100:
                return "heuristic"
            elif text_length < 500:
                return "pattern"
            else:
                return "pattern_fast"
        
        elif target_time_ms <= 100:
            if text_length < 200:
                return "pattern"
            elif text_length < 1000:
                return "ml_fast" 
            else:
                return "pattern"
        
        elif target_time_ms <= 500:
            if text_length < 1000:
                return "ml"
            else:
                return "ensemble_light"
        
        else:
            return "ensemble_full"
    
    def optimize_for_profile(self, profile_name: str) -> Dict[str, Any]:
        """Get optimization settings for a specific performance profile."""
        if profile_name not in self.profiles:
            profile_name = "interactive"  # Default
        
        profile = self.profiles[profile_name]
        
        return {
            "profile": profile,
            "method": profile.preferred_methods[0],
            "cache_enabled": True,
            "cache_ttl": profile.cache_ttl,
            "parallel_enabled": True,
            "timeout_ms": profile.max_time_ms,
            "accuracy_threshold": profile.accuracy_threshold
        }
    
    async def fast_precheck(self, text: str) -> Optional[Dict[str, Any]]:
        """Perform fast precheck to potentially avoid full detection."""
        # Very basic heuristics for instant results
        text_lower = text.lower()
        
        # Extremely short text
        if len(text) < 10:
            return {
                "is_ai_generated": False,
                "confidence_score": 0.9,
                "method_used": "length_heuristic",
                "processing_time_ms": 1
            }
        
        # Contains obvious human markers
        human_markers = ["lol", "haha", "omg", "wtf", "😂", "😊", "💀", "fr"]
        if any(marker in text_lower for marker in human_markers):
            return {
                "is_ai_generated": False,
                "confidence_score": 0.85,
                "method_used": "marker_heuristic", 
                "processing_time_ms": 2
            }
        
        # Contains obvious AI markers
        ai_markers = [
            "comprehensive analysis", "multifaceted nature", "paradigm",
            "furthermore", "it should be noted", "careful consideration"
        ]
        if any(marker in text_lower for marker in ai_markers):
            return {
                "is_ai_generated": True,
                "confidence_score": 0.8,
                "method_used": "marker_heuristic",
                "processing_time_ms": 2
            }
        
        return None  # No quick determination possible
    
    def generate_text_fingerprint(self, text: str) -> str:
        """Generate a fast fingerprint for text caching."""
        # Use a fast hash for caching
        return hashlib.md5(text.encode('utf-8')).hexdigest()[:16]
    
    async def parallel_feature_extraction(self, text: str) -> Dict[str, Any]:
        """Extract features in parallel for faster processing."""
        def extract_basic_features(text: str) -> Dict[str, float]:
            return {
                "char_count": len(text),
                "word_count": len(text.split()),
                "sentence_count": text.count('.') + text.count('!') + text.count('?'),
                "avg_word_length": sum(len(word) for word in text.split()) / max(len(text.split()), 1)
            }
        
        def extract_linguistic_features(text: str) -> Dict[str, float]:
            text_lower = text.lower()
            return {
                "formality_score": self._calculate_formality_score(text),
                "complexity_score": self._calculate_complexity_score(text),
                "punctuation_ratio": sum(1 for c in text if c in '.,!?;:') / max(len(text), 1),
                "uppercase_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1)
            }
        
        def extract_pattern_features(text: str) -> Dict[str, float]:
            return {
                "transition_markers": self._count_transition_markers(text),
                "hedging_words": self._count_hedging_words(text),
                "passive_voice": self._count_passive_voice(text),
                "modal_verbs": self._count_modal_verbs(text)
            }
        
        # Run feature extraction in parallel
        loop = asyncio.get_event_loop()
        
        basic_task = loop.run_in_executor(self.thread_pool, extract_basic_features, text)
        linguistic_task = loop.run_in_executor(self.thread_pool, extract_linguistic_features, text)
        pattern_task = loop.run_in_executor(self.thread_pool, extract_pattern_features, text)
        
        basic_features, linguistic_features, pattern_features = await asyncio.gather(
            basic_task, linguistic_task, pattern_task
        )
        
        # Combine all features
        features = {**basic_features, **linguistic_features, **pattern_features}
        return features
    
    def _calculate_formality_score(self, text: str) -> float:
        """Calculate formality score quickly."""
        formal_words = [
            "furthermore", "however", "therefore", "consequently", "nevertheless",
            "analysis", "comprehensive", "demonstrates", "indicates", "suggests"
        ]
        
        words = text.lower().split()
        formal_count = sum(1 for word in words if word in formal_words)
        return min(formal_count / max(len(words), 1) * 10, 1.0)
    
    def _calculate_complexity_score(self, text: str) -> float:
        """Calculate complexity score quickly."""
        words = text.split()
        if not words:
            return 0.0
        
        # Average word length as complexity proxy
        avg_word_length = sum(len(word) for word in words) / len(words)
        return min(avg_word_length / 10, 1.0)
    
    def _count_transition_markers(self, text: str) -> float:
        """Count transition markers quickly."""
        markers = [
            "furthermore", "however", "therefore", "moreover", "additionally",
            "consequently", "nevertheless", "meanwhile", "subsequently"
        ]
        
        text_lower = text.lower()
        count = sum(text_lower.count(marker) for marker in markers)
        return min(count / max(len(text.split()), 1) * 100, 1.0)
    
    def _count_hedging_words(self, text: str) -> float:
        """Count hedging words quickly."""
        hedges = [
            "might", "could", "may", "perhaps", "possibly", "likely",
            "seems", "appears", "suggests", "indicates"
        ]
        
        words = text.lower().split()
        hedge_count = sum(1 for word in words if word in hedges)
        return min(hedge_count / max(len(words), 1) * 10, 1.0)
    
    def _count_passive_voice(self, text: str) -> float:
        """Count passive voice indicators quickly."""
        passive_indicators = [
            "was", "were", "been", "being", "is", "are", "am"
        ]
        
        words = text.lower().split()
        passive_count = sum(1 for word in words if word in passive_indicators)
        return min(passive_count / max(len(words), 1) * 5, 1.0)
    
    def _count_modal_verbs(self, text: str) -> float:
        """Count modal verbs quickly."""
        modals = [
            "should", "would", "could", "might", "must", "shall", "will",
            "can", "may", "ought"
        ]
        
        words = text.lower().split()
        modal_count = sum(1 for word in words if word in modals)
        return min(modal_count / max(len(words), 1) * 10, 1.0)
    
    async def cached_detection(self, text: str, method: str, cache_ttl: int = 300) -> Optional[Dict[str, Any]]:
        """Check cache for previous detection results."""
        fingerprint = self.generate_text_fingerprint(text)
        cache_key = f"detection:{method}:{fingerprint}"
        
        # Check if result is cached
        cached_result = self.feature_cache.get(cache_key)
        if cached_result:
            cache_age = time.time() - cached_result.get("cached_at", 0)
            if cache_age < cache_ttl:
                cached_result["from_cache"] = True
                return cached_result
        
        return None
    
    def cache_detection_result(self, text: str, method: str, result: Dict[str, Any], cache_ttl: int = 300):
        """Cache detection result for future use."""
        fingerprint = self.generate_text_fingerprint(text)
        cache_key = f"detection:{method}:{fingerprint}"
        
        result_to_cache = result.copy()
        result_to_cache["cached_at"] = time.time()
        
        self.feature_cache[cache_key] = result_to_cache
        
        # Clean old cache entries if cache is getting large
        if len(self.feature_cache) > 1000:
            self._cleanup_cache()
    
    def _cleanup_cache(self):
        """Clean up old cache entries."""
        current_time = time.time()
        keys_to_remove = []
        
        for key, value in self.feature_cache.items():
            cache_age = current_time - value.get("cached_at", 0)
            if cache_age > 3600:  # Remove entries older than 1 hour
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.feature_cache[key]
    
    async def batch_optimize(self, texts: List[str], profile_name: str = "batch") -> List[Dict[str, Any]]:
        """Optimize batch processing of multiple texts."""
        profile = self.profiles[profile_name]
        
        # Group texts by length for optimal processing
        short_texts = [text for text in texts if len(text) < 100]
        medium_texts = [text for text in texts if 100 <= len(text) < 500]
        long_texts = [text for text in texts if len(text) >= 500]
        
        results = []
        
        # Process short texts with fast methods
        if short_texts:
            short_results = await self._process_text_batch(
                short_texts, "pattern", max_parallel=8
            )
            results.extend(short_results)
        
        # Process medium texts with balanced methods
        if medium_texts:
            medium_results = await self._process_text_batch(
                medium_texts, "ml_fast", max_parallel=4
            )
            results.extend(medium_results)
        
        # Process long texts with accuracy-focused methods
        if long_texts:
            long_results = await self._process_text_batch(
                long_texts, profile.preferred_methods[0], max_parallel=2
            )
            results.extend(long_results)
        
        return results
    
    async def _process_text_batch(self, texts: List[str], method: str, max_parallel: int = 4) -> List[Dict[str, Any]]:
        """Process a batch of texts with controlled parallelism."""
        semaphore = asyncio.Semaphore(max_parallel)
        
        async def process_single_text(text: str) -> Dict[str, Any]:
            async with semaphore:
                # Mock detection for now - would integrate with actual detection engine
                start_time = time.time()
                
                # Simulate processing time based on text length and method
                if method == "pattern":
                    processing_time = len(text) * 0.1  # 0.1ms per character
                elif method == "ml_fast":
                    processing_time = len(text) * 0.2  # 0.2ms per character
                else:
                    processing_time = len(text) * 0.5  # 0.5ms per character
                
                await asyncio.sleep(processing_time / 1000)  # Convert to seconds
                
                end_time = time.time()
                
                return {
                    "text": text[:50] + "..." if len(text) > 50 else text,
                    "is_ai_generated": len(text) > 200,  # Simple heuristic for demo
                    "confidence_score": 0.8,
                    "method_used": method,
                    "processing_time_ms": (end_time - start_time) * 1000
                }
        
        # Process all texts concurrently
        tasks = [process_single_text(text) for text in texts]
        results = await asyncio.gather(*tasks)
        
        return results
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance optimization report."""
        return {
            "optimization_level": self.level.value,
            "active_profiles": list(self.profiles.keys()),
            "cache_size": len(self.feature_cache),
            "thread_pool_size": self.thread_pool._max_workers,
            "method_timings": self.method_timings,
            "optimization_stats": {
                "cache_hits": self.metrics.get_metric("cache_hits_total").get_value() if self.metrics.get_metric("cache_hits_total") else 0,
                "fast_prechecks": self.metrics.get_metric("fast_precheck_total").get_value() if self.metrics.get_metric("fast_precheck_total") else 0,
                "parallel_extractions": self.metrics.get_metric("parallel_extraction_total").get_value() if self.metrics.get_metric("parallel_extraction_total") else 0
            }
        }