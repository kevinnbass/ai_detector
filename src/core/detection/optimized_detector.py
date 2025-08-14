"""
Optimized detection engine that achieves sub-100ms detection times.

Integrates fast pattern detection, ML detection, and heuristics with
intelligent method selection and performance optimization.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from dataclasses import dataclass

from src.core.detection.fast_pattern_detector import FastPatternDetector, HeuristicDetector
from src.core.detection.fast_ml_detector import FastMLDetector
from src.core.performance.detector_optimizer import PerformanceOptimizer, OptimizationLevel
from src.core.monitoring import get_logger, get_metrics_collector
from src.core.cache import get_cache_manager


class PerformanceMode(Enum):
    """Performance optimization modes."""
    ULTRA_FAST = "ultra_fast"    # <10ms, basic accuracy
    FAST = "fast"                # <50ms, good accuracy
    BALANCED = "balanced"        # <100ms, balanced
    ACCURATE = "accurate"        # <500ms, high accuracy


@dataclass
class DetectionTarget:
    """Performance targets for detection."""
    max_time_ms: float
    min_confidence: float
    preferred_methods: List[str]
    fallback_methods: List[str]


class OptimizedDetector:
    """High-performance AI text detection engine."""
    
    def __init__(self, mode: PerformanceMode = PerformanceMode.BALANCED):
        self.mode = mode
        self.logger = get_logger(__name__)
        self.metrics = get_metrics_collector()
        self.cache = get_cache_manager()
        
        # Initialize detection components
        self.heuristic_detector = HeuristicDetector()
        self.pattern_detector = FastPatternDetector()
        self.ml_detector = FastMLDetector()
        self.optimizer = PerformanceOptimizer()
        
        # Performance targets by mode
        self.targets = {
            PerformanceMode.ULTRA_FAST: DetectionTarget(
                max_time_ms=10.0,
                min_confidence=0.6,
                preferred_methods=["heuristic"],
                fallback_methods=["pattern_fast"]
            ),
            PerformanceMode.FAST: DetectionTarget(
                max_time_ms=50.0,
                min_confidence=0.7,
                preferred_methods=["heuristic", "pattern"],
                fallback_methods=["ml_fast"]
            ),
            PerformanceMode.BALANCED: DetectionTarget(
                max_time_ms=100.0,
                min_confidence=0.75,
                preferred_methods=["pattern", "ml_fast"],
                fallback_methods=["heuristic"]
            ),
            PerformanceMode.ACCURATE: DetectionTarget(
                max_time_ms=500.0,
                min_confidence=0.8,
                preferred_methods=["ml", "ensemble_light"],
                fallback_methods=["pattern", "ml_fast"]
            )
        }
        
        # Method timing history for adaptive selection
        self.method_timings = {}
        
        # Cache configuration
        self.cache_enabled = True
        self.cache_ttl = 300  # 5 minutes
    
    async def detect(self, text: str, **kwargs) -> Dict[str, Any]:
        """Perform optimized AI detection."""
        start_time = time.time()
        request_id = kwargs.get("request_id", f"req_{int(time.time() * 1000)}")
        
        try:
            # Check cache first
            if self.cache_enabled:
                cached_result = await self._check_cache(text)
                if cached_result:
                    cached_result["from_cache"] = True
                    return cached_result
            
            # Get performance target
            target = self.targets[self.mode]
            
            # Fast precheck for obvious cases
            precheck_result = await self._fast_precheck(text)
            if precheck_result and precheck_result["confidence_score"] >= target.min_confidence:
                await self._cache_result(text, precheck_result)
                return precheck_result
            
            # Select optimal method based on text and performance target
            method = self._select_optimal_method(text, target)
            
            # Perform detection with timeout
            result = await self._detect_with_timeout(text, method, target.max_time_ms)
            
            # Post-process result
            result = self._post_process_result(result, start_time, request_id)
            
            # Cache successful result
            if result["confidence_score"] >= target.min_confidence:
                await self._cache_result(text, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
            return self._create_error_result(start_time, str(e))
    
    async def batch_detect(self, texts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Perform optimized batch detection."""
        start_time = time.time()
        
        # Group texts by length for optimal processing
        text_groups = self._group_texts_by_length(texts)
        
        all_results = []
        
        # Process each group with optimal method
        for group_name, group_texts in text_groups.items():
            if not group_texts:
                continue
            
            # Select method based on group characteristics
            method = self._select_batch_method(group_name, len(group_texts))
            
            # Process group
            if method == "heuristic":
                group_results = await self._batch_heuristic(group_texts)
            elif method == "pattern":
                group_results = await self._batch_pattern(group_texts)
            elif method == "ml_fast":
                group_results = await self._batch_ml_fast(group_texts)
            else:
                # Fallback to individual processing
                group_results = []
                for text in group_texts:
                    result = await self.detect(text)
                    group_results.append(result)
            
            all_results.extend(group_results)
        
        # Record batch metrics
        total_time = (time.time() - start_time) * 1000
        self.metrics.observe_histogram("batch_detection_ms", total_time)
        self.metrics.observe_gauge("batch_size", len(texts))
        
        return all_results
    
    def _select_optimal_method(self, text: str, target: DetectionTarget) -> str:
        """Select the optimal detection method based on text and performance target."""
        text_length = len(text)
        
        # Ultra-fast mode
        if self.mode == PerformanceMode.ULTRA_FAST:
            return "heuristic"
        
        # Fast mode
        elif self.mode == PerformanceMode.FAST:
            if text_length < 50:
                return "heuristic"
            else:
                return "pattern"
        
        # Balanced mode
        elif self.mode == PerformanceMode.BALANCED:
            if text_length < 100:
                return "pattern"
            elif text_length < 1000:
                return "ml_fast"
            else:
                return "pattern"  # Pattern is faster for very long text
        
        # Accurate mode
        else:
            if text_length < 200:
                return "ml_fast"
            else:
                return "ensemble_light"
    
    def _group_texts_by_length(self, texts: List[str]) -> Dict[str, List[str]]:
        """Group texts by length for batch optimization."""
        groups = {
            "short": [],    # < 100 chars
            "medium": [],   # 100-500 chars
            "long": [],     # 500-2000 chars
            "very_long": [] # > 2000 chars
        }
        
        for text in texts:
            length = len(text)
            if length < 100:
                groups["short"].append(text)
            elif length < 500:
                groups["medium"].append(text)
            elif length < 2000:
                groups["long"].append(text)
            else:
                groups["very_long"].append(text)
        
        return groups
    
    def _select_batch_method(self, group_name: str, group_size: int) -> str:
        """Select optimal method for batch processing."""
        if self.mode == PerformanceMode.ULTRA_FAST:
            return "heuristic"
        
        elif self.mode == PerformanceMode.FAST:
            if group_name == "short":
                return "heuristic"
            else:
                return "pattern"
        
        elif self.mode == PerformanceMode.BALANCED:
            if group_name == "short":
                return "pattern"
            elif group_name in ["medium", "long"]:
                return "ml_fast"
            else:
                return "pattern"
        
        else:  # ACCURATE
            if group_name == "short":
                return "ml_fast"
            else:
                return "individual"  # Process individually for accuracy
    
    async def _fast_precheck(self, text: str) -> Optional[Dict[str, Any]]:
        """Perform ultra-fast precheck for obvious cases."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.heuristic_detector.detect, text
        )
    
    async def _detect_with_timeout(self, text: str, method: str, timeout_ms: float) -> Dict[str, Any]:
        """Perform detection with timeout."""
        timeout_seconds = timeout_ms / 1000
        
        try:
            if method == "heuristic":
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, self.heuristic_detector.detect, text
                    ),
                    timeout=timeout_seconds
                )
            elif method == "pattern":
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, self.pattern_detector.detect, text
                    ),
                    timeout=timeout_seconds
                )
            elif method == "ml_fast":
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, self.ml_detector.detect, text
                    ),
                    timeout=timeout_seconds
                )
            else:
                # Fallback to pattern detection
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, self.pattern_detector.detect, text
                    ),
                    timeout=timeout_seconds
                )
            
            return result
            
        except asyncio.TimeoutError:
            # Return fast heuristic result on timeout
            return self.heuristic_detector.detect(text)
    
    async def _batch_heuristic(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Batch heuristic detection."""
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(None, self.heuristic_detector.detect, text)
            for text in texts
        ]
        return await asyncio.gather(*tasks)
    
    async def _batch_pattern(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Batch pattern detection."""
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(None, self.pattern_detector.detect, text)
            for text in texts
        ]
        return await asyncio.gather(*tasks)
    
    async def _batch_ml_fast(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Batch ML detection."""
        # Use the ML detector's built-in batch processing
        return await asyncio.get_event_loop().run_in_executor(
            None, self.ml_detector.batch_detect, texts
        )
    
    async def _check_cache(self, text: str) -> Optional[Dict[str, Any]]:
        """Check cache for previous results."""
        cache_key = self._generate_cache_key(text)
        cached_result = await self.cache.get(cache_key)
        
        if cached_result:
            self.metrics.increment_counter("cache_hits_total")
            return cached_result
        
        return None
    
    async def _cache_result(self, text: str, result: Dict[str, Any]):
        """Cache detection result."""
        cache_key = self._generate_cache_key(text)
        
        # Remove non-cacheable fields
        cacheable_result = {k: v for k, v in result.items() if k not in ["processing_time_ms", "request_id"]}
        
        await self.cache.set(cache_key, cacheable_result, ttl=self.cache_ttl)
    
    def _generate_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        import hashlib
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:16]
        return f"detection:{self.mode.value}:{text_hash}"
    
    def _post_process_result(self, result: Dict[str, Any], start_time: float, request_id: str) -> Dict[str, Any]:
        """Post-process detection result."""
        # Calculate total processing time
        total_time = (time.time() - start_time) * 1000
        
        # Update result
        result.update({
            "request_id": request_id,
            "total_processing_time_ms": total_time,
            "performance_mode": self.mode.value,
            "optimization_level": "high",
            "sub_100ms": total_time < 100,
            "sub_50ms": total_time < 50,
            "sub_10ms": total_time < 10
        })
        
        # Record performance metrics
        self.metrics.observe_histogram("optimized_detection_ms", total_time)
        self.metrics.increment_counter("optimized_detections_total")
        
        if total_time < 100:
            self.metrics.increment_counter("sub_100ms_detections_total")
        if total_time < 50:
            self.metrics.increment_counter("sub_50ms_detections_total")
        if total_time < 10:
            self.metrics.increment_counter("sub_10ms_detections_total")
        
        return result
    
    def _create_error_result(self, start_time: float, error_msg: str) -> Dict[str, Any]:
        """Create error result."""
        return {
            "is_ai_generated": None,
            "confidence_score": 0.0,
            "processing_time_ms": (time.time() - start_time) * 1000,
            "method_used": "error",
            "error": error_msg,
            "performance_mode": self.mode.value
        }
    
    async def benchmark(self, test_texts: List[str]) -> Dict[str, Any]:
        """Benchmark the optimized detector."""
        benchmark_start = time.time()
        
        # Test different performance modes
        mode_results = {}
        
        for mode in PerformanceMode:
            mode_detector = OptimizedDetector(mode)
            mode_start = time.time()
            
            # Test subset of texts for each mode
            test_subset = test_texts[:50]  # Limit for benchmarking
            results = []
            
            for text in test_subset:
                result = await mode_detector.detect(text)
                results.append(result)
            
            mode_time = (time.time() - mode_start) * 1000
            
            mode_results[mode.value] = {
                "total_time_ms": mode_time,
                "average_time_ms": mode_time / len(test_subset),
                "texts_per_second": len(test_subset) / (mode_time / 1000),
                "sub_100ms_count": sum(1 for r in results if r.get("total_processing_time_ms", 999) < 100),
                "sub_50ms_count": sum(1 for r in results if r.get("total_processing_time_ms", 999) < 50),
                "sub_10ms_count": sum(1 for r in results if r.get("total_processing_time_ms", 999) < 10),
                "average_confidence": sum(r.get("confidence_score", 0) for r in results) / len(results),
                "error_count": sum(1 for r in results if "error" in r)
            }
        
        total_benchmark_time = (time.time() - benchmark_start) * 1000
        
        return {
            "benchmark_time_ms": total_benchmark_time,
            "mode_results": mode_results,
            "recommendation": self._get_mode_recommendation(mode_results),
            "optimization_summary": {
                "cache_enabled": self.cache_enabled,
                "parallel_processing": True,
                "timeout_protection": True,
                "adaptive_method_selection": True
            }
        }
    
    def _get_mode_recommendation(self, mode_results: Dict[str, Dict[str, Any]]) -> str:
        """Get performance mode recommendation."""
        balanced_result = mode_results.get("balanced", {})
        avg_time = balanced_result.get("average_time_ms", 999)
        
        if avg_time < 10:
            return "Consider ULTRA_FAST mode for even better performance"
        elif avg_time < 50:
            return "FAST mode recommended for your use case"
        elif avg_time < 100:
            return "BALANCED mode is optimal for your requirements"
        else:
            return "Consider optimizing your infrastructure or using ULTRA_FAST mode"
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            "current_mode": self.mode.value,
            "performance_target": {
                "max_time_ms": self.targets[self.mode].max_time_ms,
                "min_confidence": self.targets[self.mode].min_confidence
            },
            "cache_stats": {
                "enabled": self.cache_enabled,
                "ttl_seconds": self.cache_ttl,
                "hit_rate": self._calculate_cache_hit_rate()
            },
            "method_usage": self._get_method_usage_stats(),
            "performance_metrics": {
                "total_detections": self.metrics.get_metric("optimized_detections_total").get_value() if self.metrics.get_metric("optimized_detections_total") else 0,
                "sub_100ms_rate": self._calculate_sub_100ms_rate(),
                "average_time_ms": self._calculate_average_time()
            }
        }
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        hits = self.metrics.get_metric("cache_hits_total")
        total = self.metrics.get_metric("optimized_detections_total")
        
        if hits and total and total.get_value() > 0:
            return hits.get_value() / total.get_value()
        return 0.0
    
    def _calculate_sub_100ms_rate(self) -> float:
        """Calculate rate of sub-100ms detections."""
        sub_100ms = self.metrics.get_metric("sub_100ms_detections_total")
        total = self.metrics.get_metric("optimized_detections_total")
        
        if sub_100ms and total and total.get_value() > 0:
            return sub_100ms.get_value() / total.get_value()
        return 0.0
    
    def _calculate_average_time(self) -> float:
        """Calculate average detection time."""
        histogram = self.metrics.get_metric("optimized_detection_ms")
        if histogram:
            stats = histogram.get_value()
            return stats.get("mean", 0.0)
        return 0.0
    
    def _get_method_usage_stats(self) -> Dict[str, int]:
        """Get method usage statistics."""
        # This would track which methods are used most frequently
        return {
            "heuristic": 0,
            "pattern": 0,
            "ml_fast": 0,
            "ensemble_light": 0
        }