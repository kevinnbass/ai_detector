"""
Performance tests for detection speed optimization.

Tests to ensure detection times are consistently under 100ms
across different text types and scenarios.
"""

import pytest
import asyncio
import time
from typing import List, Dict, Any
import statistics

from src.core.detection.optimized_detector import OptimizedDetector, PerformanceMode
from src.core.detection.fast_pattern_detector import FastPatternDetector
from src.core.detection.fast_ml_detector import FastMLDetector


class TestDetectionSpeed:
    """Test suite for detection speed optimization."""
    
    @pytest.fixture
    def sample_texts(self) -> Dict[str, List[str]]:
        """Sample texts of different lengths and types."""
        return {
            "very_short": [
                "OK", "Yes", "No thanks", "Got it!", "Sounds good",
                "Maybe", "Sure thing", "Alright", "Perfect", "Exactly"
            ],
            "short": [
                "This is a short text sample for testing.",
                "Quick brown fox jumps over the lazy dog.",
                "Simple sentence with basic structure here.",
                "Testing detection speed with short content.",
                "Another example of brief text input."
            ],
            "medium": [
                "This comprehensive analysis demonstrates the multifaceted nature of contemporary discourse paradigms, necessitating careful consideration of various methodological approaches and theoretical frameworks.",
                "The implementation of sophisticated algorithms requires meticulous attention to detail and adherence to established protocols, ensuring optimal performance across diverse operational contexts.",
                "Furthermore, it should be noted that the integration of advanced technologies facilitates enhanced functionality while maintaining compatibility with existing infrastructure components.",
                "In conclusion, the systematic evaluation of these variables provides valuable insights into the underlying mechanisms that govern complex behavioral patterns.",
                "However, one must consider the potential implications of these findings within the broader context of current research methodologies and established scientific paradigms."
            ],
            "long": [
                "The rapid advancement of artificial intelligence technologies has fundamentally transformed the landscape of digital communication, creating unprecedented opportunities for innovation while simultaneously presenting significant challenges related to authenticity and verification. This comprehensive analysis examines the multifaceted nature of contemporary AI-generated content, exploring various detection methodologies and their respective efficacy rates across diverse textual domains. Furthermore, the integration of sophisticated pattern recognition algorithms with traditional linguistic analysis techniques has facilitated the development of robust detection frameworks capable of identifying subtle markers indicative of automated content generation. However, it is important to note that the continuous evolution of language models necessitates ongoing refinement of detection strategies to maintain optimal performance levels.",
                "In the context of modern computational linguistics, the emergence of large language models has introduced novel paradigms for text generation that challenge conventional approaches to content authentication. The implementation of advanced neural architectures enables the production of increasingly sophisticated textual outputs that closely approximate human writing patterns, thereby complicating traditional detection methodologies. Consequently, researchers have focused on developing innovative analytical frameworks that leverage multiple indicators including syntactic complexity, semantic coherence, and stylistic consistency to establish comprehensive assessment protocols. These multidimensional approaches demonstrate superior performance compared to single-metric evaluation systems, particularly when applied to diverse textual domains ranging from academic discourse to casual communication."
            ],
            "ai_generated": [
                "This comprehensive analysis demonstrates the multifaceted nature of contemporary discourse paradigms.",
                "Furthermore, it should be noted that careful consideration of multiple factors is required.",
                "The implementation of sophisticated methodologies facilitates enhanced outcomes.",
                "However, one must acknowledge the potential implications of these findings.",
                "In conclusion, the systematic evaluation provides valuable insights into underlying mechanisms."
            ],
            "human_written": [
                "lol that's so funny 😂 can't believe it happened",
                "omg just saw the craziest thing at starbucks today!!",
                "my cat knocked over my coffee this morning 🙄 what a mess",
                "anyone else think this weather is getting weird?? like seriously",
                "just finished binge watching that new show - totally recommend it!"
            ]
        }
    
    @pytest.fixture
    def optimized_detector(self) -> OptimizedDetector:
        """Create optimized detector for testing."""
        return OptimizedDetector(mode=PerformanceMode.BALANCED)
    
    @pytest.fixture
    def fast_pattern_detector(self) -> FastPatternDetector:
        """Create fast pattern detector for testing."""
        return FastPatternDetector()
    
    @pytest.fixture
    def fast_ml_detector(self) -> FastMLDetector:
        """Create fast ML detector for testing."""
        return FastMLDetector()
    
    @pytest.mark.asyncio
    async def test_sub_100ms_detection_target(self, optimized_detector, sample_texts):
        """Test that detection consistently achieves sub-100ms target."""
        all_texts = []
        for text_list in sample_texts.values():
            all_texts.extend(text_list)
        
        times = []
        results = []
        
        for text in all_texts:
            start_time = time.time()
            result = await optimized_detector.detect(text)
            detection_time = (time.time() - start_time) * 1000
            
            times.append(detection_time)
            results.append(result)
        
        # Check that at least 90% of detections are under 100ms
        sub_100ms_count = sum(1 for t in times if t < 100)
        sub_100ms_rate = sub_100ms_count / len(times)
        
        assert sub_100ms_rate >= 0.9, f"Only {sub_100ms_rate:.1%} of detections were under 100ms"
        
        # Check average time
        avg_time = statistics.mean(times)
        assert avg_time < 100, f"Average detection time {avg_time:.1f}ms exceeds 100ms target"
        
        # Check that all results are valid
        for result in results:
            assert "is_ai_generated" in result
            assert "confidence_score" in result
            assert "processing_time_ms" in result
    
    @pytest.mark.asyncio 
    async def test_ultra_fast_mode_performance(self, sample_texts):
        """Test ultra-fast mode achieves sub-10ms target."""
        ultra_fast_detector = OptimizedDetector(mode=PerformanceMode.ULTRA_FAST)
        
        # Test with short texts for ultra-fast mode
        test_texts = sample_texts["short"] + sample_texts["very_short"]
        
        times = []
        for text in test_texts:
            start_time = time.time()
            result = await ultra_fast_detector.detect(text)
            detection_time = (time.time() - start_time) * 1000
            times.append(detection_time)
        
        # Ultra-fast mode should achieve sub-10ms for most texts
        sub_10ms_count = sum(1 for t in times if t < 10)
        sub_10ms_rate = sub_10ms_count / len(times)
        
        assert sub_10ms_rate >= 0.7, f"Only {sub_10ms_rate:.1%} of detections were under 10ms in ultra-fast mode"
        
        avg_time = statistics.mean(times)
        assert avg_time < 20, f"Average time {avg_time:.1f}ms too high for ultra-fast mode"
    
    @pytest.mark.asyncio
    async def test_fast_mode_performance(self, sample_texts):
        """Test fast mode achieves sub-50ms target."""
        fast_detector = OptimizedDetector(mode=PerformanceMode.FAST)
        
        test_texts = sample_texts["short"] + sample_texts["medium"]
        
        times = []
        for text in test_texts:
            start_time = time.time()
            result = await fast_detector.detect(text)
            detection_time = (time.time() - start_time) * 1000
            times.append(detection_time)
        
        # Fast mode should achieve sub-50ms for most texts
        sub_50ms_count = sum(1 for t in times if t < 50)
        sub_50ms_rate = sub_50ms_count / len(times)
        
        assert sub_50ms_rate >= 0.8, f"Only {sub_50ms_rate:.1%} of detections were under 50ms in fast mode"
        
        avg_time = statistics.mean(times)
        assert avg_time < 50, f"Average time {avg_time:.1f}ms exceeds 50ms target for fast mode"
    
    def test_pattern_detector_speed(self, fast_pattern_detector, sample_texts):
        """Test pattern detector speed across different text lengths."""
        times_by_length = {}
        
        for length_category, texts in sample_texts.items():
            times = []
            for text in texts:
                start_time = time.time()
                result = fast_pattern_detector.detect(text)
                detection_time = (time.time() - start_time) * 1000
                times.append(detection_time)
            
            times_by_length[length_category] = {
                "times": times,
                "avg_time": statistics.mean(times),
                "max_time": max(times),
                "sub_100ms_rate": sum(1 for t in times if t < 100) / len(times)
            }
        
        # Verify performance across all categories
        for category, stats in times_by_length.items():
            assert stats["avg_time"] < 100, f"Pattern detector avg time for {category} texts: {stats['avg_time']:.1f}ms"
            assert stats["sub_100ms_rate"] >= 0.9, f"Pattern detector sub-100ms rate for {category}: {stats['sub_100ms_rate']:.1%}"
    
    def test_ml_detector_speed(self, fast_ml_detector, sample_texts):
        """Test ML detector speed across different text lengths."""
        times_by_length = {}
        
        for length_category, texts in sample_texts.items():
            if length_category == "very_short":
                continue  # Skip very short texts for ML
            
            times = []
            for text in texts:
                start_time = time.time()
                result = fast_ml_detector.detect(text)
                detection_time = (time.time() - start_time) * 1000
                times.append(detection_time)
            
            times_by_length[length_category] = {
                "times": times,
                "avg_time": statistics.mean(times),
                "max_time": max(times),
                "sub_100ms_rate": sum(1 for t in times if t < 100) / len(times)
            }
        
        # Verify ML performance
        for category, stats in times_by_length.items():
            assert stats["avg_time"] < 150, f"ML detector avg time for {category} texts: {stats['avg_time']:.1f}ms"
            # ML detector may be slightly slower but should still be fast
            assert stats["sub_100ms_rate"] >= 0.7, f"ML detector sub-100ms rate for {category}: {stats['sub_100ms_rate']:.1%}"
    
    @pytest.mark.asyncio
    async def test_batch_processing_speed(self, optimized_detector, sample_texts):
        """Test batch processing performance improvements."""
        # Test with medium-sized batch
        test_texts = sample_texts["short"] + sample_texts["medium"]
        
        # Single processing
        single_start = time.time()
        single_results = []
        for text in test_texts:
            result = await optimized_detector.detect(text)
            single_results.append(result)
        single_time = (time.time() - single_start) * 1000
        
        # Batch processing
        batch_start = time.time()
        batch_results = await optimized_detector.batch_detect(test_texts)
        batch_time = (time.time() - batch_start) * 1000
        
        # Batch should be faster or similar
        speedup = single_time / batch_time
        assert speedup >= 0.8, f"Batch processing slower than expected (speedup: {speedup:.2f}x)"
        
        # Verify results are similar
        assert len(batch_results) == len(single_results)
        assert all("is_ai_generated" in result for result in batch_results)
    
    @pytest.mark.asyncio
    async def test_concurrent_detection_performance(self, optimized_detector, sample_texts):
        """Test performance under concurrent load."""
        test_texts = sample_texts["short"] * 3  # Repeat for more tests
        
        # Sequential processing
        sequential_start = time.time()
        sequential_results = []
        for text in test_texts:
            result = await optimized_detector.detect(text)
            sequential_results.append(result)
        sequential_time = (time.time() - sequential_start) * 1000
        
        # Concurrent processing
        concurrent_start = time.time()
        tasks = [optimized_detector.detect(text) for text in test_texts]
        concurrent_results = await asyncio.gather(*tasks)
        concurrent_time = (time.time() - concurrent_start) * 1000
        
        # Concurrent should be faster
        speedup = sequential_time / concurrent_time
        assert speedup >= 1.5, f"Concurrent processing not fast enough (speedup: {speedup:.2f}x)"
        
        # Verify results quality
        assert len(concurrent_results) == len(sequential_results)
    
    @pytest.mark.asyncio
    async def test_cache_performance_impact(self, optimized_detector, sample_texts):
        """Test cache impact on performance."""
        test_text = sample_texts["medium"][0]
        
        # First call (cache miss)
        start_time = time.time()
        result1 = await optimized_detector.detect(test_text)
        first_call_time = (time.time() - start_time) * 1000
        
        # Second call (cache hit)
        start_time = time.time()
        result2 = await optimized_detector.detect(test_text)
        second_call_time = (time.time() - start_time) * 1000
        
        # Cache hit should be significantly faster
        speedup = first_call_time / second_call_time
        assert speedup >= 2.0, f"Cache not providing expected speedup (speedup: {speedup:.2f}x)"
        
        # Results should be equivalent
        assert result1["is_ai_generated"] == result2["is_ai_generated"]
        assert abs(result1["confidence_score"] - result2["confidence_score"]) < 0.1
    
    @pytest.mark.asyncio
    async def test_performance_degradation_under_load(self, optimized_detector, sample_texts):
        """Test performance degradation under high load."""
        # Create high load scenario
        load_texts = []
        for _ in range(10):  # Repeat texts to create load
            load_texts.extend(sample_texts["short"])
            load_texts.extend(sample_texts["medium"])
        
        times = []
        
        # Process under load
        for text in load_texts:
            start_time = time.time()
            result = await optimized_detector.detect(text)
            detection_time = (time.time() - start_time) * 1000
            times.append(detection_time)
        
        # Check for performance degradation
        first_quarter = times[:len(times)//4]
        last_quarter = times[-len(times)//4:]
        
        avg_first = statistics.mean(first_quarter)
        avg_last = statistics.mean(last_quarter)
        
        degradation = avg_last / avg_first
        assert degradation < 2.0, f"Performance degraded too much under load (degradation: {degradation:.2f}x)"
        
        # Overall performance should still be good
        overall_avg = statistics.mean(times)
        assert overall_avg < 150, f"Average performance under load too slow: {overall_avg:.1f}ms"
    
    @pytest.mark.asyncio
    async def test_accuracy_vs_speed_tradeoff(self, sample_texts):
        """Test accuracy vs speed tradeoff across performance modes."""
        test_texts = sample_texts["ai_generated"] + sample_texts["human_written"]
        
        mode_performance = {}
        
        for mode in PerformanceMode:
            detector = OptimizedDetector(mode=mode)
            
            times = []
            accuracies = []
            
            for text in test_texts:
                start_time = time.time()
                result = await detector.detect(text)
                detection_time = (time.time() - start_time) * 1000
                
                times.append(detection_time)
                # Simple accuracy check based on expected results
                if text in sample_texts["ai_generated"]:
                    accuracy = 1.0 if result["is_ai_generated"] else 0.0
                else:
                    accuracy = 1.0 if not result["is_ai_generated"] else 0.0
                accuracies.append(accuracy)
            
            mode_performance[mode.value] = {
                "avg_time": statistics.mean(times),
                "accuracy": statistics.mean(accuracies),
                "sub_target_rate": self._calculate_sub_target_rate(times, mode)
            }
        
        # Verify tradeoff expectations
        ultra_fast = mode_performance["ultra_fast"]
        balanced = mode_performance["balanced"]
        accurate = mode_performance["accurate"]
        
        # Ultra-fast should be fastest
        assert ultra_fast["avg_time"] <= balanced["avg_time"]
        assert balanced["avg_time"] <= accurate["avg_time"]
        
        # Accuracy should generally improve with time (though this is a simple test)
        # At minimum, ensure no mode is completely broken
        for mode_data in mode_performance.values():
            assert mode_data["accuracy"] >= 0.3, "Mode accuracy too low"
    
    def _calculate_sub_target_rate(self, times: List[float], mode: PerformanceMode) -> float:
        """Calculate rate of detections meeting mode target time."""
        targets = {
            PerformanceMode.ULTRA_FAST: 10.0,
            PerformanceMode.FAST: 50.0,
            PerformanceMode.BALANCED: 100.0,
            PerformanceMode.ACCURATE: 500.0
        }
        
        target = targets[mode]
        sub_target_count = sum(1 for t in times if t < target)
        return sub_target_count / len(times)
    
    def test_memory_efficiency(self, optimized_detector, sample_texts):
        """Test memory efficiency during detection."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Measure initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process many texts
        test_texts = []
        for _ in range(100):
            test_texts.extend(sample_texts["short"])
        
        # Process all texts
        for text in test_texts:
            result = optimized_detector.heuristic_detector.detect(text)
        
        # Measure final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 50, f"Memory usage increased by {memory_increase:.1f}MB - possible memory leak"
    
    @pytest.mark.asyncio
    async def test_performance_consistency(self, optimized_detector, sample_texts):
        """Test performance consistency across multiple runs."""
        test_text = sample_texts["medium"][0]
        
        times = []
        
        # Run multiple times
        for _ in range(20):
            start_time = time.time()
            result = await optimized_detector.detect(test_text)
            detection_time = (time.time() - start_time) * 1000
            times.append(detection_time)
        
        # Check consistency
        avg_time = statistics.mean(times)
        std_dev = statistics.stdev(times)
        
        # Standard deviation should be reasonable
        coefficient_of_variation = std_dev / avg_time
        assert coefficient_of_variation < 0.5, f"Performance too inconsistent (CV: {coefficient_of_variation:.2f})"
        
        # Most runs should be within 2 standard deviations
        within_2_std = sum(1 for t in times if abs(t - avg_time) <= 2 * std_dev)
        consistency_rate = within_2_std / len(times)
        assert consistency_rate >= 0.9, f"Performance consistency too low: {consistency_rate:.1%}"