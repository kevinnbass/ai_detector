"""
Performance Benchmarks for AI Detector System
Comprehensive performance testing and benchmarking suite
"""

import asyncio
import time
import statistics
import psutil
import json
import tracemalloc
from datetime import datetime, timedelta
from typing import List, Dict, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import pytest
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.core.detection.detector import GPT4oDetector
from src.core.data.processors import TextPreprocessor, FeatureExtractor
from src.core.data.collectors import ManualDataCollector
from src.core.services.detection_service import DetectionService
from src.core.api_client.unified_client import UnifiedAPIClient
from src.core.interfaces.data_interfaces import DataSample, DataBatch


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    operation: str
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    p50_time: float
    p95_time: float
    p99_time: float
    throughput: float  # operations per second
    memory_usage: Dict[str, float]
    cpu_usage: float
    error_count: int
    success_count: int
    timestamp: datetime


class PerformanceBenchmark:
    """Performance benchmark runner and metrics collector"""
    
    def __init__(self):
        self.results: List[PerformanceMetrics] = []
        self.process = psutil.Process()
        
    async def measure_async_operation(
        self, 
        operation_name: str,
        operation_func: Callable,
        iterations: int = 100,
        **kwargs
    ) -> PerformanceMetrics:
        """Measure performance of an async operation"""
        times = []
        errors = 0
        
        # Memory tracking
        tracemalloc.start()
        initial_memory = self.process.memory_info()
        
        # CPU tracking
        initial_cpu_times = self.process.cpu_times()
        start_time = time.time()
        
        for i in range(iterations):
            iter_start = time.perf_counter()
            try:
                await operation_func(**kwargs)
                iter_end = time.perf_counter()
                times.append(iter_end - iter_start)
            except Exception as e:
                errors += 1
                print(f"Error in iteration {i}: {e}")
        
        end_time = time.time()
        
        # Final measurements
        final_memory = self.process.memory_info()
        final_cpu_times = self.process.cpu_times()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Calculate metrics
        if times:
            total_time = end_time - start_time
            avg_time = statistics.mean(times)
            min_time = min(times)
            max_time = max(times)
            p50_time = statistics.median(times)
            p95_time = np.percentile(times, 95) if len(times) > 1 else times[0]
            p99_time = np.percentile(times, 99) if len(times) > 1 else times[0]
            throughput = len(times) / total_time
        else:
            total_time = avg_time = min_time = max_time = 0
            p50_time = p95_time = p99_time = throughput = 0
        
        # Memory usage in MB
        memory_usage = {
            "rss_mb": final_memory.rss / 1024 / 1024,
            "vms_mb": final_memory.vms / 1024 / 1024,
            "peak_traced_mb": peak / 1024 / 1024,
            "memory_increase_mb": (final_memory.rss - initial_memory.rss) / 1024 / 1024
        }
        
        # CPU usage calculation
        cpu_usage = (
            (final_cpu_times.user - initial_cpu_times.user) + 
            (final_cpu_times.system - initial_cpu_times.system)
        ) / total_time * 100 if total_time > 0 else 0
        
        metrics = PerformanceMetrics(
            operation=operation_name,
            total_time=total_time,
            avg_time=avg_time,
            min_time=min_time,
            max_time=max_time,
            p50_time=p50_time,
            p95_time=p95_time,
            p99_time=p99_time,
            throughput=throughput,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            error_count=errors,
            success_count=len(times),
            timestamp=datetime.now()
        )
        
        self.results.append(metrics)
        return metrics
    
    def measure_sync_operation(
        self,
        operation_name: str,
        operation_func: Callable,
        iterations: int = 100,
        **kwargs
    ) -> PerformanceMetrics:
        """Measure performance of a sync operation"""
        times = []
        errors = 0
        
        # Memory tracking
        tracemalloc.start()
        initial_memory = self.process.memory_info()
        
        # CPU tracking
        initial_cpu_times = self.process.cpu_times()
        start_time = time.time()
        
        for i in range(iterations):
            iter_start = time.perf_counter()
            try:
                operation_func(**kwargs)
                iter_end = time.perf_counter()
                times.append(iter_end - iter_start)
            except Exception as e:
                errors += 1
                print(f"Error in iteration {i}: {e}")
        
        end_time = time.time()
        
        # Final measurements
        final_memory = self.process.memory_info()
        final_cpu_times = self.process.cpu_times()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Calculate metrics (same as async version)
        if times:
            total_time = end_time - start_time
            avg_time = statistics.mean(times)
            min_time = min(times)
            max_time = max(times)
            p50_time = statistics.median(times)
            p95_time = np.percentile(times, 95) if len(times) > 1 else times[0]
            p99_time = np.percentile(times, 99) if len(times) > 1 else times[0]
            throughput = len(times) / total_time
        else:
            total_time = avg_time = min_time = max_time = 0
            p50_time = p95_time = p99_time = throughput = 0
        
        memory_usage = {
            "rss_mb": final_memory.rss / 1024 / 1024,
            "vms_mb": final_memory.vms / 1024 / 1024,
            "peak_traced_mb": peak / 1024 / 1024,
            "memory_increase_mb": (final_memory.rss - initial_memory.rss) / 1024 / 1024
        }
        
        cpu_usage = (
            (final_cpu_times.user - initial_cpu_times.user) + 
            (final_cpu_times.system - initial_cpu_times.system)
        ) / total_time * 100 if total_time > 0 else 0
        
        metrics = PerformanceMetrics(
            operation=operation_name,
            total_time=total_time,
            avg_time=avg_time,
            min_time=min_time,
            max_time=max_time,
            p50_time=p50_time,
            p95_time=p95_time,
            p99_time=p99_time,
            throughput=throughput,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            error_count=errors,
            success_count=len(times),
            timestamp=datetime.now()
        )
        
        self.results.append(metrics)
        return metrics
    
    async def measure_concurrent_operations(
        self,
        operation_name: str,
        operation_func: Callable,
        concurrent_count: int = 10,
        iterations_per_worker: int = 10,
        **kwargs
    ) -> PerformanceMetrics:
        """Measure performance under concurrent load"""
        start_time = time.time()
        tracemalloc.start()
        initial_memory = self.process.memory_info()
        
        async def worker():
            worker_times = []
            worker_errors = 0
            
            for _ in range(iterations_per_worker):
                iter_start = time.perf_counter()
                try:
                    await operation_func(**kwargs)
                    iter_end = time.perf_counter()
                    worker_times.append(iter_end - iter_start)
                except Exception:
                    worker_errors += 1
            
            return worker_times, worker_errors
        
        # Run concurrent workers
        tasks = [worker() for _ in range(concurrent_count)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate results
        all_times = []
        total_errors = 0
        
        for result in results:
            if isinstance(result, Exception):
                total_errors += iterations_per_worker
            else:
                times, errors = result
                all_times.extend(times)
                total_errors += errors
        
        end_time = time.time()
        final_memory = self.process.memory_info()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Calculate metrics
        if all_times:
            total_time = end_time - start_time
            avg_time = statistics.mean(all_times)
            min_time = min(all_times)
            max_time = max(all_times)
            p50_time = statistics.median(all_times)
            p95_time = np.percentile(all_times, 95)
            p99_time = np.percentile(all_times, 99)
            throughput = len(all_times) / total_time
        else:
            total_time = avg_time = min_time = max_time = 0
            p50_time = p95_time = p99_time = throughput = 0
        
        memory_usage = {
            "rss_mb": final_memory.rss / 1024 / 1024,
            "vms_mb": final_memory.vms / 1024 / 1024,
            "peak_traced_mb": peak / 1024 / 1024,
            "memory_increase_mb": (final_memory.rss - initial_memory.rss) / 1024 / 1024
        }
        
        metrics = PerformanceMetrics(
            operation=f"{operation_name}_concurrent_{concurrent_count}x{iterations_per_worker}",
            total_time=total_time,
            avg_time=avg_time,
            min_time=min_time,
            max_time=max_time,
            p50_time=p50_time,
            p95_time=p95_time,
            p99_time=p99_time,
            throughput=throughput,
            memory_usage=memory_usage,
            cpu_usage=0,  # Difficult to measure accurately in concurrent scenario
            error_count=total_errors,
            success_count=len(all_times),
            timestamp=datetime.now()
        )
        
        self.results.append(metrics)
        return metrics
    
    def export_results(self, filename: str = None) -> Dict[str, Any]:
        """Export benchmark results to JSON"""
        if not filename:
            filename = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        results_data = {
            "benchmark_info": {
                "timestamp": datetime.now().isoformat(),
                "system_info": {
                    "cpu_count": psutil.cpu_count(),
                    "memory_total_gb": psutil.virtual_memory().total / 1024**3,
                    "python_version": f"{psutil.version_info}",
                    "platform": psutil.Process().exe()
                }
            },
            "results": [asdict(result) for result in self.results]
        }
        
        # Convert datetime objects to strings for JSON serialization
        for result in results_data["results"]:
            result["timestamp"] = result["timestamp"].isoformat()
        
        output_path = Path("test-results") / filename
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        return results_data
    
    def print_summary(self):
        """Print a summary of benchmark results"""
        if not self.results:
            print("No benchmark results available")
            return
        
        print("\n" + "="*80)
        print("PERFORMANCE BENCHMARK SUMMARY")
        print("="*80)
        
        for result in self.results:
            print(f"\nOperation: {result.operation}")
            print(f"  Iterations: {result.success_count} successful, {result.error_count} errors")
            print(f"  Average Time: {result.avg_time*1000:.2f}ms")
            print(f"  P95 Time: {result.p95_time*1000:.2f}ms")
            print(f"  P99 Time: {result.p99_time*1000:.2f}ms")
            print(f"  Throughput: {result.throughput:.2f} ops/sec")
            print(f"  Memory Peak: {result.memory_usage['peak_traced_mb']:.2f}MB")
            print(f"  Memory Increase: {result.memory_usage['memory_increase_mb']:.2f}MB")
            
            # Performance indicators
            if result.avg_time > 1.0:
                print("  ⚠️  SLOW: Average time > 1 second")
            if result.p95_time > 2.0:
                print("  ⚠️  SLOW: P95 time > 2 seconds")
            if result.memory_usage['memory_increase_mb'] > 100:
                print("  ⚠️  HIGH MEMORY: Memory increase > 100MB")
            if result.error_count > 0:
                print(f"  ❌ ERRORS: {result.error_count} operations failed")


class DetectionBenchmarks:
    """Specific benchmarks for detection operations"""
    
    def __init__(self):
        self.benchmark = PerformanceBenchmark()
        
    async def setup_detector(self):
        """Setup detector for benchmarking"""
        self.detector = GPT4oDetector()
        await self.detector.initialize()
        
    async def benchmark_single_detection(self, text_length: int = 100):
        """Benchmark single text detection"""
        test_text = "This is a test text for performance benchmarking. " * (text_length // 50)
        
        async def detect_operation():
            return await self.detector.detect(test_text)
        
        return await self.benchmark.measure_async_operation(
            f"single_detection_{text_length}_chars",
            detect_operation,
            iterations=100
        )
    
    async def benchmark_batch_detection(self, batch_size: int = 10, text_length: int = 100):
        """Benchmark batch text detection"""
        test_texts = [
            "This is test text for batch performance benchmarking. " * (text_length // 50)
            for _ in range(batch_size)
        ]
        
        async def batch_detect_operation():
            return await self.detector.detect_batch(test_texts)
        
        return await self.benchmark.measure_async_operation(
            f"batch_detection_{batch_size}_texts_{text_length}_chars",
            batch_detect_operation,
            iterations=50
        )
    
    async def benchmark_concurrent_detection(self, concurrent_users: int = 5):
        """Benchmark concurrent detection requests"""
        test_text = "This text is used for concurrent detection benchmarking. " * 10
        
        async def detect_operation():
            return await self.detector.detect(test_text)
        
        return await self.benchmark.measure_concurrent_operations(
            "concurrent_detection",
            detect_operation,
            concurrent_count=concurrent_users,
            iterations_per_worker=20
        )
    
    async def benchmark_detection_scaling(self):
        """Benchmark detection performance across different text sizes"""
        text_sizes = [50, 100, 500, 1000, 5000, 10000]
        results = []
        
        for size in text_sizes:
            result = await self.benchmark_single_detection(text_length=size)
            results.append(result)
            
        return results


class DataProcessingBenchmarks:
    """Benchmarks for data processing operations"""
    
    def __init__(self):
        self.benchmark = PerformanceBenchmark()
        
    async def setup_processors(self):
        """Setup data processors for benchmarking"""
        self.preprocessor = TextPreprocessor()
        self.feature_extractor = FeatureExtractor()
        self.collector = ManualDataCollector()
        
        await self.preprocessor.initialize()
        await self.feature_extractor.initialize()
        await self.collector.initialize()
    
    async def benchmark_text_preprocessing(self, text_length: int = 1000):
        """Benchmark text preprocessing"""
        test_text = "This is a comprehensive text preprocessing benchmark. " * (text_length // 50)
        
        async def preprocess_operation():
            return await self.preprocessor.process_text(test_text)
        
        return await self.benchmark.measure_async_operation(
            f"text_preprocessing_{text_length}_chars",
            preprocess_operation,
            iterations=100
        )
    
    async def benchmark_feature_extraction(self, text_length: int = 1000):
        """Benchmark feature extraction"""
        test_text = "This text contains various linguistic features for extraction. " * (text_length // 60)
        
        async def extract_operation():
            return await self.feature_extractor.extract_features(test_text)
        
        return await self.benchmark.measure_async_operation(
            f"feature_extraction_{text_length}_chars",
            extract_operation,
            iterations=50
        )
    
    async def benchmark_data_collection(self, sample_count: int = 100):
        """Benchmark data collection operations"""
        samples = [
            DataSample(
                id=f"benchmark_sample_{i}",
                content=f"This is benchmark sample {i} for data collection testing.",
                label="human" if i % 2 == 0 else "ai"
            )
            for i in range(sample_count)
        ]
        
        async def collect_operation():
            for sample in samples:
                await self.collector.add_sample(sample)
        
        return await self.benchmark.measure_async_operation(
            f"data_collection_{sample_count}_samples",
            collect_operation,
            iterations=10
        )
    
    async def benchmark_batch_processing(self, batch_size: int = 100):
        """Benchmark batch data processing"""
        texts = [
            f"This is batch processing text {i} for performance testing."
            for i in range(batch_size)
        ]
        
        async def batch_preprocess_operation():
            return await self.preprocessor.process_batch(texts)
        
        async def batch_extract_operation():
            return await self.feature_extractor.extract_features_batch(texts)
        
        preprocess_result = await self.benchmark.measure_async_operation(
            f"batch_preprocessing_{batch_size}_texts",
            batch_preprocess_operation,
            iterations=20
        )
        
        extract_result = await self.benchmark.measure_async_operation(
            f"batch_feature_extraction_{batch_size}_texts",
            batch_extract_operation,
            iterations=20
        )
        
        return preprocess_result, extract_result


class APIBenchmarks:
    """Benchmarks for API client operations"""
    
    def __init__(self):
        self.benchmark = PerformanceBenchmark()
        
    async def setup_api_client(self):
        """Setup API client for benchmarking"""
        # Mock HTTP client for benchmarking
        from unittest.mock import Mock, AsyncMock
        from src.core.interfaces.api_interfaces import APIResponse
        
        mock_http_client = Mock()
        mock_http_client.request = AsyncMock(return_value=APIResponse(
            status_code=200,
            headers={"Content-Type": "application/json"},
            body={"prediction": "ai", "confidence": 0.85}
        ))
        
        self.api_client = UnifiedAPIClient(
            base_url="https://api.test.com",
            http_client=mock_http_client
        )
        await self.api_client.initialize()
    
    async def benchmark_api_requests(self, request_count: int = 100):
        """Benchmark API request performance"""
        from src.core.interfaces.api_interfaces import HTTPMethod
        
        async def api_request_operation():
            return await self.api_client.request(
                method=HTTPMethod.POST,
                endpoint="/detect",
                data={"text": "Test text for API benchmarking"}
            )
        
        return await self.benchmark.measure_async_operation(
            f"api_requests_{request_count}",
            api_request_operation,
            iterations=request_count
        )
    
    async def benchmark_api_with_queue(self):
        """Benchmark API requests with queuing enabled"""
        from src.core.interfaces.api_interfaces import HTTPMethod
        from src.core.api_client.queue_manager import Priority
        
        async def queued_request_operation():
            return await self.api_client.request(
                method=HTTPMethod.POST,
                endpoint="/detect",
                data={"text": "Queued request benchmark"},
                use_queue=True,
                priority=Priority.NORMAL
            )
        
        return await self.benchmark.measure_async_operation(
            "api_requests_with_queue",
            queued_request_operation,
            iterations=50
        )
    
    async def benchmark_api_with_cache(self):
        """Benchmark API requests with caching"""
        from src.core.interfaces.api_interfaces import HTTPMethod
        
        async def cached_request_operation():
            return await self.api_client.request(
                method=HTTPMethod.GET,
                endpoint="/cached-data",
                use_cache=True
            )
        
        return await self.benchmark.measure_async_operation(
            "api_requests_with_cache",
            cached_request_operation,
            iterations=100
        )


class SystemBenchmarks:
    """End-to-end system benchmarks"""
    
    def __init__(self):
        self.benchmark = PerformanceBenchmark()
    
    async def setup_system(self):
        """Setup complete system for benchmarking"""
        # Setup detector and service
        detector = GPT4oDetector()
        await detector.initialize()
        
        self.detection_service = DetectionService(detector=detector)
        await self.detection_service.start()
    
    async def benchmark_end_to_end_detection(self):
        """Benchmark complete detection workflow"""
        test_text = "This is an end-to-end detection benchmark text that simulates real user input."
        
        async def e2e_detection_operation():
            # Simulate complete workflow: preprocess -> detect -> store result
            result = await self.detection_service.detect_text(test_text)
            return result
        
        return await self.benchmark.measure_async_operation(
            "end_to_end_detection",
            e2e_detection_operation,
            iterations=50
        )
    
    async def benchmark_high_throughput(self, target_rps: int = 100):
        """Benchmark system under high throughput load"""
        test_texts = [
            f"High throughput benchmark text {i} for system stress testing."
            for i in range(target_rps)
        ]
        
        async def throughput_operation():
            tasks = []
            for text in test_texts:
                task = self.detection_service.detect_text(text)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            successful = sum(1 for r in results if not isinstance(r, Exception))
            return successful
        
        return await self.benchmark.measure_async_operation(
            f"high_throughput_{target_rps}_rps",
            throughput_operation,
            iterations=10
        )


# Performance test fixtures and utilities
@pytest.fixture
async def detection_benchmarks():
    """Fixture for detection benchmarks"""
    benchmarks = DetectionBenchmarks()
    await benchmarks.setup_detector()
    return benchmarks


@pytest.fixture
async def data_processing_benchmarks():
    """Fixture for data processing benchmarks"""
    benchmarks = DataProcessingBenchmarks()
    await benchmarks.setup_processors()
    return benchmarks


@pytest.fixture
async def api_benchmarks():
    """Fixture for API benchmarks"""
    benchmarks = APIBenchmarks()
    await benchmarks.setup_api_client()
    return benchmarks


@pytest.fixture
async def system_benchmarks():
    """Fixture for system benchmarks"""
    benchmarks = SystemBenchmarks()
    await benchmarks.setup_system()
    return benchmarks


# Performance test functions
@pytest.mark.performance
@pytest.mark.asyncio
async def test_detection_performance_requirements(detection_benchmarks):
    """Test that detection meets performance requirements"""
    # Requirement: Detection should complete in <100ms for typical text
    result = await detection_benchmarks.benchmark_single_detection(text_length=500)
    
    assert result.p95_time < 0.1, f"P95 detection time {result.p95_time*1000:.2f}ms exceeds 100ms requirement"
    assert result.throughput > 10, f"Throughput {result.throughput:.2f} ops/sec too low"
    assert result.error_count == 0, f"Detection had {result.error_count} errors"


@pytest.mark.performance
@pytest.mark.asyncio
async def test_batch_processing_performance(detection_benchmarks):
    """Test batch processing performance"""
    # Requirement: Batch processing should be more efficient than individual requests
    single_result = await detection_benchmarks.benchmark_single_detection()
    batch_result = await detection_benchmarks.benchmark_batch_detection(batch_size=10)
    
    # Batch should have higher throughput per item
    single_throughput = single_result.throughput
    batch_throughput_per_item = batch_result.throughput * 10  # 10 items per batch
    
    assert batch_throughput_per_item > single_throughput, "Batch processing not more efficient"


@pytest.mark.performance
@pytest.mark.asyncio
async def test_concurrent_load_handling(detection_benchmarks):
    """Test system performance under concurrent load"""
    result = await detection_benchmarks.benchmark_concurrent_detection(concurrent_users=10)
    
    # Should handle concurrent load without excessive degradation
    assert result.p95_time < 0.5, f"P95 time under load {result.p95_time*1000:.2f}ms too high"
    assert result.error_count < result.success_count * 0.05, "Too many errors under concurrent load"


@pytest.mark.performance
@pytest.mark.asyncio
async def test_memory_usage_limits(data_processing_benchmarks):
    """Test memory usage stays within limits"""
    result = await data_processing_benchmarks.benchmark_batch_processing(batch_size=100)
    
    # Memory increase should be reasonable
    assert result.memory_usage['memory_increase_mb'] < 50, "Memory increase too high"
    assert result.memory_usage['peak_traced_mb'] < 100, "Peak memory usage too high"


@pytest.mark.performance
@pytest.mark.asyncio
async def test_api_response_time_requirements(api_benchmarks):
    """Test API response time requirements"""
    # Requirement: API responses should complete in <2s
    result = await api_benchmarks.benchmark_api_requests(request_count=50)
    
    assert result.p95_time < 2.0, f"API P95 response time {result.p95_time:.2f}s exceeds 2s requirement"
    assert result.avg_time < 1.0, f"API average response time {result.avg_time:.2f}s too high"


@pytest.mark.performance
@pytest.mark.asyncio
async def test_system_throughput_requirements(system_benchmarks):
    """Test end-to-end system throughput"""
    # Requirement: System should handle >1000 tweets/min = ~16.7 tweets/sec
    result = await system_benchmarks.benchmark_end_to_end_detection()
    
    assert result.throughput > 16.7, f"System throughput {result.throughput:.2f} ops/sec below requirement"


if __name__ == "__main__":
    # Run all benchmarks when executed directly
    async def run_all_benchmarks():
        print("Starting comprehensive performance benchmarks...")
        
        # Detection benchmarks
        detection_bench = DetectionBenchmarks()
        await detection_bench.setup_detector()
        
        await detection_bench.benchmark_single_detection()
        await detection_bench.benchmark_batch_detection()
        await detection_bench.benchmark_concurrent_detection()
        
        # Data processing benchmarks
        data_bench = DataProcessingBenchmarks()
        await data_bench.setup_processors()
        
        await data_bench.benchmark_text_preprocessing()
        await data_bench.benchmark_feature_extraction()
        await data_bench.benchmark_batch_processing()
        
        # API benchmarks
        api_bench = APIBenchmarks()
        await api_bench.setup_api_client()
        
        await api_bench.benchmark_api_requests()
        await api_bench.benchmark_api_with_queue()
        await api_bench.benchmark_api_with_cache()
        
        # System benchmarks
        system_bench = SystemBenchmarks()
        await system_bench.setup_system()
        
        await system_bench.benchmark_end_to_end_detection()
        await system_bench.benchmark_high_throughput()
        
        # Print comprehensive summary
        print("\n" + "="*100)
        print("DETECTION BENCHMARKS")
        detection_bench.benchmark.print_summary()
        
        print("\n" + "="*100)
        print("DATA PROCESSING BENCHMARKS")
        data_bench.benchmark.print_summary()
        
        print("\n" + "="*100)
        print("API BENCHMARKS")
        api_bench.benchmark.print_summary()
        
        print("\n" + "="*100)
        print("SYSTEM BENCHMARKS")
        system_bench.benchmark.print_summary()
        
        # Export all results
        all_results = []
        all_results.extend(detection_bench.benchmark.results)
        all_results.extend(data_bench.benchmark.results)
        all_results.extend(api_bench.benchmark.results)
        all_results.extend(system_bench.benchmark.results)
        
        combined_benchmark = PerformanceBenchmark()
        combined_benchmark.results = all_results
        combined_benchmark.export_results("comprehensive_benchmark_results.json")
        
        print(f"\nBenchmark results exported to test-results/")
    
    asyncio.run(run_all_benchmarks())