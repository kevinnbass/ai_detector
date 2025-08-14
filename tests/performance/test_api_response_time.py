"""
Performance tests for API response time optimization.

Tests to ensure API responses are consistently under 2 seconds
across different request types and load scenarios.
"""

import pytest
import asyncio
import time
import httpx
from typing import List, Dict, Any
import statistics

from src.api.performance.api_optimizer import APIPerformanceOptimizer, APIPerformanceConfig
from src.api.rest.optimized_app import app


class TestAPIResponseTime:
    """Test suite for API response time optimization."""
    
    @pytest.fixture
    def api_config(self) -> APIPerformanceConfig:
        """Create optimized API configuration."""
        return APIPerformanceConfig(
            max_concurrent_requests=50,
            request_timeout_seconds=1.8,
            enable_compression=True,
            compression_threshold=512,
            enable_request_batching=True,
            batch_size=10,
            batch_timeout_ms=30,
            enable_response_caching=True,
            cache_ttl_seconds=300
        )
    
    @pytest.fixture
    def api_optimizer(self, api_config) -> APIPerformanceOptimizer:
        """Create API optimizer instance."""
        return APIPerformanceOptimizer(api_config)
    
    @pytest.fixture
    def sample_requests(self) -> Dict[str, List[Dict[str, Any]]]:
        """Sample API requests of different types."""
        return {
            "short_texts": [
                {"text": "OK", "performance_mode": "ultra_fast"},
                {"text": "Yes, that works", "performance_mode": "ultra_fast"},
                {"text": "Simple test message", "performance_mode": "fast"},
                {"text": "Quick response needed", "performance_mode": "fast"},
                {"text": "Brief content analysis", "performance_mode": "fast"}
            ],
            "medium_texts": [
                {
                    "text": "This comprehensive analysis demonstrates the multifaceted nature of contemporary discourse paradigms.",
                    "performance_mode": "balanced"
                },
                {
                    "text": "The implementation of sophisticated algorithms requires meticulous attention to detail and adherence to protocols.",
                    "performance_mode": "balanced"
                },
                {
                    "text": "Furthermore, it should be noted that the integration of advanced technologies facilitates enhanced functionality.",
                    "performance_mode": "balanced"
                },
                {
                    "text": "In conclusion, the systematic evaluation provides valuable insights into underlying mechanisms.",
                    "performance_mode": "balanced"
                },
                {
                    "text": "However, one must consider the potential implications within the broader research context.",
                    "performance_mode": "balanced"
                }
            ],
            "long_texts": [
                {
                    "text": "The rapid advancement of artificial intelligence technologies has fundamentally transformed the landscape of digital communication, creating unprecedented opportunities for innovation while simultaneously presenting significant challenges related to authenticity and verification. This comprehensive analysis examines the multifaceted nature of contemporary AI-generated content, exploring various detection methodologies and their respective efficacy rates across diverse textual domains.",
                    "performance_mode": "accurate"
                },
                {
                    "text": "In the context of modern computational linguistics, the emergence of large language models has introduced novel paradigms for text generation that challenge conventional approaches to content authentication. The implementation of advanced neural architectures enables the production of increasingly sophisticated textual outputs that closely approximate human writing patterns, thereby complicating traditional detection methodologies.",
                    "performance_mode": "accurate"
                }
            ],
            "human_texts": [
                {"text": "lol that's so funny 😂 can't believe it happened", "performance_mode": "fast"},
                {"text": "omg just saw the craziest thing at starbucks today!!", "performance_mode": "fast"},
                {"text": "my cat knocked over my coffee this morning 🙄", "performance_mode": "fast"}
            ]
        }
    
    @pytest.mark.asyncio
    async def test_sub_2s_response_target(self, api_optimizer, sample_requests):
        """Test that API responses consistently achieve sub-2s target."""
        all_requests = []
        for request_list in sample_requests.values():
            all_requests.extend(request_list)
        
        response_times = []
        
        for request_data in all_requests:
            start_time = time.time()
            response_data, headers = await api_optimizer.process_request(request_data)
            response_time = (time.time() - start_time) * 1000
            
            response_times.append(response_time)
        
        # Check that at least 95% of responses are under 2 seconds
        sub_2s_count = sum(1 for t in response_times if t < 2000)
        sub_2s_rate = sub_2s_count / len(response_times)
        
        assert sub_2s_rate >= 0.95, f"Only {sub_2s_rate:.1%} of responses were under 2s"
        
        # Check average response time
        avg_time = statistics.mean(response_times)
        assert avg_time < 2000, f"Average response time {avg_time:.1f}ms exceeds 2s target"
        
        # Check 99th percentile
        p99_time = statistics.quantiles(sorted(response_times), n=100)[98]  # 99th percentile
        assert p99_time < 2000, f"99th percentile response time {p99_time:.1f}ms exceeds 2s"
    
    @pytest.mark.asyncio
    async def test_sub_1s_response_optimization(self, api_optimizer, sample_requests):
        """Test sub-1s response optimization for fast modes."""
        fast_requests = sample_requests["short_texts"] + sample_requests["human_texts"]
        
        response_times = []
        
        for request_data in fast_requests:
            start_time = time.time()
            response_data, headers = await api_optimizer.process_request(request_data)
            response_time = (time.time() - start_time) * 1000
            
            response_times.append(response_time)
        
        # Fast requests should achieve sub-1s for most cases
        sub_1s_count = sum(1 for t in response_times if t < 1000)
        sub_1s_rate = sub_1s_count / len(response_times)
        
        assert sub_1s_rate >= 0.8, f"Only {sub_1s_rate:.1%} of fast requests were under 1s"
        
        avg_time = statistics.mean(response_times)
        assert avg_time < 1000, f"Average fast response time {avg_time:.1f}ms exceeds 1s"
    
    @pytest.mark.asyncio
    async def test_response_compression_performance(self, api_optimizer, sample_requests):
        """Test response compression impact on performance."""
        # Test with longer responses that benefit from compression
        long_requests = sample_requests["long_texts"] + sample_requests["medium_texts"]
        
        compressed_times = []
        uncompressed_times = []
        
        # Test with compression enabled
        for request_data in long_requests:
            start_time = time.time()
            response_data, headers = await api_optimizer.process_request(request_data)
            compressed_time = (time.time() - start_time) * 1000
            compressed_times.append(compressed_time)
        
        # Test with compression disabled
        api_optimizer.config.enable_compression = False
        for request_data in long_requests:
            start_time = time.time()
            response_data, headers = await api_optimizer.process_request(request_data)
            uncompressed_time = (time.time() - start_time) * 1000
            uncompressed_times.append(uncompressed_time)
        
        # Re-enable compression
        api_optimizer.config.enable_compression = True
        
        # Compression should not significantly slow down processing
        avg_compressed = statistics.mean(compressed_times)
        avg_uncompressed = statistics.mean(uncompressed_times)
        
        # Allow some overhead for compression
        assert avg_compressed < avg_uncompressed * 1.2, f"Compression overhead too high: {avg_compressed:.1f}ms vs {avg_uncompressed:.1f}ms"
    
    @pytest.mark.asyncio
    async def test_caching_performance_impact(self, api_optimizer, sample_requests):
        """Test caching impact on response performance."""
        test_request = sample_requests["medium_texts"][0]
        
        # First request (cache miss)
        start_time = time.time()
        response1, headers1 = await api_optimizer.process_request(test_request)
        first_response_time = (time.time() - start_time) * 1000
        
        # Second request (cache hit)
        start_time = time.time()
        response2, headers2 = await api_optimizer.process_request(test_request)
        second_response_time = (time.time() - start_time) * 1000
        
        # Cache hit should be significantly faster
        speedup = first_response_time / second_response_time
        assert speedup >= 2.0, f"Cache not providing expected speedup: {speedup:.2f}x"
        
        # Cache hit should be very fast
        assert second_response_time < 100, f"Cache hit too slow: {second_response_time:.1f}ms"
        
        # Verify response consistency
        assert response1["is_ai_generated"] == response2["is_ai_generated"]
        assert response2["from_cache"] is True
    
    @pytest.mark.asyncio
    async def test_batch_processing_performance(self, api_optimizer, sample_requests):
        """Test batch processing performance improvements."""
        batch_texts = [req["text"] for req in sample_requests["short_texts"]]
        batch_requests = [{"text": text, "performance_mode": "fast"} for text in batch_texts]
        
        # Individual processing
        individual_start = time.time()
        individual_results = []
        for request in batch_requests:
            response, headers = await api_optimizer.process_request(request)
            individual_results.append(response)
        individual_time = (time.time() - individual_start) * 1000
        
        # Batch processing
        batch_start = time.time()
        batch_results = await api_optimizer.process_batch_requests(batch_requests)
        batch_time = (time.time() - batch_start) * 1000
        
        # Batch should be faster or similar
        speedup = individual_time / batch_time
        assert speedup >= 0.8, f"Batch processing not efficient: {speedup:.2f}x"
        
        # Batch processing should still be fast
        assert batch_time < 2000, f"Batch processing too slow: {batch_time:.1f}ms"
        
        # Results should be consistent
        assert len(batch_results) == len(individual_results)
    
    @pytest.mark.asyncio
    async def test_concurrent_request_performance(self, api_optimizer, sample_requests):
        """Test performance under concurrent load."""
        # Create concurrent requests
        concurrent_requests = []
        for _ in range(10):
            concurrent_requests.extend(sample_requests["short_texts"])
        
        # Sequential processing
        sequential_start = time.time()
        sequential_results = []
        for request in concurrent_requests:
            response, headers = await api_optimizer.process_request(request)
            sequential_results.append(response)
        sequential_time = (time.time() - sequential_start) * 1000
        
        # Concurrent processing
        concurrent_start = time.time()
        tasks = [api_optimizer.process_request(request) for request in concurrent_requests]
        concurrent_responses = await asyncio.gather(*tasks)
        concurrent_time = (time.time() - concurrent_start) * 1000
        
        # Concurrent should be significantly faster
        speedup = sequential_time / concurrent_time
        assert speedup >= 2.0, f"Concurrent processing not fast enough: {speedup:.2f}x"
        
        # All requests should still be sub-2s
        individual_times = [
            float(response[1].get("X-Response-Time", "0ms").replace("ms", ""))
            for response in concurrent_responses
        ]
        
        sub_2s_count = sum(1 for t in individual_times if t < 2000)
        sub_2s_rate = sub_2s_count / len(individual_times)
        assert sub_2s_rate >= 0.9, f"Concurrent load degraded performance: {sub_2s_rate:.1%} sub-2s"
    
    @pytest.mark.asyncio
    async def test_timeout_protection(self, api_optimizer):
        """Test timeout protection works correctly."""
        # Create request that might take longer
        slow_request = {
            "text": "A" * 10000,  # Very long text
            "performance_mode": "accurate"
        }
        
        start_time = time.time()
        response, headers = await api_optimizer.process_request(slow_request)
        response_time = (time.time() - start_time) * 1000
        
        # Should respect timeout
        assert response_time < 2000, f"Request not properly timed out: {response_time:.1f}ms"
        
        # Should have valid response even if timed out
        assert "is_ai_generated" in response
        assert "confidence_score" in response
    
    @pytest.mark.asyncio 
    async def test_error_response_performance(self, api_optimizer):
        """Test error responses are still fast."""
        # Invalid requests
        error_requests = [
            {"text": ""},  # Empty text
            {"text": None},  # Invalid text type
            {"invalid_field": "test"}  # Missing required fields
        ]
        
        error_times = []
        
        for request in error_requests:
            start_time = time.time()
            try:
                response, headers = await api_optimizer.process_request(request)
            except Exception:
                pass  # Expected errors
            
            error_time = (time.time() - start_time) * 1000
            error_times.append(error_time)
        
        # Error responses should be very fast
        avg_error_time = statistics.mean(error_times)
        assert avg_error_time < 100, f"Error responses too slow: {avg_error_time:.1f}ms"
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self, api_optimizer, sample_requests):
        """Test performance under sustained load."""
        # Create sustained load
        load_requests = []
        for _ in range(100):  # 100 requests
            load_requests.extend(sample_requests["short_texts"][:2])
        
        response_times = []
        start_load_time = time.time()
        
        # Process requests with some concurrency
        batch_size = 10
        for i in range(0, len(load_requests), batch_size):
            batch = load_requests[i:i + batch_size]
            
            batch_start = time.time()
            tasks = [api_optimizer.process_request(request) for request in batch]
            batch_responses = await asyncio.gather(*tasks)
            batch_time = (time.time() - batch_start) * 1000
            
            # Track individual response times
            for _ in batch_responses:
                response_times.append(batch_time / len(batch))
        
        total_load_time = (time.time() - start_load_time) * 1000
        
        # Check performance didn't degrade significantly
        first_quarter = response_times[:len(response_times)//4]
        last_quarter = response_times[-len(response_times)//4:]
        
        avg_first = statistics.mean(first_quarter)
        avg_last = statistics.mean(last_quarter)
        
        degradation = avg_last / avg_first
        assert degradation < 2.0, f"Performance degraded too much: {degradation:.2f}x"
        
        # Overall performance should still be good
        overall_avg = statistics.mean(response_times)
        assert overall_avg < 1000, f"Average performance under load: {overall_avg:.1f}ms"
        
        # Throughput should be reasonable
        throughput = len(load_requests) / (total_load_time / 1000)
        assert throughput >= 50, f"Throughput too low: {throughput:.1f} req/s"
    
    @pytest.mark.asyncio
    async def test_memory_efficiency_under_load(self, api_optimizer, sample_requests):
        """Test memory efficiency during sustained processing."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process many requests
        load_requests = sample_requests["short_texts"] * 50  # 250 requests
        
        for request in load_requests:
            response, headers = await api_optimizer.process_request(request)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 100, f"Memory usage increased by {memory_increase:.1f}MB"
    
    @pytest.mark.asyncio
    async def test_response_size_optimization(self, api_optimizer, sample_requests):
        """Test response size optimization."""
        response_sizes = []
        
        for request_list in sample_requests.values():
            for request in request_list:
                response, headers = await api_optimizer.process_request(request)
                
                # Calculate response size
                import json
                response_json = json.dumps(response)
                response_size = len(response_json.encode('utf-8'))
                response_sizes.append(response_size)
        
        # Responses should be reasonably sized
        avg_size = statistics.mean(response_sizes)
        max_size = max(response_sizes)
        
        assert avg_size < 2048, f"Average response size too large: {avg_size} bytes"
        assert max_size < 5120, f"Maximum response size too large: {max_size} bytes"
    
    def test_performance_configuration_validation(self):
        """Test performance configuration validation."""
        # Valid configuration
        valid_config = APIPerformanceConfig(
            max_concurrent_requests=100,
            request_timeout_seconds=1.8,
            enable_compression=True
        )
        optimizer = APIPerformanceOptimizer(valid_config)
        assert optimizer.config.max_concurrent_requests == 100
        
        # Invalid timeout (too high)
        with pytest.raises(ValueError):
            APIPerformanceConfig(request_timeout_seconds=5.0)  # Should be < 2.0
        
        # Invalid concurrent requests (too high)
        with pytest.raises(ValueError):
            APIPerformanceConfig(max_concurrent_requests=1000)  # Too many
    
    @pytest.mark.asyncio
    async def test_health_check_performance(self, api_optimizer):
        """Test health check endpoint performance."""
        start_time = time.time()
        health_result = await api_optimizer.health_check()
        health_time = (time.time() - start_time) * 1000
        
        # Health check should be very fast
        assert health_time < 100, f"Health check too slow: {health_time:.1f}ms"
        
        # Should return valid health status
        assert health_result["status"] in ["healthy", "unhealthy"]
        assert "response_time_ms" in health_result
        assert health_result["sub_2s_compliant"] is True