"""
API performance optimizer for achieving sub-2s response times.

Implements connection pooling, request batching, response compression,
async processing, and caching to optimize API performance.
"""

import asyncio
import time
import gzip
import json
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import threading

from src.core.monitoring import get_logger, get_metrics_collector
from src.core.cache import get_cache_manager


@dataclass
class APIPerformanceConfig:
    """Configuration for API performance optimization."""
    max_concurrent_requests: int = 100
    request_timeout_seconds: float = 1.8  # Leave buffer for 2s total
    enable_compression: bool = True
    compression_threshold: int = 1024  # Compress responses > 1KB
    enable_request_batching: bool = True
    batch_size: int = 10
    batch_timeout_ms: int = 50
    enable_response_caching: bool = True
    cache_ttl_seconds: int = 300
    connection_pool_size: int = 20


class ConnectionPool:
    """Connection pool for external API calls."""
    
    def __init__(self, pool_size: int = 20):
        self.pool_size = pool_size
        self.semaphore = asyncio.Semaphore(pool_size)
        self.active_connections = 0
        self._lock = threading.Lock()
        self.logger = get_logger(__name__)
    
    @asynccontextmanager
    async def get_connection(self):
        """Get a connection from the pool."""
        async with self.semaphore:
            with self._lock:
                self.active_connections += 1
            
            try:
                yield
            finally:
                with self._lock:
                    self.active_connections -= 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        with self._lock:
            return {
                "pool_size": self.pool_size,
                "active_connections": self.active_connections,
                "available_connections": self.pool_size - self.active_connections,
                "utilization": self.active_connections / self.pool_size
            }


class RequestBatcher:
    """Batches multiple requests for efficient processing."""
    
    def __init__(self, batch_size: int = 10, timeout_ms: int = 50):
        self.batch_size = batch_size
        self.timeout_ms = timeout_ms
        self.pending_requests = []
        self.batch_futures = []
        self._lock = asyncio.Lock()
        self.logger = get_logger(__name__)
        self.metrics = get_metrics_collector()
        
        # Start batch processor
        self._batch_task = asyncio.create_task(self._process_batches())
    
    async def add_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add request to batch and return result."""
        future = asyncio.Future()
        
        async with self._lock:
            self.pending_requests.append({
                "data": request_data,
                "future": future,
                "timestamp": time.time()
            })
        
        # Wait for result
        return await future
    
    async def _process_batches(self):
        """Process batches continuously."""
        while True:
            try:
                await asyncio.sleep(self.timeout_ms / 1000)
                await self._process_pending_batch()
            except Exception as e:
                self.logger.error(f"Batch processing error: {e}")
    
    async def _process_pending_batch(self):
        """Process current pending requests as a batch."""
        async with self._lock:
            if not self.pending_requests:
                return
            
            # Take up to batch_size requests
            batch = self.pending_requests[:self.batch_size]
            self.pending_requests = self.pending_requests[self.batch_size:]
        
        if batch:
            await self._execute_batch(batch)
    
    async def _execute_batch(self, batch: List[Dict[str, Any]]):
        """Execute a batch of requests."""
        start_time = time.time()
        
        try:
            # Extract request data
            requests_data = [item["data"] for item in batch]
            
            # Process batch (this would integrate with actual detection engine)
            results = await self._mock_batch_process(requests_data)
            
            # Set results for all futures
            for item, result in zip(batch, results):
                if not item["future"].done():
                    item["future"].set_result(result)
            
            # Record metrics
            batch_time = (time.time() - start_time) * 1000
            self.metrics.observe_histogram("api_batch_processing_ms", batch_time)
            self.metrics.observe_gauge("api_batch_size", len(batch))
            
        except Exception as e:
            # Set exception for all futures
            for item in batch:
                if not item["future"].done():
                    item["future"].set_exception(e)
    
    async def _mock_batch_process(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Mock batch processing (would integrate with actual detection engine)."""
        # Simulate batch processing
        await asyncio.sleep(0.01)  # 10ms processing time
        
        results = []
        for request in requests:
            # Mock result based on text length
            text = request.get("text", "")
            results.append({
                "is_ai_generated": len(text) > 100,
                "confidence_score": 0.8,
                "processing_time_ms": 10,
                "method_used": "batch_optimized"
            })
        
        return results


class ResponseCompressor:
    """Compresses API responses for faster transmission."""
    
    def __init__(self, threshold: int = 1024):
        self.threshold = threshold
        self.logger = get_logger(__name__)
        self.metrics = get_metrics_collector()
    
    def compress_response(self, data: Union[Dict, List, str]) -> tuple[bytes, Dict[str, str]]:
        """Compress response data if it exceeds threshold."""
        # Convert to JSON string
        if isinstance(data, (dict, list)):
            json_str = json.dumps(data, separators=(',', ':'))  # Compact JSON
        else:
            json_str = str(data)
        
        json_bytes = json_str.encode('utf-8')
        original_size = len(json_bytes)
        
        headers = {"Content-Type": "application/json"}
        
        # Compress if above threshold
        if original_size > self.threshold:
            compressed_data = gzip.compress(json_bytes)
            compression_ratio = len(compressed_data) / original_size
            
            # Only use compression if it provides significant benefit
            if compression_ratio < 0.8:
                headers["Content-Encoding"] = "gzip"
                headers["Content-Length"] = str(len(compressed_data))
                
                # Record compression metrics
                self.metrics.observe_histogram("api_compression_ratio", compression_ratio)
                self.metrics.increment_counter("api_responses_compressed_total")
                
                return compressed_data, headers
        
        # Return uncompressed
        headers["Content-Length"] = str(original_size)
        return json_bytes, headers


class APIPerformanceOptimizer:
    """Main API performance optimization coordinator."""
    
    def __init__(self, config: Optional[APIPerformanceConfig] = None):
        self.config = config or APIPerformanceConfig()
        self.logger = get_logger(__name__)
        self.metrics = get_metrics_collector()
        self.cache = get_cache_manager()
        
        # Initialize components
        self.connection_pool = ConnectionPool(self.config.connection_pool_size)
        self.request_batcher = RequestBatcher(
            self.config.batch_size, 
            self.config.batch_timeout_ms
        ) if self.config.enable_request_batching else None
        self.compressor = ResponseCompressor(
            self.config.compression_threshold
        ) if self.config.enable_compression else None
        
        # Request concurrency control
        self.request_semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        # Performance tracking
        self.request_times = []
        self.response_sizes = []
        self._stats_lock = threading.Lock()
    
    async def process_request(self, request_data: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, str]]:
        """Process API request with all optimizations."""
        start_time = time.time()
        request_id = request_data.get("request_id", f"req_{int(time.time() * 1000)}")
        
        try:
            # Check cache first
            if self.config.enable_response_caching:
                cached_response = await self._check_cache(request_data)
                if cached_response:
                    response_data = cached_response
                    response_data["from_cache"] = True
                else:
                    # Process request
                    async with self.request_semaphore:
                        response_data = await self._process_with_timeout(request_data)
                    
                    # Cache successful response
                    if response_data.get("confidence_score", 0) >= 0.7:
                        await self._cache_response(request_data, response_data)
            else:
                # Process without caching
                async with self.request_semaphore:
                    response_data = await self._process_with_timeout(request_data)
            
            # Add performance metadata
            processing_time = (time.time() - start_time) * 1000
            response_data.update({
                "api_processing_time_ms": processing_time,
                "request_id": request_id,
                "optimizations_applied": self._get_applied_optimizations()
            })
            
            # Compress response if enabled
            if self.compressor:
                response_bytes, headers = self.compressor.compress_response(response_data)
            else:
                response_bytes = json.dumps(response_data).encode('utf-8')
                headers = {"Content-Type": "application/json"}
            
            # Record performance metrics
            self._record_request_metrics(processing_time, len(response_bytes))
            
            return response_data, headers
            
        except Exception as e:
            self.logger.error(f"Request processing failed: {e}")
            error_response = {
                "error": "Internal server error",
                "request_id": request_id,
                "api_processing_time_ms": (time.time() - start_time) * 1000
            }
            return error_response, {"Content-Type": "application/json"}
    
    async def _process_with_timeout(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process request with timeout protection."""
        try:
            if self.request_batcher and self._should_batch_request(request_data):
                # Use batching for eligible requests
                result = await asyncio.wait_for(
                    self.request_batcher.add_request(request_data),
                    timeout=self.config.request_timeout_seconds
                )
            else:
                # Process individually
                result = await asyncio.wait_for(
                    self._individual_process(request_data),
                    timeout=self.config.request_timeout_seconds
                )
            
            return result
            
        except asyncio.TimeoutError:
            self.metrics.increment_counter("api_timeouts_total")
            return {
                "error": "Request timeout",
                "timeout_ms": self.config.request_timeout_seconds * 1000,
                "is_ai_generated": None,
                "confidence_score": 0.0
            }
    
    def _should_batch_request(self, request_data: Dict[str, Any]) -> bool:
        """Determine if request should be batched."""
        # Batch shorter texts that can benefit from batching
        text = request_data.get("text", "")
        return len(text) < 500 and not request_data.get("priority") == "high"
    
    async def _individual_process(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process individual request."""
        # This would integrate with the optimized detector
        async with self.connection_pool.get_connection():
            # Simulate processing
            text = request_data.get("text", "")
            
            # Simple processing based on text characteristics
            processing_time = min(len(text) * 0.1, 50)  # Max 50ms
            await asyncio.sleep(processing_time / 1000)
            
            return {
                "is_ai_generated": len(text) > 200,
                "confidence_score": 0.8,
                "processing_time_ms": processing_time,
                "method_used": "individual_optimized"
            }
    
    async def _check_cache(self, request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check cache for previous response."""
        cache_key = self._generate_cache_key(request_data)
        cached_response = await self.cache.get(cache_key)
        
        if cached_response:
            self.metrics.increment_counter("api_cache_hits_total")
            return cached_response
        
        return None
    
    async def _cache_response(self, request_data: Dict[str, Any], response_data: Dict[str, Any]):
        """Cache successful response."""
        cache_key = self._generate_cache_key(request_data)
        
        # Remove non-cacheable fields
        cacheable_response = {
            k: v for k, v in response_data.items() 
            if k not in ["api_processing_time_ms", "request_id", "processing_time_ms"]
        }
        
        await self.cache.set(cache_key, cacheable_response, ttl=self.config.cache_ttl_seconds)
    
    def _generate_cache_key(self, request_data: Dict[str, Any]) -> str:
        """Generate cache key for request."""
        import hashlib
        
        # Create key based on text and options
        text = request_data.get("text", "")
        options = request_data.get("options", {})
        
        key_data = f"{text}:{json.dumps(options, sort_keys=True)}"
        key_hash = hashlib.md5(key_data.encode('utf-8')).hexdigest()[:16]
        
        return f"api_response:{key_hash}"
    
    def _get_applied_optimizations(self) -> List[str]:
        """Get list of applied optimizations."""
        optimizations = []
        
        if self.config.enable_response_caching:
            optimizations.append("response_caching")
        if self.config.enable_request_batching:
            optimizations.append("request_batching")
        if self.config.enable_compression:
            optimizations.append("response_compression")
        
        optimizations.extend([
            "connection_pooling",
            "timeout_protection",
            "concurrent_processing"
        ])
        
        return optimizations
    
    def _record_request_metrics(self, processing_time: float, response_size: int):
        """Record request performance metrics."""
        self.metrics.observe_histogram("api_request_duration_ms", processing_time)
        self.metrics.observe_histogram("api_response_size_bytes", response_size)
        self.metrics.increment_counter("api_requests_total")
        
        # Track sub-2s performance
        if processing_time < 2000:
            self.metrics.increment_counter("api_sub_2s_requests_total")
        if processing_time < 1000:
            self.metrics.increment_counter("api_sub_1s_requests_total")
        if processing_time < 500:
            self.metrics.increment_counter("api_sub_500ms_requests_total")
        
        # Store for statistics
        with self._stats_lock:
            self.request_times.append(processing_time)
            self.response_sizes.append(response_size)
            
            # Keep only recent data
            if len(self.request_times) > 1000:
                self.request_times = self.request_times[-1000:]
                self.response_sizes = self.response_sizes[-1000:]
    
    async def process_batch_requests(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple requests efficiently."""
        start_time = time.time()
        
        # Split into groups for optimal processing
        individual_requests = []
        batch_requests = []
        
        for request in requests:
            if self._should_batch_request(request):
                batch_requests.append(request)
            else:
                individual_requests.append(request)
        
        # Process groups concurrently
        tasks = []
        
        # Individual requests
        for request in individual_requests:
            task = asyncio.create_task(self.process_request(request))
            tasks.append(task)
        
        # Batch requests
        if batch_requests and self.request_batcher:
            batch_task = asyncio.create_task(self._process_batch_group(batch_requests))
            tasks.append(batch_task)
        
        # Wait for all results
        results = []
        if tasks:
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Flatten results
            for result in batch_results:
                if isinstance(result, list):
                    results.extend(result)
                elif not isinstance(result, Exception):
                    results.append(result[0])  # Extract response data from tuple
        
        # Record batch metrics
        batch_time = (time.time() - start_time) * 1000
        self.metrics.observe_histogram("api_batch_request_duration_ms", batch_time)
        self.metrics.observe_gauge("api_batch_request_size", len(requests))
        
        return results
    
    async def _process_batch_group(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a group of batchable requests."""
        results = []
        for request in requests:
            result, _ = await self.process_request(request)
            results.append(result)
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        with self._stats_lock:
            request_times = self.request_times.copy()
            response_sizes = self.response_sizes.copy()
        
        stats = {
            "request_performance": self._calculate_request_stats(request_times),
            "response_performance": self._calculate_response_stats(response_sizes),
            "connection_pool": self.connection_pool.get_stats(),
            "optimization_config": {
                "max_concurrent_requests": self.config.max_concurrent_requests,
                "request_timeout_seconds": self.config.request_timeout_seconds,
                "enable_compression": self.config.enable_compression,
                "enable_batching": self.config.enable_request_batching,
                "enable_caching": self.config.enable_response_caching
            },
            "performance_targets": {
                "sub_2s_rate": self._calculate_sub_target_rate(request_times, 2000),
                "sub_1s_rate": self._calculate_sub_target_rate(request_times, 1000),
                "sub_500ms_rate": self._calculate_sub_target_rate(request_times, 500)
            }
        }
        
        return stats
    
    def _calculate_request_stats(self, times: List[float]) -> Dict[str, float]:
        """Calculate request timing statistics."""
        if not times:
            return {"count": 0}
        
        import statistics
        
        return {
            "count": len(times),
            "mean_ms": statistics.mean(times),
            "median_ms": statistics.median(times),
            "p95_ms": self._percentile(times, 95),
            "p99_ms": self._percentile(times, 99),
            "min_ms": min(times),
            "max_ms": max(times),
            "std_dev_ms": statistics.stdev(times) if len(times) > 1 else 0
        }
    
    def _calculate_response_stats(self, sizes: List[int]) -> Dict[str, float]:
        """Calculate response size statistics."""
        if not sizes:
            return {"count": 0}
        
        import statistics
        
        return {
            "count": len(sizes),
            "mean_bytes": statistics.mean(sizes),
            "median_bytes": statistics.median(sizes),
            "min_bytes": min(sizes),
            "max_bytes": max(sizes)
        }
    
    def _calculate_sub_target_rate(self, times: List[float], target_ms: float) -> float:
        """Calculate rate of requests under target time."""
        if not times:
            return 0.0
        
        sub_target_count = sum(1 for t in times if t < target_ms)
        return sub_target_count / len(times)
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        
        import statistics
        return statistics.quantiles(sorted(data), n=100)[int(percentile) - 1]
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform API health check."""
        start_time = time.time()
        
        # Test request processing
        test_request = {
            "text": "This is a test request for health checking.",
            "request_id": "health_check"
        }
        
        try:
            result, headers = await self.process_request(test_request)
            health_time = (time.time() - start_time) * 1000
            
            return {
                "status": "healthy",
                "response_time_ms": health_time,
                "sub_2s_compliant": health_time < 2000,
                "optimizations_active": self._get_applied_optimizations(),
                "connection_pool_healthy": self.connection_pool.get_stats()["utilization"] < 0.9
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "response_time_ms": (time.time() - start_time) * 1000
            }