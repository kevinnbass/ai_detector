"""
Unit Tests for API Client System
Comprehensive tests for unified API client and related components
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, call
from datetime import datetime, timedelta
import json
import time

from src.core.api_client.unified_client import UnifiedAPIClient
from src.core.api_client.queue_manager import QueueManager, QueuedRequest, Priority
from src.core.api_client.retry_handler import RetryHandler, RetryStrategy
from src.core.api_client.rate_limiter import RateLimiter, TokenBucket, SlidingWindow
from src.core.api_client.response_cache import ResponseCache, CacheEntry, CacheStrategy
from src.core.api_client.middleware import (
    MiddlewarePipeline, AuthMiddleware, LoggingMiddleware, 
    TimingMiddleware, ValidationMiddleware
)
from src.core.interfaces.api_interfaces import HTTPMethod, APIResponse


class TestQueuedRequest:
    """Test suite for QueuedRequest"""
    
    @pytest.mark.unit
    def test_queued_request_creation(self):
        """Test QueuedRequest creation and properties"""
        request = QueuedRequest(
            id="test_req_1",
            method=HTTPMethod.GET,
            endpoint="/api/test",
            priority=Priority.HIGH,
            data={"key": "value"}
        )
        
        assert request.id == "test_req_1"
        assert request.method == HTTPMethod.GET
        assert request.endpoint == "/api/test"
        assert request.priority == Priority.HIGH
        assert request.data["key"] == "value"
        assert isinstance(request.created_at, datetime)
    
    @pytest.mark.unit
    def test_queued_request_comparison(self):
        """Test QueuedRequest priority comparison"""
        high_req = QueuedRequest("1", HTTPMethod.GET, "/test", Priority.HIGH)
        normal_req = QueuedRequest("2", HTTPMethod.GET, "/test", Priority.NORMAL)
        low_req = QueuedRequest("3", HTTPMethod.GET, "/test", Priority.LOW)
        
        # Higher priority should be "less than" for min-heap
        assert high_req < normal_req
        assert normal_req < low_req
        assert high_req < low_req
    
    @pytest.mark.unit
    def test_queued_request_aging(self):
        """Test QueuedRequest age calculation"""
        past_time = datetime.now() - timedelta(seconds=30)
        request = QueuedRequest(
            "aged", HTTPMethod.GET, "/test", Priority.NORMAL,
            created_at=past_time
        )
        
        age = request.get_age()
        assert isinstance(age, timedelta)
        assert age.total_seconds() >= 30


class TestQueueManager:
    """Test suite for QueueManager"""
    
    @pytest.fixture
    def queue_manager(self):
        """Create QueueManager instance"""
        return QueueManager(max_size=100)
    
    @pytest.mark.unit
    async def test_queue_manager_basic_operations(self, queue_manager):
        """Test basic queue operations"""
        await queue_manager.initialize()
        
        # Test queue is empty initially
        assert queue_manager.size() == 0
        assert queue_manager.is_empty()
        
        # Add request
        request = QueuedRequest("test_1", HTTPMethod.GET, "/test", Priority.NORMAL)
        await queue_manager.enqueue(request)
        
        assert queue_manager.size() == 1
        assert not queue_manager.is_empty()
        
        # Get request
        retrieved = await queue_manager.dequeue()
        assert retrieved.id == "test_1"
        assert queue_manager.size() == 0
    
    @pytest.mark.unit
    async def test_queue_manager_priority_ordering(self, queue_manager):
        """Test priority-based queue ordering"""
        await queue_manager.initialize()
        
        # Add requests with different priorities
        requests = [
            QueuedRequest("low", HTTPMethod.GET, "/test", Priority.LOW),
            QueuedRequest("high", HTTPMethod.GET, "/test", Priority.HIGH),
            QueuedRequest("normal", HTTPMethod.GET, "/test", Priority.NORMAL)
        ]
        
        for req in requests:
            await queue_manager.enqueue(req)
        
        # Should dequeue in priority order: HIGH, NORMAL, LOW
        first = await queue_manager.dequeue()
        assert first.id == "high"
        
        second = await queue_manager.dequeue()
        assert second.id == "normal"
        
        third = await queue_manager.dequeue()
        assert third.id == "low"
    
    @pytest.mark.unit
    async def test_queue_manager_capacity(self, queue_manager):
        """Test queue capacity limits"""
        await queue_manager.initialize()
        
        # Fill queue to capacity
        for i in range(100):
            request = QueuedRequest(f"req_{i}", HTTPMethod.GET, "/test", Priority.NORMAL)
            await queue_manager.enqueue(request)
        
        assert queue_manager.size() == 100
        assert queue_manager.is_full()
        
        # Adding another should raise exception
        overflow_request = QueuedRequest("overflow", HTTPMethod.GET, "/test", Priority.NORMAL)
        with pytest.raises(RuntimeError, match="Queue is full"):
            await queue_manager.enqueue(overflow_request)
    
    @pytest.mark.unit
    async def test_queue_manager_timeout(self, queue_manager):
        """Test queue timeout handling"""
        await queue_manager.initialize()
        
        # Try to dequeue from empty queue with timeout
        start_time = time.time()
        result = await queue_manager.dequeue(timeout=0.1)
        end_time = time.time()
        
        assert result is None
        assert (end_time - start_time) >= 0.1


class TestRetryHandler:
    """Test suite for RetryHandler"""
    
    @pytest.fixture
    def retry_handler(self):
        """Create RetryHandler instance"""
        return RetryHandler(max_retries=3, base_delay=0.1)
    
    @pytest.mark.unit
    async def test_retry_handler_success(self, retry_handler):
        """Test retry handler with successful operation"""
        call_count = 0
        
        async def successful_operation():
            nonlocal call_count
            call_count += 1
            return {"success": True, "data": "test"}
        
        result = await retry_handler.execute_with_retry(successful_operation)
        
        assert result["success"] is True
        assert call_count == 1  # Should succeed on first try
    
    @pytest.mark.unit
    async def test_retry_handler_eventual_success(self, retry_handler):
        """Test retry handler with eventual success"""
        call_count = 0
        
        async def eventually_successful():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return {"success": True, "attempt": call_count}
        
        result = await retry_handler.execute_with_retry(eventually_successful)
        
        assert result["success"] is True
        assert result["attempt"] == 3
        assert call_count == 3
    
    @pytest.mark.unit
    async def test_retry_handler_max_retries(self, retry_handler):
        """Test retry handler exceeding max retries"""
        call_count = 0
        
        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise Exception(f"Failure {call_count}")
        
        with pytest.raises(Exception, match="Failure 4"):  # 3 retries + 1 initial = 4 calls
            await retry_handler.execute_with_retry(always_fails)
        
        assert call_count == 4
    
    @pytest.mark.unit
    async def test_retry_strategies(self):
        """Test different retry strategies"""
        # Fixed delay strategy
        fixed_handler = RetryHandler(
            max_retries=2,
            base_delay=0.1,
            strategy=RetryStrategy.FIXED
        )
        
        delays = []
        for i in range(3):
            delay = fixed_handler._calculate_delay(i)
            delays.append(delay)
        
        assert all(d == 0.1 for d in delays)
        
        # Exponential backoff strategy
        exponential_handler = RetryHandler(
            max_retries=3,
            base_delay=0.1,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF
        )
        
        delays = []
        for i in range(4):
            delay = exponential_handler._calculate_delay(i)
            delays.append(delay)
        
        # Should increase exponentially
        assert delays[1] > delays[0]
        assert delays[2] > delays[1]
        assert delays[3] > delays[2]


class TestRateLimiter:
    """Test suite for rate limiting components"""
    
    @pytest.mark.unit
    def test_token_bucket_initialization(self):
        """Test TokenBucket initialization"""
        bucket = TokenBucket(capacity=10, refill_rate=5.0)
        
        assert bucket.capacity == 10
        assert bucket.refill_rate == 5.0
        assert bucket.tokens == 10  # Should start full
    
    @pytest.mark.unit
    def test_token_bucket_consumption(self):
        """Test TokenBucket token consumption"""
        bucket = TokenBucket(capacity=5, refill_rate=1.0)
        
        # Should be able to consume available tokens
        assert bucket.try_consume(3) is True
        assert bucket.tokens == 2
        
        # Should fail if not enough tokens
        assert bucket.try_consume(3) is False
        assert bucket.tokens == 2  # Unchanged
        
        # Should succeed with remaining tokens
        assert bucket.try_consume(2) is True
        assert bucket.tokens == 0
    
    @pytest.mark.unit
    def test_token_bucket_refill(self):
        """Test TokenBucket token refill"""
        bucket = TokenBucket(capacity=5, refill_rate=10.0)  # High rate for testing
        
        # Consume all tokens
        bucket.try_consume(5)
        assert bucket.tokens == 0
        
        # Wait and refill
        time.sleep(0.1)
        bucket._refill()
        
        # Should have refilled some tokens
        assert bucket.tokens > 0
        assert bucket.tokens <= 5  # But not exceed capacity
    
    @pytest.mark.unit
    def test_sliding_window_rate_limiter(self):
        """Test SlidingWindow rate limiter"""
        # Allow 3 requests per second
        limiter = SlidingWindow(max_requests=3, window_seconds=1.0)
        
        # First 3 requests should succeed
        assert limiter.is_allowed() is True
        assert limiter.is_allowed() is True
        assert limiter.is_allowed() is True
        
        # 4th request should fail
        assert limiter.is_allowed() is False
        
        # After window passes, should work again
        time.sleep(1.1)
        assert limiter.is_allowed() is True
    
    @pytest.mark.unit
    async def test_rate_limiter_integration(self):
        """Test RateLimiter with different strategies"""
        # Token bucket limiter
        bucket_limiter = RateLimiter(
            strategy="token_bucket",
            requests_per_second=2,
            burst_capacity=5
        )
        
        # Should allow initial burst
        for _ in range(5):
            assert await bucket_limiter.is_allowed() is True
        
        # Should be rate limited after burst
        assert await bucket_limiter.is_allowed() is False
        
        # Sliding window limiter
        window_limiter = RateLimiter(
            strategy="sliding_window",
            requests_per_second=1,
            window_seconds=1.0
        )
        
        assert await window_limiter.is_allowed() is True
        assert await window_limiter.is_allowed() is False  # Over limit


class TestResponseCache:
    """Test suite for ResponseCache"""
    
    @pytest.fixture
    def cache(self):
        """Create ResponseCache instance"""
        return ResponseCache(max_size=100, default_ttl=300)
    
    @pytest.mark.unit
    def test_cache_basic_operations(self, cache):
        """Test basic cache operations"""
        # Cache should be empty initially
        assert cache.size() == 0
        
        # Store entry
        response = APIResponse(
            status_code=200,
            headers={"Content-Type": "application/json"},
            body={"data": "test"}
        )
        
        cache.set("test_key", response)
        assert cache.size() == 1
        
        # Retrieve entry
        cached = cache.get("test_key")
        assert cached is not None
        assert cached.body["data"] == "test"
        
        # Check existence
        assert cache.has("test_key") is True
        assert cache.has("nonexistent") is False
    
    @pytest.mark.unit
    def test_cache_ttl_expiration(self, cache):
        """Test cache TTL expiration"""
        response = APIResponse(status_code=200, body={"data": "test"})
        
        # Store with short TTL
        cache.set("expiring_key", response, ttl=0.1)
        
        # Should be available immediately
        assert cache.get("expiring_key") is not None
        
        # Should be expired after TTL
        time.sleep(0.2)
        assert cache.get("expiring_key") is None
        assert cache.size() == 0  # Should be cleaned up
    
    @pytest.mark.unit
    def test_cache_strategies(self):
        """Test different cache eviction strategies"""
        # LRU cache
        lru_cache = ResponseCache(max_size=3, strategy=CacheStrategy.LRU)
        
        # Fill cache
        for i in range(3):
            response = APIResponse(status_code=200, body={"id": i})
            lru_cache.set(f"key_{i}", response)
        
        # Access key_0 to make it recently used
        lru_cache.get("key_0")
        
        # Add new item - should evict key_1 (least recently used)
        response = APIResponse(status_code=200, body={"id": 3})
        lru_cache.set("key_3", response)
        
        assert lru_cache.has("key_0") is True   # Recently accessed
        assert lru_cache.has("key_1") is False  # Should be evicted
        assert lru_cache.has("key_2") is True   # Still there
        assert lru_cache.has("key_3") is True   # Newly added
    
    @pytest.mark.unit
    def test_cache_statistics(self, cache):
        """Test cache statistics tracking"""
        response = APIResponse(status_code=200, body={"data": "test"})
        
        # Initial stats
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        
        # Cache miss
        cache.get("nonexistent")
        stats = cache.get_stats()
        assert stats["misses"] == 1
        
        # Cache set and hit
        cache.set("test_key", response)
        cache.get("test_key")
        stats = cache.get_stats()
        assert stats["hits"] == 1


class TestMiddleware:
    """Test suite for middleware components"""
    
    @pytest.mark.unit
    async def test_auth_middleware(self):
        """Test AuthMiddleware"""
        middleware = AuthMiddleware(api_key="test_key_123")
        
        request_data = {
            "method": HTTPMethod.GET,
            "endpoint": "/api/test",
            "headers": {}
        }
        
        # Process request
        processed = await middleware.process_request(request_data)
        
        # Should add authorization header
        assert "Authorization" in processed["headers"]
        assert processed["headers"]["Authorization"] == "Bearer test_key_123"
    
    @pytest.mark.unit
    async def test_logging_middleware(self):
        """Test LoggingMiddleware"""
        middleware = LoggingMiddleware()
        
        request_data = {
            "method": HTTPMethod.POST,
            "endpoint": "/api/data",
            "data": {"key": "value"}
        }
        
        response_data = APIResponse(
            status_code=201,
            body={"success": True}
        )
        
        # Should not raise exceptions
        await middleware.process_request(request_data)
        await middleware.process_response(response_data, request_data)
    
    @pytest.mark.unit
    async def test_timing_middleware(self):
        """Test TimingMiddleware"""
        middleware = TimingMiddleware()
        
        request_data = {
            "method": HTTPMethod.GET,
            "endpoint": "/api/test"
        }
        
        # Process request (starts timer)
        processed_request = await middleware.process_request(request_data)
        
        # Simulate some processing time
        await asyncio.sleep(0.1)
        
        response = APIResponse(status_code=200, body={})
        
        # Process response (stops timer)
        processed_response = await middleware.process_response(response, processed_request)
        
        # Should add timing information
        assert "X-Response-Time" in processed_response.headers
        response_time = float(processed_response.headers["X-Response-Time"])
        assert response_time >= 0.1
    
    @pytest.mark.unit
    async def test_validation_middleware(self):
        """Test ValidationMiddleware"""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number", "minimum": 0}
            },
            "required": ["name"]
        }
        
        middleware = ValidationMiddleware(request_schema=schema)
        
        # Valid request
        valid_request = {
            "method": HTTPMethod.POST,
            "endpoint": "/api/user",
            "data": {"name": "John", "age": 30}
        }
        
        # Should pass validation
        processed = await middleware.process_request(valid_request)
        assert processed == valid_request
        
        # Invalid request
        invalid_request = {
            "method": HTTPMethod.POST,
            "endpoint": "/api/user",
            "data": {"age": -5}  # Missing required name, invalid age
        }
        
        # Should raise validation error
        with pytest.raises(ValueError, match="validation"):
            await middleware.process_request(invalid_request)
    
    @pytest.mark.unit
    async def test_middleware_pipeline(self):
        """Test MiddlewarePipeline"""
        pipeline = MiddlewarePipeline()
        
        # Add middleware
        auth_middleware = AuthMiddleware(api_key="test_key")
        timing_middleware = TimingMiddleware()
        
        pipeline.add_middleware(auth_middleware)
        pipeline.add_middleware(timing_middleware)
        
        # Process request through pipeline
        request_data = {
            "method": HTTPMethod.GET,
            "endpoint": "/api/test",
            "headers": {}
        }
        
        processed_request = await pipeline.process_request(request_data)
        
        # Should have auth header from auth middleware
        assert "Authorization" in processed_request["headers"]
        
        # Should have timing data from timing middleware
        assert "_start_time" in processed_request
        
        # Process response through pipeline
        response = APIResponse(status_code=200, body={})
        processed_response = await pipeline.process_response(response, processed_request)
        
        # Should have timing header from timing middleware
        assert "X-Response-Time" in processed_response.headers


class TestUnifiedAPIClient:
    """Test suite for UnifiedAPIClient"""
    
    @pytest.fixture
    def mock_http_client(self):
        """Mock HTTP client for testing"""
        client = Mock()
        
        async def mock_request(method, url, **kwargs):
            return APIResponse(
                status_code=200,
                headers={"Content-Type": "application/json"},
                body={"success": True, "url": url, "method": method.value}
            )
        
        client.request = AsyncMock(side_effect=mock_request)
        return client
    
    @pytest.fixture
    def api_client(self, mock_http_client):
        """Create UnifiedAPIClient with mocked HTTP client"""
        client = UnifiedAPIClient(
            base_url="https://api.example.com",
            http_client=mock_http_client
        )
        return client
    
    @pytest.mark.unit
    async def test_api_client_initialization(self, api_client):
        """Test UnifiedAPIClient initialization"""
        assert not api_client.is_initialized()
        
        await api_client.initialize()
        assert api_client.is_initialized()
    
    @pytest.mark.unit
    async def test_api_client_basic_request(self, api_client, mock_http_client):
        """Test basic API request"""
        await api_client.initialize()
        
        response = await api_client.request(
            method=HTTPMethod.GET,
            endpoint="/test",
            params={"key": "value"}
        )
        
        assert isinstance(response, APIResponse)
        assert response.status_code == 200
        assert response.body["success"] is True
        
        # Verify HTTP client was called
        mock_http_client.request.assert_called_once()
    
    @pytest.mark.unit
    async def test_api_client_with_queue(self, api_client):
        """Test API client with request queuing"""
        await api_client.initialize()
        
        # Configure to use queue
        response = await api_client.request(
            method=HTTPMethod.POST,
            endpoint="/queued",
            data={"test": "data"},
            use_queue=True,
            priority=Priority.HIGH
        )
        
        assert response.status_code == 200
        assert response.body["success"] is True
    
    @pytest.mark.unit
    async def test_api_client_with_cache(self, api_client):
        """Test API client with response caching"""
        await api_client.initialize()
        
        # First request (cache miss)
        response1 = await api_client.request(
            method=HTTPMethod.GET,
            endpoint="/cached",
            use_cache=True
        )
        
        # Second request (cache hit)
        response2 = await api_client.request(
            method=HTTPMethod.GET,
            endpoint="/cached",
            use_cache=True
        )
        
        # Both should return the same data
        assert response1.body == response2.body
        
        # Should have cache statistics
        cache_stats = api_client.get_cache_stats()
        assert cache_stats["hits"] >= 1
    
    @pytest.mark.unit
    async def test_api_client_retry_on_failure(self, api_client, mock_http_client):
        """Test API client retry on failure"""
        await api_client.initialize()
        
        # Configure to fail first few attempts
        call_count = 0
        
        async def failing_request(method, url, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Network error")
            return APIResponse(status_code=200, body={"success": True, "attempts": call_count})
        
        mock_http_client.request = AsyncMock(side_effect=failing_request)
        
        # Should eventually succeed after retries
        response = await api_client.request(
            method=HTTPMethod.GET,
            endpoint="/retry-test"
        )
        
        assert response.status_code == 200
        assert response.body["attempts"] == 3
    
    @pytest.mark.unit
    async def test_api_client_rate_limiting(self, api_client):
        """Test API client rate limiting"""
        await api_client.initialize()
        
        # Configure strict rate limit for testing
        api_client._rate_limiter._bucket.capacity = 2
        api_client._rate_limiter._bucket.tokens = 2
        api_client._rate_limiter._bucket.refill_rate = 0.1  # Very slow refill
        
        # First two requests should succeed
        response1 = await api_client.request(HTTPMethod.GET, "/test1")
        response2 = await api_client.request(HTTPMethod.GET, "/test2")
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        # Third request should be delayed due to rate limiting
        start_time = time.time()
        response3 = await api_client.request(HTTPMethod.GET, "/test3")
        end_time = time.time()
        
        assert response3.status_code == 200
        # Should have been delayed
        assert (end_time - start_time) > 0.1
    
    @pytest.mark.unit
    async def test_api_client_middleware_integration(self, api_client):
        """Test API client with middleware"""
        await api_client.initialize()
        
        # Add custom middleware
        auth_middleware = AuthMiddleware(api_key="test_integration")
        api_client.add_middleware(auth_middleware)
        
        response = await api_client.request(
            method=HTTPMethod.GET,
            endpoint="/authenticated"
        )
        
        assert response.status_code == 200
        # Middleware should have been applied
    
    @pytest.mark.unit
    async def test_api_client_cleanup(self, api_client):
        """Test API client cleanup"""
        await api_client.initialize()
        
        # Make some requests to populate caches/queues
        await api_client.request(HTTPMethod.GET, "/test", use_cache=True)
        
        # Cleanup should not raise exceptions
        await api_client.close()
        assert not api_client.is_initialized()


@pytest.mark.integration
class TestAPIClientIntegration:
    """Integration tests for API client system"""
    
    @pytest.mark.integration
    async def test_complete_api_workflow(self):
        """Test complete API client workflow with all features"""
        # Create client with real-like configuration
        client = UnifiedAPIClient(
            base_url="https://api.test.com",
            max_retries=2,
            rate_limit_requests_per_second=5,
            cache_enabled=True,
            queue_enabled=True
        )
        
        # Mock the underlying HTTP client
        mock_http = Mock()
        mock_http.request = AsyncMock(return_value=APIResponse(
            status_code=200,
            headers={"Content-Type": "application/json"},
            body={"integration": "test", "timestamp": datetime.now().isoformat()}
        ))
        
        client._http_client = mock_http
        
        await client.initialize()
        
        try:
            # Test various request patterns
            responses = []
            
            # High priority request
            response = await client.request(
                HTTPMethod.POST,
                "/priority",
                data={"urgent": True},
                priority=Priority.HIGH,
                use_queue=True
            )
            responses.append(response)
            
            # Cached request
            response = await client.request(
                HTTPMethod.GET,
                "/cached-data",
                use_cache=True
            )
            responses.append(response)
            
            # Same cached request (should hit cache)
            response = await client.request(
                HTTPMethod.GET,
                "/cached-data",
                use_cache=True
            )
            responses.append(response)
            
            # Verify all requests succeeded
            assert all(r.status_code == 200 for r in responses)
            
            # Verify cache was used
            cache_stats = client.get_cache_stats()
            assert cache_stats["hits"] >= 1
            
            # Verify queue was used
            queue_stats = client.get_queue_stats()
            assert queue_stats["total_processed"] >= 1
            
        finally:
            await client.close()