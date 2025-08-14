"""
Connection and external service optimization for API performance.

Implements connection pooling, circuit breakers, and optimized
HTTP client configurations for sub-2s API response times.
"""

import asyncio
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import httpx
from contextlib import asynccontextmanager

from src.core.monitoring import get_logger, get_metrics_collector


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    success_threshold: int = 3


class CircuitBreaker:
    """Circuit breaker for external service calls."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.logger = get_logger(__name__)
        self.metrics = get_metrics_collector()
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.config.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                self.logger.info(f"Circuit breaker {self.name} entering HALF_OPEN state")
            else:
                self.metrics.increment_counter("circuit_breaker_rejections_total", labels={"circuit": self.name})
                raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise e
    
    async def _on_success(self):
        """Handle successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.logger.info(f"Circuit breaker {self.name} closed - service recovered")
                self.metrics.increment_counter("circuit_breaker_recoveries_total", labels={"circuit": self.name})
        else:
            self.failure_count = 0
    
    async def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            self.logger.warning(f"Circuit breaker {self.name} opened - service failing")
            self.metrics.increment_counter("circuit_breaker_opens_total", labels={"circuit": self.name})
    
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time
        }


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class OptimizedHTTPClient:
    """Optimized HTTP client for external API calls."""
    
    def __init__(self, base_url: Optional[str] = None, timeout: float = 1.5):
        self.base_url = base_url
        self.timeout = timeout
        self.logger = get_logger(__name__)
        self.metrics = get_metrics_collector()
        
        # Configure optimized HTTP client
        limits = httpx.Limits(
            max_keepalive_connections=20,
            max_connections=100,
            keepalive_expiry=30.0
        )
        
        self.client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout),
            limits=limits,
            http2=True,  # Enable HTTP/2 for better performance
            headers={
                "User-Agent": "AI-Detector-Optimized/1.0",
                "Accept": "application/json",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive"
            }
        )
        
        # Circuit breakers for different endpoints
        self.circuit_breakers = {}
    
    def add_circuit_breaker(self, endpoint: str, config: CircuitBreakerConfig):
        """Add circuit breaker for endpoint."""
        self.circuit_breakers[endpoint] = CircuitBreaker(endpoint, config)
    
    async def get(self, url: str, **kwargs) -> httpx.Response:
        """Optimized GET request."""
        return await self._request("GET", url, **kwargs)
    
    async def post(self, url: str, **kwargs) -> httpx.Response:
        """Optimized POST request."""
        return await self._request("POST", url, **kwargs)
    
    async def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Execute optimized HTTP request with circuit breaker protection."""
        start_time = time.time()
        endpoint = self._get_endpoint_name(url)
        
        try:
            # Check circuit breaker
            circuit_breaker = self.circuit_breakers.get(endpoint)
            
            if circuit_breaker:
                response = await circuit_breaker.call(
                    self._execute_request, method, url, **kwargs
                )
            else:
                response = await self._execute_request(method, url, **kwargs)
            
            # Record success metrics
            duration = (time.time() - start_time) * 1000
            self.metrics.observe_histogram("http_request_duration_ms", duration, 
                                         labels={"method": method, "endpoint": endpoint})
            self.metrics.increment_counter("http_requests_total", 
                                         labels={"method": method, "endpoint": endpoint, "status": str(response.status_code)})
            
            return response
            
        except Exception as e:
            # Record failure metrics
            duration = (time.time() - start_time) * 1000
            self.metrics.observe_histogram("http_request_duration_ms", duration,
                                         labels={"method": method, "endpoint": endpoint})
            self.metrics.increment_counter("http_request_errors_total",
                                         labels={"method": method, "endpoint": endpoint})
            
            self.logger.error(f"HTTP request failed: {method} {url} - {e}")
            raise
    
    async def _execute_request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Execute the actual HTTP request."""
        response = await self.client.request(method, url, **kwargs)
        response.raise_for_status()
        return response
    
    def _get_endpoint_name(self, url: str) -> str:
        """Extract endpoint name for metrics."""
        if self.base_url and url.startswith('/'):
            return url.split('?')[0]  # Remove query parameters
        else:
            # Extract host for external URLs
            try:
                from urllib.parse import urlparse
                parsed = urlparse(url)
                return parsed.netloc
            except:
                return "unknown"
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get HTTP client statistics."""
        return {
            "base_url": self.base_url,
            "timeout": self.timeout,
            "circuit_breakers": {
                name: cb.get_state() 
                for name, cb in self.circuit_breakers.items()
            },
            "connection_pool": {
                "max_connections": self.client._limits.max_connections,
                "max_keepalive": self.client._limits.max_keepalive_connections
            }
        }


class DatabaseConnectionPool:
    """Optimized database connection pool."""
    
    def __init__(self, connection_string: str, pool_size: int = 20, timeout: float = 1.0):
        self.connection_string = connection_string
        self.pool_size = pool_size
        self.timeout = timeout
        self.logger = get_logger(__name__)
        self.metrics = get_metrics_collector()
        
        # Connection pool
        self.pool = None
        self.active_connections = 0
        self._lock = asyncio.Lock()
    
    async def initialize(self):
        """Initialize connection pool."""
        try:
            # This would use actual database library (asyncpg, aiomysql, etc.)
            # For now, simulate pool initialization
            self.pool = MockConnectionPool(self.pool_size)
            self.logger.info(f"Database connection pool initialized with {self.pool_size} connections")
        except Exception as e:
            self.logger.error(f"Failed to initialize database pool: {e}")
            raise
    
    @asynccontextmanager
    async def get_connection(self):
        """Get database connection from pool."""
        if not self.pool:
            await self.initialize()
        
        async with self._lock:
            self.active_connections += 1
        
        start_time = time.time()
        
        try:
            # Get connection from pool
            connection = await asyncio.wait_for(
                self.pool.acquire(),
                timeout=self.timeout
            )
            
            acquire_time = (time.time() - start_time) * 1000
            self.metrics.observe_histogram("db_connection_acquire_ms", acquire_time)
            
            yield connection
            
        except asyncio.TimeoutError:
            self.metrics.increment_counter("db_connection_timeouts_total")
            raise
        finally:
            # Return connection to pool
            if 'connection' in locals():
                await self.pool.release(connection)
            
            async with self._lock:
                self.active_connections -= 1
    
    async def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Execute optimized database query."""
        start_time = time.time()
        
        async with self.get_connection() as conn:
            try:
                # Execute query (mock implementation)
                await asyncio.sleep(0.01)  # Simulate query execution
                result = [{"id": 1, "data": "mock_result"}]
                
                query_time = (time.time() - start_time) * 1000
                self.metrics.observe_histogram("db_query_duration_ms", query_time)
                self.metrics.increment_counter("db_queries_total", labels={"type": "select"})
                
                return result
                
            except Exception as e:
                self.metrics.increment_counter("db_query_errors_total")
                self.logger.error(f"Database query failed: {e}")
                raise
    
    async def close(self):
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        return {
            "pool_size": self.pool_size,
            "active_connections": self.active_connections,
            "timeout": self.timeout,
            "utilization": self.active_connections / self.pool_size if self.pool_size > 0 else 0
        }


class MockConnectionPool:
    """Mock connection pool for testing."""
    
    def __init__(self, size: int):
        self.size = size
        self.connections = asyncio.Queue(maxsize=size)
        
        # Initialize pool with mock connections
        for i in range(size):
            self.connections.put_nowait(f"connection_{i}")
    
    async def acquire(self):
        """Acquire connection from pool."""
        return await self.connections.get()
    
    async def release(self, connection):
        """Release connection back to pool."""
        await self.connections.put(connection)
    
    async def close(self):
        """Close connection pool."""
        pass


class ExternalServiceManager:
    """Manages connections to external services with optimization."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.metrics = get_metrics_collector()
        
        # Service clients
        self.llm_client = None
        self.cache_client = None
        self.db_pool = None
        
        # Circuit breaker configurations
        self.circuit_configs = {
            "llm_service": CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=30.0,
                success_threshold=2
            ),
            "cache_service": CircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout=10.0,
                success_threshold=3
            )
        }
    
    async def initialize(self):
        """Initialize all external service connections."""
        # Initialize LLM service client
        self.llm_client = OptimizedHTTPClient(
            base_url="https://api.gemini.google.com",
            timeout=1.5
        )
        self.llm_client.add_circuit_breaker(
            "llm_service", 
            self.circuit_configs["llm_service"]
        )
        
        # Initialize cache client (Redis, Memcached, etc.)
        self.cache_client = OptimizedHTTPClient(
            base_url="http://cache-service:6379",
            timeout=0.5
        )
        self.cache_client.add_circuit_breaker(
            "cache_service",
            self.circuit_configs["cache_service"]
        )
        
        # Initialize database pool
        self.db_pool = DatabaseConnectionPool(
            connection_string="postgresql://user:pass@db:5432/aidetector",
            pool_size=20,
            timeout=1.0
        )
        await self.db_pool.initialize()
        
        self.logger.info("External service connections initialized")
    
    async def call_llm_service(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Call LLM service with optimization."""
        try:
            response = await self.llm_client.post("/v1/analyze", json=payload)
            return response.json()
        except CircuitBreakerOpenError:
            # Return cached or default response when circuit is open
            return {
                "is_ai_generated": None,
                "confidence_score": 0.0,
                "error": "LLM service unavailable",
                "fallback": True
            }
    
    async def get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get result from cache service."""
        try:
            response = await self.cache_client.get(f"/get/{cache_key}")
            if response.status_code == 200:
                return response.json()
            return None
        except (CircuitBreakerOpenError, Exception):
            return None
    
    async def set_cached_result(self, cache_key: str, data: Dict[str, Any], ttl: int = 300):
        """Set result in cache service."""
        try:
            await self.cache_client.post(f"/set/{cache_key}", json={
                "data": data,
                "ttl": ttl
            })
        except (CircuitBreakerOpenError, Exception):
            # Cache failures shouldn't break the request
            pass
    
    async def query_database(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Query database with optimization."""
        return await self.db_pool.execute_query(query, params)
    
    async def close(self):
        """Close all external service connections."""
        if self.llm_client:
            await self.llm_client.close()
        if self.cache_client:
            await self.cache_client.close()
        if self.db_pool:
            await self.db_pool.close()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all external services."""
        return {
            "llm_service": self._get_service_health("llm_service"),
            "cache_service": self._get_service_health("cache_service"),
            "database": self.db_pool.get_stats() if self.db_pool else {"status": "not_initialized"}
        }
    
    def _get_service_health(self, service_name: str) -> Dict[str, Any]:
        """Get health status for a specific service."""
        client = getattr(self, f"{service_name.split('_')[0]}_client", None)
        if not client:
            return {"status": "not_initialized"}
        
        circuit_breaker = client.circuit_breakers.get(service_name)
        if circuit_breaker:
            state = circuit_breaker.get_state()
            return {
                "status": "healthy" if state["state"] == "closed" else "degraded",
                "circuit_state": state["state"],
                "failure_count": state["failure_count"]
            }
        
        return {"status": "healthy", "circuit_state": "none"}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            "services": {}
        }
        
        # LLM service stats
        if self.llm_client:
            stats["services"]["llm"] = self.llm_client.get_stats()
        
        # Cache service stats
        if self.cache_client:
            stats["services"]["cache"] = self.cache_client.get_stats()
        
        # Database stats
        if self.db_pool:
            stats["services"]["database"] = self.db_pool.get_stats()
        
        return stats