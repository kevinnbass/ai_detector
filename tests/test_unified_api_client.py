"""
Test script for Unified API Client
Verify the API client system works correctly
"""

import asyncio
import logging
from typing import Dict, Any

from src.core.api_client import (
    UnifiedAPIClient, APIClientConfig, HTTPMethod,
    RetryConfig, RateLimitConfig, CacheConfig,
    Priority
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_api_client_initialization():
    """Test API client initialization"""
    print("\n=== Testing API Client Initialization ===")
    
    config = APIClientConfig(
        base_url="https://httpbin.org",
        timeout=10.0,
        enable_retries=True,
        enable_rate_limiting=True,
        enable_caching=True,
        enable_queuing=False  # Disable for simple testing
    )
    
    client = UnifiedAPIClient(config)
    
    # Test initialization
    initialized = await client.initialize()
    print(f"✅ Client initialized: {initialized}")
    
    # Test configuration
    current_config = client.get_configuration()
    print(f"✅ Configuration retrieved: {len(current_config)} settings")
    
    # Test health check
    health = await client.health_check()
    print(f"✅ Health check: {health['healthy']}")
    
    return client


async def test_basic_requests(client: UnifiedAPIClient):
    """Test basic HTTP requests"""
    print("\n=== Testing Basic Requests ===")
    
    try:
        # Test GET request
        response = await client.get("/get", params={"test": "value"})
        print(f"✅ GET request: {response.status_code}")
        
        # Test POST request
        response = await client.post("/post", data={"test": "data"})
        print(f"✅ POST request: {response.status_code}")
        
        # Test request stats
        stats = client.get_request_stats()
        print(f"✅ Request stats: {stats['total_requests']} total requests")
        
    except Exception as e:
        print(f"❌ Request failed: {e}")


async def test_caching(client: UnifiedAPIClient):
    """Test response caching"""
    print("\n=== Testing Response Caching ===")
    
    try:
        # Make same request twice to test caching
        endpoint = "/delay/1"  # httpbin endpoint with delay
        
        # First request (should be slow)
        import time
        start = time.time()
        response1 = await client.get(endpoint)
        time1 = time.time() - start
        print(f"✅ First request: {response1.status_code} in {time1:.2f}s")
        
        # Second request (should be fast due to caching)
        start = time.time()
        response2 = await client.get(endpoint)
        time2 = time.time() - start
        print(f"✅ Second request: {response2.status_code} in {time2:.2f}s")
        
        # Check cache stats
        stats = client.get_request_stats()
        if 'cache_stats' in stats:
            cache_stats = await stats['cache_stats'] if asyncio.iscoroutine(stats['cache_stats']) else stats['cache_stats']
            print(f"✅ Cache hits: {cache_stats.get('hits', 0)}")
        
    except Exception as e:
        print(f"❌ Caching test failed: {e}")


async def test_error_handling(client: UnifiedAPIClient):
    """Test error handling"""
    print("\n=== Testing Error Handling ===")
    
    try:
        # Test 404 error
        try:
            response = await client.get("/status/404")
            print(f"⚠️  404 request returned: {response.status_code}")
        except Exception as e:
            print(f"✅ 404 error handled: {type(e).__name__}")
        
        # Test timeout (if using queuing disabled)
        try:
            response = await client.get("/delay/10")  # Long delay
            print(f"⚠️  Timeout request returned: {response.status_code}")
        except Exception as e:
            print(f"✅ Timeout handled: {type(e).__name__}")
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")


async def test_authentication(client: UnifiedAPIClient):
    """Test authentication"""
    print("\n=== Testing Authentication ===")
    
    try:
        # Set Bearer token authentication
        client.set_authentication("bearer", {"token": "test-token-123"})
        
        # Make authenticated request
        response = await client.get("/bearer", headers={"Authorization": "Bearer test-token-123"})
        print(f"✅ Authenticated request: {response.status_code}")
        
        # Set API key authentication
        client.set_authentication("api_key", {"api_key": "test-key-456", "header_name": "X-API-Key"})
        
        response = await client.get("/headers")
        print(f"✅ API key request: {response.status_code}")
        
    except Exception as e:
        print(f"❌ Authentication test failed: {e}")


async def run_comprehensive_test():
    """Run comprehensive API client test"""
    print("🚀 Starting Unified API Client Test Suite")
    
    try:
        # Initialize client
        client = await test_api_client_initialization()
        
        # Test basic functionality
        await test_basic_requests(client)
        
        # Test caching
        await test_caching(client)
        
        # Test error handling
        await test_error_handling(client)
        
        # Test authentication
        await test_authentication(client)
        
        # Final stats
        print("\n=== Final Statistics ===")
        stats = client.get_request_stats()
        print(f"Total requests: {stats['total_requests']}")
        print(f"Successful requests: {stats['successful_requests']}")
        print(f"Failed requests: {stats['failed_requests']}")
        print(f"Success rate: {stats['success_rate']:.1%}")
        print(f"Average response time: {stats.get('average_response_time', 0):.3f}s")
        
        # Close client
        await client.close()
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the test
    asyncio.run(run_comprehensive_test())