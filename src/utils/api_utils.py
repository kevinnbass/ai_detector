"""
API Utilities - Shared API and HTTP Functions
"""

import asyncio
import aiohttp
import requests
import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging

from .common import RetryManager, RateLimiter, SimpleCache

logger = logging.getLogger(__name__)


# ============================================
# Data Classes for API Responses
# ============================================

@dataclass
class APIResponse:
    """Standard API response structure"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    status_code: Optional[int] = None
    headers: Optional[Dict[str, str]] = None
    response_time: Optional[float] = None


@dataclass
class APIConfig:
    """API configuration"""
    base_url: str
    api_key: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    rate_limit_calls: int = 100
    rate_limit_window: int = 60  # seconds
    cache_ttl: int = 900  # seconds


# ============================================
# Base API Client
# ============================================

class BaseAPIClient:
    """Base API client with common functionality"""
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.session = None
        self.cache = SimpleCache(config.cache_ttl)
        self.rate_limiter = RateLimiter(config.rate_limit_calls, config.rate_limit_window)
        self.retry_manager = RetryManager(config.max_retries)
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def get_headers(self) -> Dict[str, str]:
        """Get default headers"""
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'AI-Detector/1.0'
        }
        
        if self.config.api_key:
            headers['Authorization'] = f'Bearer {self.config.api_key}'
        
        return headers
    
    def build_url(self, endpoint: str) -> str:
        """Build full URL"""
        return f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    
    def get_cache_key(self, method: str, url: str, params: Optional[Dict] = None,
                     data: Optional[Any] = None) -> str:
        """Generate cache key"""
        key_parts = [method.upper(), url]
        if params:
            key_parts.append(json.dumps(params, sort_keys=True))
        if data:
            key_parts.append(json.dumps(data, sort_keys=True))
        return '|'.join(key_parts)
    
    async def request(self, method: str, endpoint: str, 
                     params: Optional[Dict] = None,
                     data: Optional[Any] = None,
                     use_cache: bool = True) -> APIResponse:
        """Make async HTTP request"""
        url = self.build_url(endpoint)
        cache_key = self.get_cache_key(method, url, params, data)
        
        # Check cache for GET requests
        if method.upper() == 'GET' and use_cache:
            cached = self.cache.get(cache_key)
            if cached:
                return cached
        
        # Rate limiting
        if not self.rate_limiter.allow():
            wait_time = self.rate_limiter.wait_time()
            await asyncio.sleep(wait_time)
        
        start_time = time.time()
        
        try:
            async def make_request():
                async with self.session.request(
                    method,
                    url,
                    params=params,
                    json=data if data else None,
                    headers=self.get_headers()
                ) as response:
                    response_data = await response.json() if response.content_type == 'application/json' else await response.text()
                    
                    api_response = APIResponse(
                        success=response.status < 400,
                        data=response_data,
                        status_code=response.status,
                        headers=dict(response.headers),
                        response_time=time.time() - start_time
                    )
                    
                    if not api_response.success:
                        api_response.error = f"HTTP {response.status}: {response_data}"
                    
                    return api_response
            
            result = await self.retry_manager.execute(make_request)
            
            # Cache successful GET responses
            if result.success and method.upper() == 'GET' and use_cache:
                self.cache.set(cache_key, result)
            
            return result
            
        except Exception as e:
            return APIResponse(
                success=False,
                error=str(e),
                response_time=time.time() - start_time
            )


# ============================================
# Sync API Client
# ============================================

class SyncAPIClient:
    """Synchronous API client"""
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.cache = SimpleCache(config.cache_ttl)
        self.rate_limiter = RateLimiter(config.rate_limit_calls, config.rate_limit_window)
        self.retry_manager = RetryManager(config.max_retries)
        self.session = requests.Session()
        self.session.headers.update(self.get_headers())
    
    def get_headers(self) -> Dict[str, str]:
        """Get default headers"""
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'AI-Detector/1.0'
        }
        
        if self.config.api_key:
            headers['Authorization'] = f'Bearer {self.config.api_key}'
        
        return headers
    
    def build_url(self, endpoint: str) -> str:
        """Build full URL"""
        return f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    
    def get_cache_key(self, method: str, url: str, params: Optional[Dict] = None,
                     data: Optional[Any] = None) -> str:
        """Generate cache key"""
        key_parts = [method.upper(), url]
        if params:
            key_parts.append(json.dumps(params, sort_keys=True))
        if data:
            key_parts.append(json.dumps(data, sort_keys=True))
        return '|'.join(key_parts)
    
    def request(self, method: str, endpoint: str,
               params: Optional[Dict] = None,
               data: Optional[Any] = None,
               use_cache: bool = True) -> APIResponse:
        """Make sync HTTP request"""
        url = self.build_url(endpoint)
        cache_key = self.get_cache_key(method, url, params, data)
        
        # Check cache for GET requests
        if method.upper() == 'GET' and use_cache:
            cached = self.cache.get(cache_key)
            if cached:
                return cached
        
        # Rate limiting
        if not self.rate_limiter.allow():
            wait_time = self.rate_limiter.wait_time()
            time.sleep(wait_time)
        
        start_time = time.time()
        
        def make_request():
            response = self.session.request(
                method,
                url,
                params=params,
                json=data if data else None,
                timeout=self.config.timeout
            )
            
            try:
                response_data = response.json()
            except:
                response_data = response.text
            
            api_response = APIResponse(
                success=response.status_code < 400,
                data=response_data,
                status_code=response.status_code,
                headers=dict(response.headers),
                response_time=time.time() - start_time
            )
            
            if not api_response.success:
                api_response.error = f"HTTP {response.status_code}: {response_data}"
            
            return api_response
        
        try:
            result = self.retry_manager.execute(make_request)
            
            # Cache successful GET responses
            if result.success and method.upper() == 'GET' and use_cache:
                self.cache.set(cache_key, result)
            
            return result
            
        except Exception as e:
            return APIResponse(
                success=False,
                error=str(e),
                response_time=time.time() - start_time
            )


# ============================================
# LLM API Utilities
# ============================================

class LLMAPIClient:
    """Specialized client for LLM APIs"""
    
    def __init__(self, config: APIConfig):
        self.client = SyncAPIClient(config)
    
    def generate_text(self, prompt: str, model: str = "gpt-3.5-turbo",
                     max_tokens: int = 150, temperature: float = 0.7) -> APIResponse:
        """Generate text using LLM"""
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        return self.client.request("POST", "/chat/completions", data=data)
    
    def analyze_text(self, text: str, analysis_type: str = "sentiment") -> APIResponse:
        """Analyze text using LLM"""
        prompts = {
            "sentiment": f"Analyze the sentiment of this text: {text}",
            "ai_detection": f"Determine if this text was written by AI: {text}",
            "classification": f"Classify this text: {text}"
        }
        
        prompt = prompts.get(analysis_type, f"Analyze this text: {text}")
        return self.generate_text(prompt)
    
    def batch_analyze(self, texts: List[str], analysis_type: str = "ai_detection") -> List[APIResponse]:
        """Batch analyze multiple texts"""
        results = []
        for text in texts:
            result = self.analyze_text(text, analysis_type)
            results.append(result)
        return results


# ============================================
# Response Processing Utilities
# ============================================

def extract_json_from_response(response: APIResponse, key: Optional[str] = None) -> Optional[Any]:
    """Extract JSON data from API response"""
    if not response.success or not response.data:
        return None
    
    data = response.data
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return None
    
    if key and isinstance(data, dict):
        return data.get(key)
    
    return data


def validate_api_response(response: APIResponse, required_fields: List[str] = None) -> Tuple[bool, List[str]]:
    """Validate API response structure"""
    errors = []
    
    if not response.success:
        errors.append(f"Request failed: {response.error}")
        return False, errors
    
    if required_fields and isinstance(response.data, dict):
        for field in required_fields:
            if field not in response.data:
                errors.append(f"Missing required field: {field}")
    
    return len(errors) == 0, errors


def format_error_message(response: APIResponse) -> str:
    """Format user-friendly error message"""
    if response.error:
        return response.error
    
    if response.status_code:
        if response.status_code == 401:
            return "Authentication failed. Please check your API key."
        elif response.status_code == 429:
            return "Rate limit exceeded. Please try again later."
        elif response.status_code >= 500:
            return "Server error. Please try again later."
        else:
            return f"Request failed with status {response.status_code}"
    
    return "Unknown error occurred"


# ============================================
# Webhook Utilities
# ============================================

class WebhookHandler:
    """Simple webhook handler"""
    
    def __init__(self):
        self.handlers = {}
    
    def register(self, event_type: str, handler_func):
        """Register event handler"""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler_func)
    
    def handle(self, event_type: str, data: Dict[str, Any]) -> List[Any]:
        """Handle webhook event"""
        results = []
        if event_type in self.handlers:
            for handler in self.handlers[event_type]:
                try:
                    result = handler(data)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error in webhook handler: {e}")
                    results.append(None)
        return results


# ============================================
# API Testing Utilities
# ============================================

def test_api_endpoint(client: Union[BaseAPIClient, SyncAPIClient], 
                     endpoint: str, method: str = "GET",
                     expected_status: int = 200) -> Dict[str, Any]:
    """Test API endpoint"""
    result = {
        'endpoint': endpoint,
        'method': method,
        'success': False,
        'response_time': 0,
        'status_code': None,
        'error': None
    }
    
    try:
        if isinstance(client, BaseAPIClient):
            # Async client - would need to run in async context
            result['error'] = "Async testing not implemented"
        else:
            response = client.request(method, endpoint)
            result['success'] = response.success and response.status_code == expected_status
            result['response_time'] = response.response_time
            result['status_code'] = response.status_code
            result['error'] = response.error
    
    except Exception as e:
        result['error'] = str(e)
    
    return result


# Export all utilities
__all__ = [
    # Data Classes
    'APIResponse', 'APIConfig',
    
    # Clients
    'BaseAPIClient', 'SyncAPIClient', 'LLMAPIClient',
    
    # Response Processing
    'extract_json_from_response', 'validate_api_response', 'format_error_message',
    
    # Webhook
    'WebhookHandler',
    
    # Testing
    'test_api_endpoint'
]