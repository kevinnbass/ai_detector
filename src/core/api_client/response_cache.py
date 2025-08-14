"""
Response Cache for API Client
Intelligent caching of API responses with various strategies
"""

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union, List
from datetime import datetime, timedelta
from enum import Enum
import logging

from ..interfaces.api_interfaces import APIResponse, ResponseFormat

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache strategy types"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    FIFO = "fifo"  # First In, First Out


@dataclass
class CacheConfig:
    """Configuration for response caching"""
    
    # Basic settings
    enabled: bool = True
    max_size: int = 1000  # Maximum number of cached responses
    default_ttl: int = 300  # Default TTL in seconds (5 minutes)
    max_ttl: int = 3600  # Maximum TTL in seconds (1 hour)
    
    # Strategy
    strategy: CacheStrategy = CacheStrategy.LRU
    
    # Size limits
    max_response_size: int = 1024 * 1024  # 1MB max response size to cache
    
    # Cache keys
    include_headers_in_key: bool = False
    include_params_in_key: bool = True
    custom_key_function: Optional[callable] = None
    
    # Cache behavior
    cache_error_responses: bool = False
    error_response_ttl: int = 60  # Cache errors for 1 minute
    
    # Endpoint-specific TTLs
    endpoint_ttls: Dict[str, int] = None
    
    # Compression
    compress_responses: bool = True
    compression_threshold: int = 1024  # Compress responses larger than 1KB
    
    def __post_init__(self):
        if self.endpoint_ttls is None:
            self.endpoint_ttls = {}


@dataclass
class CacheEntry:
    """Represents a cached response"""
    
    key: str
    response: APIResponse
    created_at: datetime
    ttl: int
    access_count: int = 0
    last_accessed: datetime = None
    size: int = 0
    compressed: bool = False
    
    def __post_init__(self):
        if self.last_accessed is None:
            self.last_accessed = self.created_at
        
        # Calculate approximate size
        if hasattr(self.response, 'body'):
            try:
                self.size = len(json.dumps(self.response.body).encode('utf-8'))
            except:
                self.size = len(str(self.response.body).encode('utf-8'))
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        if self.ttl <= 0:  # Never expires
            return False
        
        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl
    
    def is_stale(self, staleness_threshold: int = 3600) -> bool:
        """Check if cache entry is stale (old but not expired)"""
        age = (datetime.now() - self.created_at).total_seconds()
        return age > staleness_threshold
    
    def access(self):
        """Record cache access"""
        self.access_count += 1
        self.last_accessed = datetime.now()
    
    def get_age(self) -> float:
        """Get cache entry age in seconds"""
        return (datetime.now() - self.created_at).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "key": self.key,
            "created_at": self.created_at.isoformat(),
            "ttl": self.ttl,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "size": self.size,
            "compressed": self.compressed,
            "age": self.get_age(),
            "expired": self.is_expired()
        }


class ResponseCache:
    """Response cache with multiple strategies"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []  # For LRU
        self._access_frequency: Dict[str, int] = {}  # For LFU
        self._insertion_order: List[str] = []  # For FIFO
        self._lock = asyncio.Lock()
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_size": 0,
            "compression_saves": 0
        }
        
        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._cleanup_interval = 300  # 5 minutes
    
    async def initialize(self) -> bool:
        """Initialize the cache"""
        if not self.config.enabled:
            logger.info("Response cache disabled")
            return True
        
        try:
            # Start cleanup task
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            logger.info(f"Response cache initialized with {self.config.strategy.value} strategy")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize response cache: {e}")
            return False
    
    async def get(self, url: str, params: Optional[Dict[str, Any]] = None,
                 headers: Optional[Dict[str, str]] = None) -> Optional[APIResponse]:
        """Get cached response"""
        
        if not self.config.enabled:
            return None
        
        cache_key = self._generate_cache_key(url, params, headers)
        
        async with self._lock:
            entry = self._cache.get(cache_key)
            
            if entry is None:
                self.stats["misses"] += 1
                logger.debug(f"Cache miss for key: {cache_key}")
                return None
            
            if entry.is_expired():
                # Remove expired entry
                await self._remove_entry(cache_key)
                self.stats["misses"] += 1
                logger.debug(f"Cache expired for key: {cache_key}")
                return None
            
            # Update access patterns
            entry.access()
            self._update_access_patterns(cache_key)
            
            self.stats["hits"] += 1
            logger.debug(f"Cache hit for key: {cache_key} (age: {entry.get_age():.1f}s)")
            
            return entry.response
    
    async def set(self, url: str, params: Optional[Dict[str, Any]] = None,
                 response: APIResponse = None, headers: Optional[Dict[str, str]] = None,
                 ttl: Optional[int] = None) -> bool:
        """Cache response"""
        
        if not self.config.enabled or response is None:
            return False
        
        # Don't cache error responses unless configured to do so
        if not self.config.cache_error_responses and response.status_code >= 400:
            return False
        
        # Don't cache responses that are too large
        response_size = self._estimate_response_size(response)
        if response_size > self.config.max_response_size:
            logger.debug(f"Response too large to cache: {response_size} bytes")
            return False
        
        cache_key = self._generate_cache_key(url, params, headers)
        
        # Determine TTL
        effective_ttl = self._get_effective_ttl(url, response.status_code, ttl)
        
        async with self._lock:
            # Check if we need to evict entries
            await self._ensure_capacity()
            
            # Create cache entry
            entry = CacheEntry(
                key=cache_key,
                response=response,
                created_at=datetime.now(),
                ttl=effective_ttl,
                size=response_size
            )
            
            # Compress if configured and beneficial
            if self.config.compress_responses and response_size > self.config.compression_threshold:
                compressed_response = self._compress_response(response)
                if compressed_response:
                    entry.response = compressed_response
                    entry.compressed = True
                    original_size = response_size
                    entry.size = self._estimate_response_size(compressed_response)
                    self.stats["compression_saves"] += original_size - entry.size
            
            # Store in cache
            self._cache[cache_key] = entry
            
            # Update tracking structures
            if self.config.strategy == CacheStrategy.LRU:
                if cache_key in self._access_order:
                    self._access_order.remove(cache_key)
                self._access_order.append(cache_key)
            
            elif self.config.strategy == CacheStrategy.LFU:
                self._access_frequency[cache_key] = self._access_frequency.get(cache_key, 0) + 1
            
            elif self.config.strategy == CacheStrategy.FIFO:
                if cache_key not in self._insertion_order:
                    self._insertion_order.append(cache_key)
            
            # Update stats
            self.stats["total_size"] += entry.size
            
            logger.debug(f"Cached response for key: {cache_key} (TTL: {effective_ttl}s, Size: {entry.size} bytes)")
            
            return True
    
    async def delete(self, url: str, params: Optional[Dict[str, Any]] = None,
                    headers: Optional[Dict[str, str]] = None) -> bool:
        """Delete cached response"""
        
        cache_key = self._generate_cache_key(url, params, headers)
        
        async with self._lock:
            return await self._remove_entry(cache_key)
    
    async def clear(self, pattern: Optional[str] = None) -> int:
        """Clear cache entries"""
        
        async with self._lock:
            if pattern is None:
                # Clear all entries
                count = len(self._cache)
                self._cache.clear()
                self._access_order.clear()
                self._access_frequency.clear()
                self._insertion_order.clear()
                self.stats["total_size"] = 0
                
                logger.info(f"Cleared all {count} cache entries")
                return count
            else:
                # Clear entries matching pattern
                keys_to_remove = [key for key in self._cache.keys() if pattern in key]
                count = 0
                
                for key in keys_to_remove:
                    if await self._remove_entry(key):
                        count += 1
                
                logger.info(f"Cleared {count} cache entries matching pattern: {pattern}")
                return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0.0
        
        return {
            **self.stats,
            "hit_rate": hit_rate,
            "cache_size": len(self._cache),
            "max_size": self.config.max_size,
            "utilization": len(self._cache) / self.config.max_size if self.config.max_size > 0 else 0.0,
            "average_entry_size": self.stats["total_size"] / max(1, len(self._cache)),
            "enabled": self.config.enabled,
            "strategy": self.config.strategy.value
        }
    
    def _generate_cache_key(self, url: str, params: Optional[Dict[str, Any]] = None,
                           headers: Optional[Dict[str, str]] = None) -> str:
        """Generate cache key"""
        
        if self.config.custom_key_function:
            return self.config.custom_key_function(url, params, headers)
        
        # Build key components
        key_parts = [url]
        
        if self.config.include_params_in_key and params:
            # Sort params for consistent keys
            sorted_params = sorted(params.items())
            key_parts.append(json.dumps(sorted_params, sort_keys=True))
        
        if self.config.include_headers_in_key and headers:
            # Only include certain headers that affect response
            relevant_headers = {k: v for k, v in headers.items() 
                              if k.lower() in ['accept', 'accept-language', 'authorization']}
            if relevant_headers:
                sorted_headers = sorted(relevant_headers.items())
                key_parts.append(json.dumps(sorted_headers, sort_keys=True))
        
        # Generate hash
        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode('utf-8')).hexdigest()
    
    def _get_effective_ttl(self, url: str, status_code: int, requested_ttl: Optional[int] = None) -> int:
        """Determine effective TTL for cache entry"""
        
        # Error responses have shorter TTL
        if status_code >= 400:
            return self.config.error_response_ttl
        
        # Use requested TTL if provided
        if requested_ttl is not None:
            return min(requested_ttl, self.config.max_ttl)
        
        # Check endpoint-specific TTL
        for endpoint_pattern, ttl in self.config.endpoint_ttls.items():
            if endpoint_pattern in url:
                return min(ttl, self.config.max_ttl)
        
        # Use default TTL
        return self.config.default_ttl
    
    def _estimate_response_size(self, response: APIResponse) -> int:
        """Estimate response size in bytes"""
        try:
            if hasattr(response, 'body'):
                return len(json.dumps(response.body).encode('utf-8'))
            else:
                return len(str(response).encode('utf-8'))
        except:
            return 1024  # Default estimate
    
    def _compress_response(self, response: APIResponse) -> Optional[APIResponse]:
        """Compress response if beneficial"""
        # Placeholder for compression logic
        # In practice, would use gzip or other compression
        return None
    
    async def _ensure_capacity(self):
        """Ensure cache has capacity for new entry"""
        while len(self._cache) >= self.config.max_size:
            await self._evict_entry()
    
    async def _evict_entry(self):
        """Evict entry based on strategy"""
        if not self._cache:
            return
        
        key_to_evict = None
        
        if self.config.strategy == CacheStrategy.LRU:
            # Remove least recently used
            key_to_evict = self._access_order[0] if self._access_order else None
        
        elif self.config.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            if self._access_frequency:
                key_to_evict = min(self._access_frequency.keys(), 
                                 key=lambda k: self._access_frequency[k])
        
        elif self.config.strategy == CacheStrategy.FIFO:
            # Remove first inserted
            key_to_evict = self._insertion_order[0] if self._insertion_order else None
        
        elif self.config.strategy == CacheStrategy.TTL:
            # Remove entry with shortest remaining TTL
            now = datetime.now()
            min_remaining_ttl = float('inf')
            
            for key, entry in self._cache.items():
                remaining_ttl = entry.ttl - (now - entry.created_at).total_seconds()
                if remaining_ttl < min_remaining_ttl:
                    min_remaining_ttl = remaining_ttl
                    key_to_evict = key
        
        if key_to_evict:
            await self._remove_entry(key_to_evict)
            self.stats["evictions"] += 1
            logger.debug(f"Evicted cache entry: {key_to_evict}")
    
    async def _remove_entry(self, key: str) -> bool:
        """Remove cache entry and update tracking structures"""
        entry = self._cache.get(key)
        if not entry:
            return False
        
        # Remove from cache
        del self._cache[key]
        
        # Update tracking structures
        if key in self._access_order:
            self._access_order.remove(key)
        
        if key in self._access_frequency:
            del self._access_frequency[key]
        
        if key in self._insertion_order:
            self._insertion_order.remove(key)
        
        # Update stats
        self.stats["total_size"] -= entry.size
        
        return True
    
    def _update_access_patterns(self, key: str):
        """Update access patterns for cache strategies"""
        
        if self.config.strategy == CacheStrategy.LRU:
            # Move to end of access order
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
        
        elif self.config.strategy == CacheStrategy.LFU:
            # Increment access frequency
            self._access_frequency[key] = self._access_frequency.get(key, 0) + 1
    
    async def _cleanup_loop(self):
        """Periodic cleanup of expired entries"""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                await self._cleanup_expired()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
                await asyncio.sleep(60)  # Wait on error
    
    async def _cleanup_expired(self):
        """Remove expired entries"""
        async with self._lock:
            expired_keys = []
            
            for key, entry in self._cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            count = 0
            for key in expired_keys:
                if await self._remove_entry(key):
                    count += 1
            
            if count > 0:
                logger.debug(f"Cleaned up {count} expired cache entries")
    
    async def close(self):
        """Close cache and cleanup resources"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        await self.clear()  # Clear all entries
        
        logger.info("Response cache closed")