"""
Queue Manager for API Requests
Handles request queuing, prioritization, and processing
"""

import asyncio
import heapq
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, AsyncGenerator
from datetime import datetime, timedelta
from enum import Enum
import logging

from .exceptions import QueueFullError, APIClientError

logger = logging.getLogger(__name__)


class Priority(Enum):
    """Request priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class RequestStatus(Enum):
    """Request processing status"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class QueuedRequest:
    """Represents a queued API request"""
    id: str
    method: str
    url: str
    data: Optional[Dict[str, Any]] = None
    headers: Optional[Dict[str, str]] = None
    params: Optional[Dict[str, Any]] = None
    priority: Priority = Priority.NORMAL
    timeout: float = 30.0
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    callback: Optional[Callable] = None
    context: Optional[Dict[str, Any]] = None
    
    # Status tracking
    status: RequestStatus = RequestStatus.QUEUED
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[Exception] = None
    response: Optional[Any] = None
    
    def __lt__(self, other: 'QueuedRequest') -> bool:
        """Priority queue ordering (higher priority first)"""
        if self.priority != other.priority:
            return self.priority.value > other.priority.value
        return self.created_at < other.created_at
    
    def get_age(self) -> timedelta:
        """Get request age"""
        return datetime.now() - self.created_at
    
    def is_expired(self, max_age: timedelta) -> bool:
        """Check if request has expired"""
        return self.get_age() > max_age
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "method": self.method,
            "url": self.url,
            "priority": self.priority.name,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "status": self.status.name,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "age_seconds": self.get_age().total_seconds(),
            "has_error": self.error is not None,
            "error_message": str(self.error) if self.error else None
        }


class RequestQueue:
    """Priority queue for API requests"""
    
    def __init__(self, max_size: int = 1000, max_age: timedelta = timedelta(hours=1)):
        self.max_size = max_size
        self.max_age = max_age
        self._queue: List[QueuedRequest] = []
        self._requests: Dict[str, QueuedRequest] = {}
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Condition()
        
        # Statistics
        self.stats = {
            "total_queued": 0,
            "total_processed": 0,
            "total_failed": 0,
            "total_expired": 0,
            "total_cancelled": 0
        }
    
    async def put(self, request: QueuedRequest) -> None:
        """Add request to queue"""
        async with self._lock:
            if len(self._queue) >= self.max_size:
                raise QueueFullError(
                    f"Queue is full (size: {len(self._queue)}, max: {self.max_size})",
                    queue_size=len(self._queue),
                    max_size=self.max_size
                )
            
            # Clean up expired requests first
            await self._cleanup_expired()
            
            heapq.heappush(self._queue, request)
            self._requests[request.id] = request
            self.stats["total_queued"] += 1
            
            logger.debug(f"Queued request {request.id} with priority {request.priority.name}")
            
            async with self._not_empty:
                self._not_empty.notify()
    
    async def get(self) -> Optional[QueuedRequest]:
        """Get next request from queue"""
        async with self._not_empty:
            while not self._queue:
                await self._not_empty.wait()
            
            async with self._lock:
                if not self._queue:
                    return None
                
                request = heapq.heappop(self._queue)
                
                # Check if request is still valid
                if request.is_expired(self.max_age):
                    self._handle_expired_request(request)
                    return await self.get()  # Try again
                
                request.status = RequestStatus.PROCESSING
                request.started_at = datetime.now()
                
                logger.debug(f"Dequeued request {request.id}")
                return request
    
    async def complete_request(self, request_id: str, response: Any = None, error: Exception = None) -> None:
        """Mark request as completed"""
        async with self._lock:
            request = self._requests.get(request_id)
            if not request:
                return
            
            request.completed_at = datetime.now()
            
            if error:
                request.status = RequestStatus.FAILED
                request.error = error
                self.stats["total_failed"] += 1
                logger.warning(f"Request {request_id} failed: {error}")
            else:
                request.status = RequestStatus.COMPLETED
                request.response = response
                self.stats["total_processed"] += 1
                logger.debug(f"Request {request_id} completed successfully")
            
            # Execute callback if provided
            if request.callback:
                try:
                    await request.callback(request)
                except Exception as e:
                    logger.error(f"Callback error for request {request_id}: {e}")
    
    async def cancel_request(self, request_id: str) -> bool:
        """Cancel a queued request"""
        async with self._lock:
            request = self._requests.get(request_id)
            if not request or request.status != RequestStatus.QUEUED:
                return False
            
            # Remove from queue
            try:
                self._queue.remove(request)
                heapq.heapify(self._queue)  # Restore heap property
            except ValueError:
                pass  # Request not in queue
            
            request.status = RequestStatus.CANCELLED
            request.completed_at = datetime.now()
            self.stats["total_cancelled"] += 1
            
            logger.debug(f"Cancelled request {request_id}")
            return True
    
    async def get_request(self, request_id: str) -> Optional[QueuedRequest]:
        """Get request by ID"""
        async with self._lock:
            return self._requests.get(request_id)
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        async with self._lock:
            return {
                **self.stats,
                "queue_size": len(self._queue),
                "max_size": self.max_size,
                "pending_requests": sum(1 for r in self._requests.values() 
                                      if r.status == RequestStatus.QUEUED),
                "processing_requests": sum(1 for r in self._requests.values() 
                                         if r.status == RequestStatus.PROCESSING),
                "priority_distribution": self._get_priority_distribution()
            }
    
    async def clear_completed(self, older_than: timedelta = timedelta(minutes=30)) -> int:
        """Clear completed requests older than specified time"""
        async with self._lock:
            cutoff_time = datetime.now() - older_than
            to_remove = []
            
            for request_id, request in self._requests.items():
                if (request.status in [RequestStatus.COMPLETED, RequestStatus.FAILED, RequestStatus.CANCELLED] and
                    request.completed_at and request.completed_at < cutoff_time):
                    to_remove.append(request_id)
            
            for request_id in to_remove:
                del self._requests[request_id]
            
            logger.debug(f"Cleaned up {len(to_remove)} completed requests")
            return len(to_remove)
    
    async def _cleanup_expired(self) -> None:
        """Clean up expired requests"""
        expired_requests = []
        
        for request in list(self._queue):
            if request.is_expired(self.max_age):
                expired_requests.append(request)
        
        for request in expired_requests:
            try:
                self._queue.remove(request)
                self._handle_expired_request(request)
            except ValueError:
                pass  # Already removed
        
        if expired_requests:
            heapq.heapify(self._queue)  # Restore heap property
            logger.debug(f"Cleaned up {len(expired_requests)} expired requests")
    
    def _handle_expired_request(self, request: QueuedRequest) -> None:
        """Handle expired request"""
        request.status = RequestStatus.TIMEOUT
        request.completed_at = datetime.now()
        request.error = APIClientError("Request expired in queue")
        self.stats["total_expired"] += 1
        
        logger.warning(f"Request {request.id} expired after {request.get_age()}")
    
    def _get_priority_distribution(self) -> Dict[str, int]:
        """Get distribution of requests by priority"""
        distribution = {priority.name: 0 for priority in Priority}
        
        for request in self._queue:
            distribution[request.priority.name] += 1
        
        return distribution


class QueueManager:
    """Manages multiple request queues and processing"""
    
    def __init__(self, 
                 max_concurrent: int = 10,
                 max_queue_size: int = 1000,
                 cleanup_interval: int = 300):  # 5 minutes
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        self.cleanup_interval = cleanup_interval
        
        # Queues for different endpoints/services
        self.queues: Dict[str, RequestQueue] = {}
        self.default_queue = RequestQueue(max_size=max_queue_size)
        
        # Processing control
        self._workers: List[asyncio.Task] = []
        self._running = False
        self._semaphore = asyncio.Semaphore(max_concurrent)
        
        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.global_stats = {
            "total_requests": 0,
            "active_workers": 0,
            "queues_count": 0
        }
    
    async def start(self) -> None:
        """Start queue processing"""
        if self._running:
            return
        
        self._running = True
        logger.info(f"Starting queue manager with {self.max_concurrent} workers")
        
        # Start worker tasks
        for i in range(self.max_concurrent):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self._workers.append(worker)
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Queue manager started successfully")
    
    async def stop(self) -> None:
        """Stop queue processing"""
        if not self._running:
            return
        
        self._running = False
        logger.info("Stopping queue manager...")
        
        # Cancel workers
        for worker in self._workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Queue manager stopped")
    
    def get_queue(self, name: str = "default") -> RequestQueue:
        """Get or create a queue by name"""
        if name == "default":
            return self.default_queue
        
        if name not in self.queues:
            self.queues[name] = RequestQueue(max_size=self.max_queue_size)
            self.global_stats["queues_count"] = len(self.queues) + 1  # +1 for default
        
        return self.queues[name]
    
    async def enqueue_request(self, 
                            method: str,
                            url: str,
                            data: Optional[Dict[str, Any]] = None,
                            headers: Optional[Dict[str, str]] = None,
                            params: Optional[Dict[str, Any]] = None,
                            priority: Priority = Priority.NORMAL,
                            queue_name: str = "default",
                            timeout: float = 30.0,
                            max_retries: int = 3,
                            callback: Optional[Callable] = None,
                            context: Optional[Dict[str, Any]] = None) -> str:
        """Enqueue an API request"""
        
        request_id = str(uuid.uuid4())
        request = QueuedRequest(
            id=request_id,
            method=method,
            url=url,
            data=data,
            headers=headers,
            params=params,
            priority=priority,
            timeout=timeout,
            max_retries=max_retries,
            callback=callback,
            context=context
        )
        
        queue = self.get_queue(queue_name)
        await queue.put(request)
        
        self.global_stats["total_requests"] += 1
        
        logger.debug(f"Enqueued request {request_id} to queue '{queue_name}'")
        return request_id
    
    async def get_request_status(self, request_id: str, queue_name: str = "default") -> Optional[Dict[str, Any]]:
        """Get request status"""
        queue = self.get_queue(queue_name)
        request = await queue.get_request(request_id)
        return request.to_dict() if request else None
    
    async def cancel_request(self, request_id: str, queue_name: str = "default") -> bool:
        """Cancel a request"""
        queue = self.get_queue(queue_name)
        return await queue.cancel_request(request_id)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        stats = {**self.global_stats}
        
        # Add queue-specific stats
        stats["default_queue"] = await self.default_queue.get_queue_stats()
        stats["named_queues"] = {}
        
        for name, queue in self.queues.items():
            stats["named_queues"][name] = await queue.get_queue_stats()
        
        stats["active_workers"] = sum(1 for w in self._workers if not w.done())
        stats["is_running"] = self._running
        
        return stats
    
    async def _worker(self, worker_name: str) -> None:
        """Worker coroutine for processing requests"""
        logger.debug(f"Worker {worker_name} started")
        
        while self._running:
            try:
                # Try all queues (default first, then named queues)
                all_queues = [("default", self.default_queue)] + list(self.queues.items())
                
                request = None
                queue_name = None
                
                for name, queue in all_queues:
                    try:
                        # Non-blocking get with short timeout
                        request = await asyncio.wait_for(queue.get(), timeout=0.1)
                        if request:
                            queue_name = name
                            break
                    except asyncio.TimeoutError:
                        continue
                
                if not request:
                    # No requests available, wait a bit
                    await asyncio.sleep(0.1)
                    continue
                
                # Process the request
                async with self._semaphore:
                    await self._process_request(request, queue_name)
                    
            except asyncio.CancelledError:
                logger.debug(f"Worker {worker_name} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
                await asyncio.sleep(1)  # Brief pause on error
        
        logger.debug(f"Worker {worker_name} stopped")
    
    async def _process_request(self, request: QueuedRequest, queue_name: str) -> None:
        """Process a single request"""
        queue = self.get_queue(queue_name)
        
        try:
            # This is where the actual HTTP request would be made
            # For now, we'll simulate processing
            logger.debug(f"Processing request {request.id}: {request.method} {request.url}")
            
            # Simulate processing time
            await asyncio.sleep(0.1)
            
            # Mark as completed (would normally have real response)
            await queue.complete_request(request.id, response={"status": "simulated"})
            
        except Exception as e:
            logger.error(f"Error processing request {request.id}: {e}")
            await queue.complete_request(request.id, error=e)
    
    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of completed requests"""
        while self._running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                # Clean up all queues
                total_cleaned = await self.default_queue.clear_completed()
                
                for queue in self.queues.values():
                    total_cleaned += await queue.clear_completed()
                
                if total_cleaned > 0:
                    logger.debug(f"Cleaned up {total_cleaned} completed requests")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(60)  # Wait a minute on error