"""
Observer Pattern Implementation - Event Bus System
Provides decoupled communication between components
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Set
import asyncio
import logging
from datetime import datetime
from enum import Enum
import uuid
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class EventPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Event:
    """Base event class"""
    event_type: str
    data: Any
    source: str
    priority: EventPriority = EventPriority.NORMAL
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "data": self.data,
            "source": self.source,
            "priority": self.priority.value,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "metadata": self.metadata
        }


class IEventHandler(ABC):
    """Interface for event handlers"""
    
    @abstractmethod
    async def handle(self, event: Event) -> None:
        """Handle event"""
        pass
    
    @abstractmethod
    def get_handler_info(self) -> Dict[str, Any]:
        """Get handler information"""
        pass
    
    @abstractmethod
    def can_handle(self, event: Event) -> bool:
        """Check if handler can process this event"""
        pass


class BaseEventHandler(IEventHandler):
    """Base event handler implementation"""
    
    def __init__(self, name: str, supported_events: Optional[Set[str]] = None):
        self.name = name
        self.supported_events = supported_events or set()
        self.processed_count = 0
        self.error_count = 0
        self.last_processed = None
        
    async def handle(self, event: Event) -> None:
        """Handle event with error tracking"""
        try:
            if self.can_handle(event):
                await self._process_event(event)
                self.processed_count += 1
                self.last_processed = datetime.utcnow()
                logger.debug(f"Handler {self.name} processed event {event.event_id}")
            else:
                logger.warning(f"Handler {self.name} cannot process event type {event.event_type}")
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error in handler {self.name} processing event {event.event_id}: {e}")
            raise
    
    @abstractmethod
    async def _process_event(self, event: Event) -> None:
        """Process event - override in subclasses"""
        pass
    
    def can_handle(self, event: Event) -> bool:
        """Check if handler can process this event"""
        if not self.supported_events:
            return True  # Handle all events if no specific types defined
        return event.event_type in self.supported_events
    
    def get_handler_info(self) -> Dict[str, Any]:
        """Get handler information"""
        return {
            "name": self.name,
            "supported_events": list(self.supported_events),
            "processed_count": self.processed_count,
            "error_count": self.error_count,
            "last_processed": self.last_processed.isoformat() if self.last_processed else None
        }


class FunctionEventHandler(BaseEventHandler):
    """Event handler that wraps a function"""
    
    def __init__(self, name: str, handler_func: Callable[[Event], Any], 
                 supported_events: Optional[Set[str]] = None):
        super().__init__(name, supported_events)
        self.handler_func = handler_func
        
    async def _process_event(self, event: Event) -> None:
        """Process event using wrapped function"""
        if asyncio.iscoroutinefunction(self.handler_func):
            await self.handler_func(event)
        else:
            # Run sync function in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.handler_func, event)


class IEventBus(ABC):
    """Event bus interface"""
    
    @abstractmethod
    async def publish(self, event: Event) -> None:
        """Publish event to all subscribers"""
        pass
    
    @abstractmethod
    def subscribe(self, event_type: str, handler: IEventHandler) -> str:
        """Subscribe handler to event type"""
        pass
    
    @abstractmethod
    def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe handler"""
        pass
    
    @abstractmethod
    async def start(self) -> None:
        """Start event bus"""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop event bus"""
        pass


class EventBus(IEventBus):
    """Main event bus implementation"""
    
    def __init__(self, max_queue_size: int = 1000, max_workers: int = 5):
        self.max_queue_size = max_queue_size
        self.max_workers = max_workers
        self.subscriptions = {}  # event_type -> list of (subscription_id, handler)
        self.subscription_by_id = {}  # subscription_id -> (event_type, handler)
        self.event_queue = asyncio.Queue(maxsize=max_queue_size)
        self.workers = []
        self.running = False
        self.stats = {
            "events_published": 0,
            "events_processed": 0,
            "events_failed": 0,
            "subscriptions_count": 0
        }
        
    async def start(self) -> None:
        """Start event bus workers"""
        if self.running:
            return
            
        self.running = True
        
        # Start worker tasks
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
            
        logger.info(f"Started event bus with {self.max_workers} workers")
    
    async def stop(self) -> None:
        """Stop event bus"""
        if not self.running:
            return
            
        self.running = False
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
        
        logger.info("Stopped event bus")
    
    async def _worker(self, worker_name: str) -> None:
        """Event processing worker"""
        logger.debug(f"Started event bus worker: {worker_name}")
        
        while self.running:
            try:
                # Wait for event with timeout to allow checking running status
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                await self._process_event(event)
                self.event_queue.task_done()
                
            except asyncio.TimeoutError:
                continue  # Check running status again
            except Exception as e:
                logger.error(f"Error in event bus worker {worker_name}: {e}")
        
        logger.debug(f"Stopped event bus worker: {worker_name}")
    
    async def _process_event(self, event: Event) -> None:
        """Process single event"""
        try:
            handlers = self.subscriptions.get(event.event_type, [])
            wildcard_handlers = self.subscriptions.get("*", [])
            all_handlers = handlers + wildcard_handlers
            
            if not all_handlers:
                logger.debug(f"No handlers for event type: {event.event_type}")
                return
            
            # Process handlers based on priority
            priority_groups = {}
            for subscription_id, handler in all_handlers:
                priority = getattr(handler, 'priority', EventPriority.NORMAL)
                if priority not in priority_groups:
                    priority_groups[priority] = []
                priority_groups[priority].append(handler)
            
            # Process in priority order (highest first)
            for priority in sorted(priority_groups.keys(), key=lambda p: p.value, reverse=True):
                handlers_group = priority_groups[priority]
                
                # Run handlers in parallel for same priority
                tasks = [handler.handle(event) for handler in handlers_group]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Log any errors
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        handler_name = getattr(handlers_group[i], 'name', 'unknown')
                        logger.error(f"Handler {handler_name} failed for event {event.event_id}: {result}")
                        self.stats["events_failed"] += 1
            
            self.stats["events_processed"] += 1
            logger.debug(f"Processed event {event.event_id} with {len(all_handlers)} handlers")
            
        except Exception as e:
            logger.error(f"Error processing event {event.event_id}: {e}")
            self.stats["events_failed"] += 1
    
    async def publish(self, event: Event) -> None:
        """Publish event to queue"""
        if not self.running:
            raise RuntimeError("Event bus not running")
        
        try:
            # Add to queue (will block if queue is full)
            await self.event_queue.put(event)
            self.stats["events_published"] += 1
            logger.debug(f"Published event {event.event_id} of type {event.event_type}")
            
        except Exception as e:
            logger.error(f"Failed to publish event {event.event_id}: {e}")
            raise
    
    def subscribe(self, event_type: str, handler: IEventHandler) -> str:
        """Subscribe handler to event type"""
        subscription_id = str(uuid.uuid4())
        
        # Add to subscriptions
        if event_type not in self.subscriptions:
            self.subscriptions[event_type] = []
        
        self.subscriptions[event_type].append((subscription_id, handler))
        self.subscription_by_id[subscription_id] = (event_type, handler)
        
        self.stats["subscriptions_count"] += 1
        
        handler_name = getattr(handler, 'name', 'unknown')
        logger.info(f"Subscribed handler {handler_name} to event type {event_type}")
        
        return subscription_id
    
    def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe handler"""
        if subscription_id not in self.subscription_by_id:
            logger.warning(f"Subscription ID not found: {subscription_id}")
            return
        
        event_type, handler = self.subscription_by_id[subscription_id]
        
        # Remove from subscriptions
        if event_type in self.subscriptions:
            self.subscriptions[event_type] = [
                (sid, h) for sid, h in self.subscriptions[event_type]
                if sid != subscription_id
            ]
            
            # Remove empty event type entries
            if not self.subscriptions[event_type]:
                del self.subscriptions[event_type]
        
        del self.subscription_by_id[subscription_id]
        self.stats["subscriptions_count"] -= 1
        
        handler_name = getattr(handler, 'name', 'unknown')
        logger.info(f"Unsubscribed handler {handler_name} from event type {event_type}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics"""
        return {
            **self.stats,
            "queue_size": self.event_queue.qsize(),
            "max_queue_size": self.max_queue_size,
            "running": self.running,
            "worker_count": len(self.workers),
            "subscription_types": list(self.subscriptions.keys())
        }
    
    def get_handlers(self, event_type: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Get information about registered handlers"""
        if event_type:
            handlers = self.subscriptions.get(event_type, [])
            return {
                event_type: [handler.get_handler_info() for _, handler in handlers]
            }
        
        result = {}
        for et, handlers in self.subscriptions.items():
            result[et] = [handler.get_handler_info() for _, handler in handlers]
        
        return result


# Specific Event Types for the AI Detector System

@dataclass
class DetectionEvent(Event):
    """Event for detection operations"""
    def __init__(self, event_type: str, text: str, result: Dict[str, Any], 
                 user_id: Optional[str] = None, **kwargs):
        super().__init__(
            event_type=event_type,
            data={
                "text": text,
                "result": result,
                "user_id": user_id
            },
            **kwargs
        )


@dataclass
class TrainingEvent(Event):
    """Event for training operations"""
    def __init__(self, event_type: str, model_info: Dict[str, Any], 
                 metrics: Dict[str, Any] = None, **kwargs):
        super().__init__(
            event_type=event_type,
            data={
                "model_info": model_info,
                "metrics": metrics or {}
            },
            **kwargs
        )


@dataclass
class SystemEvent(Event):
    """Event for system operations"""
    def __init__(self, event_type: str, component: str, status: str, 
                 details: Dict[str, Any] = None, **kwargs):
        super().__init__(
            event_type=event_type,
            data={
                "component": component,
                "status": status,
                "details": details or {}
            },
            **kwargs
        )


# Event Type Constants
class EventTypes:
    """Constants for event types"""
    
    # Detection Events
    DETECTION_STARTED = "detection.started"
    DETECTION_COMPLETED = "detection.completed"
    DETECTION_FAILED = "detection.failed"
    
    # Training Events
    TRAINING_STARTED = "training.started"
    TRAINING_PROGRESS = "training.progress"
    TRAINING_COMPLETED = "training.completed"
    TRAINING_FAILED = "training.failed"
    MODEL_UPDATED = "model.updated"
    
    # Data Collection Events
    DATA_COLLECTED = "data.collected"
    DATA_PROCESSED = "data.processed"
    DATA_VALIDATED = "data.validated"
    
    # System Events
    SERVICE_STARTED = "system.service_started"
    SERVICE_STOPPED = "system.service_stopped"
    ERROR_OCCURRED = "system.error"
    HEALTH_CHECK = "system.health_check"
    
    # User Events
    USER_ACTION = "user.action"
    SETTINGS_CHANGED = "user.settings_changed"


# Global event bus instance
event_bus = EventBus()


# Decorator for easy event publishing
def publish_event(event_type: str, source: str = None):
    """Decorator to automatically publish events"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            
            # Create event
            event_source = source or getattr(args[0], '__class__.__name__', 'unknown') if args else 'unknown'
            event = Event(
                event_type=event_type,
                data={"args": args, "kwargs": kwargs, "result": result},
                source=event_source
            )
            
            # Publish event
            await event_bus.publish(event)
            
            return result
        return wrapper
    return decorator


__all__ = [
    'Event', 'EventPriority', 'IEventHandler', 'BaseEventHandler', 'FunctionEventHandler',
    'IEventBus', 'EventBus', 'DetectionEvent', 'TrainingEvent', 'SystemEvent',
    'EventTypes', 'event_bus', 'publish_event'
]