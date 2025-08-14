"""
Distributed tracing and telemetry for the AI Detector.

Provides OpenTelemetry-compatible tracing for request tracking
across system components.
"""

import time
import uuid
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from contextvars import ContextVar
import json
import logging

from .logger import get_logger
from .metrics import get_metrics_collector


logger = get_logger(__name__)


class SpanKind(Enum):
    """Types of spans in distributed tracing."""
    
    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"


class SpanStatus(Enum):
    """Status of a span."""
    
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


@dataclass
class SpanContext:
    """Context for distributed tracing."""
    
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    trace_flags: int = 0
    trace_state: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "trace_flags": self.trace_flags,
            "trace_state": self.trace_state
        }
    
    @classmethod
    def generate(cls, parent: Optional['SpanContext'] = None) -> 'SpanContext':
        """Generate a new span context."""
        if parent:
            return cls(
                trace_id=parent.trace_id,
                span_id=str(uuid.uuid4()),
                parent_span_id=parent.span_id,
                trace_flags=parent.trace_flags,
                trace_state=parent.trace_state.copy()
            )
        else:
            return cls(
                trace_id=str(uuid.uuid4()),
                span_id=str(uuid.uuid4())
            )


@dataclass
class Span:
    """
    Represents a span in distributed tracing.
    
    Tracks a single operation within a larger trace.
    """
    
    name: str
    context: SpanContext
    kind: SpanKind = SpanKind.INTERNAL
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    status: SpanStatus = SpanStatus.UNSET
    status_message: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    links: List[SpanContext] = field(default_factory=list)
    
    def set_attribute(self, key: str, value: Any):
        """Set a span attribute."""
        self.attributes[key] = value
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add an event to the span."""
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {}
        })
    
    def set_status(self, status: SpanStatus, message: Optional[str] = None):
        """Set span status."""
        self.status = status
        self.status_message = message
    
    def end(self, end_time: Optional[float] = None):
        """End the span."""
        self.end_time = end_time or time.time()
    
    def duration_ms(self) -> Optional[float]:
        """Get span duration in milliseconds."""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary format."""
        return {
            "name": self.name,
            "context": self.context.to_dict(),
            "kind": self.kind.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms(),
            "status": self.status.value,
            "status_message": self.status_message,
            "attributes": self.attributes,
            "events": self.events,
            "links": [link.to_dict() for link in self.links]
        }


class Tracer:
    """
    Creates and manages spans for distributed tracing.
    
    Provides context propagation and span lifecycle management.
    """
    
    def __init__(
        self,
        name: str,
        telemetry_client: Optional['TelemetryClient'] = None
    ):
        """
        Initialize tracer.
        
        Args:
            name: Tracer name (usually module or component name)
            telemetry_client: Telemetry client for span export
        """
        self.name = name
        self.telemetry_client = telemetry_client or get_telemetry_client()
    
    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
        links: Optional[List[SpanContext]] = None,
        parent: Optional[Span] = None
    ) -> Span:
        """
        Start a new span.
        
        Args:
            name: Span name
            kind: Span kind
            attributes: Initial attributes
            links: Links to other spans
            parent: Parent span
            
        Returns:
            New span instance
        """
        # Get or create context
        if parent:
            context = SpanContext.generate(parent.context)
        else:
            current_context = current_span_context.get()
            if current_context:
                context = SpanContext.generate(current_context)
            else:
                context = SpanContext.generate()
        
        # Create span
        span = Span(
            name=name,
            context=context,
            kind=kind,
            attributes=attributes or {},
            links=links or []
        )
        
        # Add tracer name
        span.set_attribute("tracer.name", self.name)
        
        # Register with telemetry client
        self.telemetry_client.register_span(span)
        
        return span
    
    def start_as_current_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for creating a span as the current span.
        
        Args:
            name: Span name
            kind: Span kind
            attributes: Initial attributes
            
        Returns:
            Context manager that yields the span
        """
        class SpanContextManager:
            def __init__(context_self, tracer: Tracer):
                context_self.tracer = tracer
                context_self.span = None
                context_self.token = None
            
            def __enter__(context_self) -> Span:
                context_self.span = self.start_span(name, kind, attributes)
                context_self.token = current_span_context.set(context_self.span.context)
                return context_self.span
            
            def __exit__(context_self, exc_type, exc_val, exc_tb):
                if context_self.span:
                    if exc_type:
                        context_self.span.set_status(
                            SpanStatus.ERROR,
                            f"{exc_type.__name__}: {exc_val}"
                        )
                        context_self.span.add_event(
                            "exception",
                            {
                                "exception.type": exc_type.__name__,
                                "exception.message": str(exc_val)
                            }
                        )
                    else:
                        context_self.span.set_status(SpanStatus.OK)
                    
                    context_self.span.end()
                    self.telemetry_client.export_span(context_self.span)
                
                if context_self.token:
                    current_span_context.reset(context_self.token)
        
        return SpanContextManager(self)


class TelemetryClient:
    """
    Central telemetry collection and export client.
    
    Manages spans, metrics, and logs for observability.
    """
    
    def __init__(self):
        """Initialize telemetry client."""
        self._spans: List[Span] = []
        self._active_spans: Dict[str, Span] = {}
        self._exporters: List[Callable[[Span], None]] = []
        self._metrics_collector = get_metrics_collector()
        
        # Register default exporters
        self._register_default_exporters()
    
    def _register_default_exporters(self):
        """Register default span exporters."""
        # Console exporter for debugging
        if logger.logger.level <= logging.DEBUG:
            self.register_exporter(self._console_exporter)
        
        # Metrics exporter
        self.register_exporter(self._metrics_exporter)
    
    def register_span(self, span: Span):
        """
        Register a new span.
        
        Args:
            span: Span to register
        """
        span_key = f"{span.context.trace_id}:{span.context.span_id}"
        self._active_spans[span_key] = span
        
        # Log span start
        logger.debug(
            f"Span started",
            span_name=span.name,
            trace_id=span.context.trace_id,
            span_id=span.context.span_id,
            parent_span_id=span.context.parent_span_id
        )
    
    def export_span(self, span: Span):
        """
        Export a completed span.
        
        Args:
            span: Completed span to export
        """
        # Remove from active spans
        span_key = f"{span.context.trace_id}:{span.context.span_id}"
        self._active_spans.pop(span_key, None)
        
        # Store completed span
        self._spans.append(span)
        
        # Export through registered exporters
        for exporter in self._exporters:
            try:
                exporter(span)
            except Exception as e:
                logger.error(f"Span export failed: {e}")
    
    def register_exporter(self, exporter: Callable[[Span], None]):
        """
        Register a span exporter.
        
        Args:
            exporter: Function to export spans
        """
        self._exporters.append(exporter)
    
    def _console_exporter(self, span: Span):
        """Export span to console for debugging."""
        logger.debug(
            f"Span completed",
            span_data=span.to_dict()
        )
    
    def _metrics_exporter(self, span: Span):
        """Export span metrics."""
        if span.duration_ms():
            # Record operation duration
            histogram_name = f"{span.name.replace('.', '_')}_duration_ms"
            histogram = self._metrics_collector.get_metric(histogram_name)
            
            if not histogram:
                histogram = self._metrics_collector.register_histogram(
                    histogram_name,
                    f"Duration of {span.name} operations in milliseconds"
                )
            
            histogram.observe(span.duration_ms())
            
            # Record success/failure
            if span.status == SpanStatus.ERROR:
                self._metrics_collector.increment_counter(
                    "span_errors_total",
                    labels={"span_name": span.name}
                )
            else:
                self._metrics_collector.increment_counter(
                    "span_success_total",
                    labels={"span_name": span.name}
                )
    
    def get_active_spans(self) -> List[Span]:
        """Get list of currently active spans."""
        return list(self._active_spans.values())
    
    def get_trace(self, trace_id: str) -> List[Span]:
        """
        Get all spans for a trace ID.
        
        Args:
            trace_id: Trace identifier
            
        Returns:
            List of spans in the trace
        """
        return [
            span for span in self._spans
            if span.context.trace_id == trace_id
        ]
    
    def export_traces(self, format: str = "json") -> str:
        """
        Export all traces in specified format.
        
        Args:
            format: Export format (json, jaeger, etc.)
            
        Returns:
            Exported traces
        """
        if format == "json":
            traces = {}
            for span in self._spans:
                trace_id = span.context.trace_id
                if trace_id not in traces:
                    traces[trace_id] = []
                traces[trace_id].append(span.to_dict())
            
            return json.dumps(traces, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def clear_traces(self):
        """Clear all stored traces."""
        self._spans.clear()
        self._active_spans.clear()


# Context variable for current span
current_span_context: ContextVar[Optional[SpanContext]] = ContextVar(
    'current_span_context',
    default=None
)


# Global telemetry client
_telemetry_client: Optional[TelemetryClient] = None


def get_telemetry_client() -> TelemetryClient:
    """Get global telemetry client instance."""
    global _telemetry_client
    if _telemetry_client is None:
        _telemetry_client = TelemetryClient()
    return _telemetry_client


def setup_telemetry():
    """Set up global telemetry configuration."""
    client = get_telemetry_client()
    
    # Add any additional setup here
    logger.info("Telemetry system initialized")
    
    return client


def get_tracer(name: str) -> Tracer:
    """
    Get a tracer for the specified component.
    
    Args:
        name: Component name
        
    Returns:
        Tracer instance
    """
    return Tracer(name)


def trace(
    name: Optional[str] = None,
    kind: SpanKind = SpanKind.INTERNAL
):
    """
    Decorator for automatic span creation around functions.
    
    Args:
        name: Span name (defaults to function name)
        kind: Span kind
        
    Returns:
        Decorated function
    """
    def decorator(func):
        span_name = name or f"{func.__module__}.{func.__name__}"
        tracer = get_tracer(func.__module__)
        
        def wrapper(*args, **kwargs):
            with tracer.start_as_current_span(span_name, kind=kind) as span:
                # Add function arguments as attributes
                span.set_attribute("function.module", func.__module__)
                span.set_attribute("function.name", func.__name__)
                
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.set_status(SpanStatus.ERROR, str(e))
                    raise
        
        async def async_wrapper(*args, **kwargs):
            with tracer.start_as_current_span(span_name, kind=kind) as span:
                # Add function arguments as attributes
                span.set_attribute("function.module", func.__module__)
                span.set_attribute("function.name", func.__name__)
                
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.set_status(SpanStatus.ERROR, str(e))
                    raise
        
        # Return appropriate wrapper
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator