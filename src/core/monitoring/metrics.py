"""
Metrics collection system for the AI Detector.

Provides comprehensive metrics collection, aggregation, and export
for monitoring system performance and behavior.
"""

import time
import threading
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import json
import statistics
from abc import ABC, abstractmethod


class MetricType(Enum):
    """Types of metrics supported."""
    
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


@dataclass
class MetricValue:
    """Container for metric values with metadata."""
    
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "labels": self.labels
        }


class Metric(ABC):
    """
    Abstract base class for metrics.
    
    Provides common functionality for all metric types.
    """
    
    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[Dict[str, str]] = None,
        unit: Optional[str] = None
    ):
        """
        Initialize metric.
        
        Args:
            name: Metric name
            description: Metric description
            labels: Static labels for the metric
            unit: Unit of measurement
        """
        self.name = name
        self.description = description
        self.labels = labels or {}
        self.unit = unit
        self._lock = threading.Lock()
    
    @abstractmethod
    def get_value(self) -> Any:
        """Get current metric value."""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset metric to initial state."""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary format."""
        return {
            "name": self.name,
            "type": self.__class__.__name__.lower(),
            "description": self.description,
            "labels": self.labels,
            "unit": self.unit,
            "value": self.get_value()
        }


class Counter(Metric):
    """
    Counter metric that only increases.
    
    Used for counting events, requests, errors, etc.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize counter."""
        super().__init__(*args, **kwargs)
        self._value = 0
    
    def increment(self, value: float = 1, labels: Optional[Dict[str, str]] = None):
        """
        Increment counter.
        
        Args:
            value: Amount to increment (must be positive)
            labels: Additional labels for this increment
        """
        if value < 0:
            raise ValueError("Counter can only be incremented with positive values")
        
        with self._lock:
            self._value += value
    
    def get_value(self) -> float:
        """Get current counter value."""
        with self._lock:
            return self._value
    
    def reset(self):
        """Reset counter to zero."""
        with self._lock:
            self._value = 0


class Gauge(Metric):
    """
    Gauge metric that can increase or decrease.
    
    Used for measuring current values like memory usage, queue size, etc.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize gauge."""
        super().__init__(*args, **kwargs)
        self._value = 0
    
    def set(self, value: float, labels: Optional[Dict[str, str]] = None):
        """
        Set gauge value.
        
        Args:
            value: New gauge value
            labels: Additional labels
        """
        with self._lock:
            self._value = value
    
    def increment(self, value: float = 1):
        """Increment gauge value."""
        with self._lock:
            self._value += value
    
    def decrement(self, value: float = 1):
        """Decrement gauge value."""
        with self._lock:
            self._value -= value
    
    def get_value(self) -> float:
        """Get current gauge value."""
        with self._lock:
            return self._value
    
    def reset(self):
        """Reset gauge to zero."""
        with self._lock:
            self._value = 0


class Histogram(Metric):
    """
    Histogram metric for tracking value distributions.
    
    Used for measuring request durations, response sizes, etc.
    """
    
    def __init__(
        self,
        *args,
        buckets: Optional[List[float]] = None,
        max_samples: int = 10000,
        **kwargs
    ):
        """
        Initialize histogram.
        
        Args:
            buckets: Bucket boundaries for histogram
            max_samples: Maximum number of samples to keep
        """
        super().__init__(*args, **kwargs)
        self.buckets = buckets or [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
        self.max_samples = max_samples
        self._samples = deque(maxlen=max_samples)
        self._bucket_counts = defaultdict(int)
        self._sum = 0
        self._count = 0
    
    def observe(self, value: float, labels: Optional[Dict[str, str]] = None):
        """
        Record an observation.
        
        Args:
            value: Observed value
            labels: Additional labels
        """
        with self._lock:
            self._samples.append(value)
            self._sum += value
            self._count += 1
            
            # Update bucket counts
            for bucket in self.buckets:
                if value <= bucket:
                    self._bucket_counts[bucket] += 1
                    break
            else:
                # Value exceeds all buckets
                self._bucket_counts[float('inf')] += 1
    
    def get_value(self) -> Dict[str, Any]:
        """Get histogram statistics."""
        with self._lock:
            if not self._samples:
                return {
                    "count": 0,
                    "sum": 0,
                    "mean": 0,
                    "min": 0,
                    "max": 0,
                    "percentiles": {}
                }
            
            samples = list(self._samples)
            sorted_samples = sorted(samples)
            
            return {
                "count": self._count,
                "sum": self._sum,
                "mean": self._sum / self._count if self._count > 0 else 0,
                "min": min(samples),
                "max": max(samples),
                "stddev": statistics.stdev(samples) if len(samples) > 1 else 0,
                "percentiles": {
                    "p50": self._percentile(sorted_samples, 0.5),
                    "p75": self._percentile(sorted_samples, 0.75),
                    "p90": self._percentile(sorted_samples, 0.9),
                    "p95": self._percentile(sorted_samples, 0.95),
                    "p99": self._percentile(sorted_samples, 0.99)
                },
                "buckets": dict(self._bucket_counts)
            }
    
    def _percentile(self, sorted_samples: List[float], p: float) -> float:
        """Calculate percentile from sorted samples."""
        if not sorted_samples:
            return 0
        
        index = int(len(sorted_samples) * p)
        if index >= len(sorted_samples):
            index = len(sorted_samples) - 1
        
        return sorted_samples[index]
    
    def reset(self):
        """Reset histogram."""
        with self._lock:
            self._samples.clear()
            self._bucket_counts.clear()
            self._sum = 0
            self._count = 0


class Timer(Histogram):
    """
    Timer metric for measuring durations.
    
    Specialized histogram for timing operations.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize timer with appropriate buckets for time measurements."""
        # Default buckets in seconds
        kwargs.setdefault('buckets', [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10])
        kwargs.setdefault('unit', 'seconds')
        super().__init__(*args, **kwargs)
    
    def time(self, labels: Optional[Dict[str, str]] = None):
        """
        Context manager for timing operations.
        
        Args:
            labels: Additional labels for this timing
            
        Returns:
            Timer context manager
        """
        class TimerContext:
            def __init__(context_self, timer: Timer):
                context_self.timer = timer
                context_self.start_time = None
                context_self.labels = labels
            
            def __enter__(context_self):
                context_self.start_time = time.perf_counter()
                return context_self
            
            def __exit__(context_self, exc_type, exc_val, exc_tb):
                duration = time.perf_counter() - context_self.start_time
                context_self.timer.observe(duration, context_self.labels)
        
        return TimerContext(self)
    
    def time_function(self, func: Callable) -> Callable:
        """
        Decorator for timing function execution.
        
        Args:
            func: Function to time
            
        Returns:
            Wrapped function
        """
        def wrapper(*args, **kwargs):
            with self.time():
                return func(*args, **kwargs)
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper


class Rate(Metric):
    """
    Rate metric for measuring rates over time windows.
    
    Used for requests per second, errors per minute, etc.
    """
    
    def __init__(
        self,
        *args,
        window_seconds: int = 60,
        **kwargs
    ):
        """
        Initialize rate metric.
        
        Args:
            window_seconds: Time window for rate calculation
        """
        super().__init__(*args, **kwargs)
        self.window_seconds = window_seconds
        self._events = deque()
    
    def record(self, count: float = 1, labels: Optional[Dict[str, str]] = None):
        """
        Record events for rate calculation.
        
        Args:
            count: Number of events
            labels: Additional labels
        """
        now = time.time()
        with self._lock:
            self._events.append((now, count))
            self._cleanup()
    
    def _cleanup(self):
        """Remove old events outside the window."""
        cutoff = time.time() - self.window_seconds
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()
    
    def get_value(self) -> float:
        """Get current rate (events per second)."""
        with self._lock:
            self._cleanup()
            
            if not self._events:
                return 0.0
            
            total_events = sum(count for _, count in self._events)
            time_span = time.time() - self._events[0][0] if self._events else 1
            
            return total_events / max(time_span, 1)
    
    def reset(self):
        """Reset rate metric."""
        with self._lock:
            self._events.clear()


class MetricsCollector:
    """
    Central metrics collection and management system.
    
    Collects, aggregates, and exports metrics from all components.
    """
    
    def __init__(self):
        """Initialize metrics collector."""
        self._metrics: Dict[str, Metric] = {}
        self._lock = threading.Lock()
        self._export_callbacks: List[Callable] = []
        
        # Register default system metrics
        self._register_default_metrics()
    
    def _register_default_metrics(self):
        """Register default system metrics."""
        # Request metrics
        self.register_counter("requests_total", "Total number of requests")
        self.register_counter("errors_total", "Total number of errors")
        self.register_histogram("request_duration_seconds", "Request duration in seconds")
        
        # Detection metrics
        self.register_counter("detections_total", "Total number of detections")
        self.register_histogram("detection_duration_seconds", "Detection duration in seconds")
        self.register_gauge("detection_confidence", "Average detection confidence")
        
        # System metrics
        self.register_gauge("memory_usage_bytes", "Memory usage in bytes")
        self.register_gauge("cpu_usage_percent", "CPU usage percentage")
        self.register_gauge("active_connections", "Number of active connections")
        
        # Performance metrics
        self.register_rate("requests_per_second", "Request rate per second")
        self.register_rate("errors_per_minute", "Error rate per minute", window_seconds=60)
    
    def register_counter(
        self,
        name: str,
        description: str = "",
        labels: Optional[Dict[str, str]] = None
    ) -> Counter:
        """Register a counter metric."""
        counter = Counter(name, description, labels)
        with self._lock:
            self._metrics[name] = counter
        return counter
    
    def register_gauge(
        self,
        name: str,
        description: str = "",
        labels: Optional[Dict[str, str]] = None
    ) -> Gauge:
        """Register a gauge metric."""
        gauge = Gauge(name, description, labels)
        with self._lock:
            self._metrics[name] = gauge
        return gauge
    
    def register_histogram(
        self,
        name: str,
        description: str = "",
        buckets: Optional[List[float]] = None,
        labels: Optional[Dict[str, str]] = None
    ) -> Histogram:
        """Register a histogram metric."""
        histogram = Histogram(name, description, labels, buckets=buckets)
        with self._lock:
            self._metrics[name] = histogram
        return histogram
    
    def register_timer(
        self,
        name: str,
        description: str = "",
        labels: Optional[Dict[str, str]] = None
    ) -> Timer:
        """Register a timer metric."""
        timer = Timer(name, description, labels)
        with self._lock:
            self._metrics[name] = timer
        return timer
    
    def register_rate(
        self,
        name: str,
        description: str = "",
        window_seconds: int = 60,
        labels: Optional[Dict[str, str]] = None
    ) -> Rate:
        """Register a rate metric."""
        rate = Rate(name, description, labels, window_seconds=window_seconds)
        with self._lock:
            self._metrics[name] = rate
        return rate
    
    def get_metric(self, name: str) -> Optional[Metric]:
        """Get a metric by name."""
        with self._lock:
            return self._metrics.get(name)
    
    def increment_counter(
        self,
        name: str,
        value: float = 1,
        labels: Optional[Dict[str, str]] = None
    ):
        """Increment a counter metric."""
        metric = self.get_metric(name)
        if isinstance(metric, Counter):
            metric.increment(value, labels)
    
    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ):
        """Set a gauge metric value."""
        metric = self.get_metric(name)
        if isinstance(metric, Gauge):
            metric.set(value, labels)
    
    def observe_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ):
        """Record a histogram observation."""
        metric = self.get_metric(name)
        if isinstance(metric, Histogram):
            metric.observe(value, labels)
    
    def record_rate(
        self,
        name: str,
        count: float = 1,
        labels: Optional[Dict[str, str]] = None
    ):
        """Record rate events."""
        metric = self.get_metric(name)
        if isinstance(metric, Rate):
            metric.record(count, labels)
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics as dictionary."""
        with self._lock:
            return {
                name: metric.to_dict()
                for name, metric in self._metrics.items()
            }
    
    def export_metrics(self, format: str = "json") -> Union[str, Dict[str, Any]]:
        """
        Export metrics in specified format.
        
        Args:
            format: Export format (json, prometheus, etc.)
            
        Returns:
            Exported metrics
        """
        metrics = self.get_all_metrics()
        
        if format == "json":
            return json.dumps(metrics, indent=2, default=str)
        elif format == "prometheus":
            return self._export_prometheus(metrics)
        else:
            return metrics
    
    def _export_prometheus(self, metrics: Dict[str, Any]) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        for name, metric_data in metrics.items():
            # Add HELP and TYPE comments
            if metric_data.get('description'):
                lines.append(f"# HELP {name} {metric_data['description']}")
            lines.append(f"# TYPE {name} {metric_data['type']}")
            
            # Format value based on type
            value = metric_data.get('value')
            if isinstance(value, dict):
                # For histograms, export multiple metrics
                if 'count' in value:
                    lines.append(f"{name}_count {value['count']}")
                if 'sum' in value:
                    lines.append(f"{name}_sum {value['sum']}")
                if 'buckets' in value:
                    for bucket, count in value['buckets'].items():
                        if bucket == float('inf'):
                            lines.append(f'{name}_bucket{{le="+Inf"}} {count}')
                        else:
                            lines.append(f'{name}_bucket{{le="{bucket}"}} {count}')
            else:
                # Simple value
                lines.append(f"{name} {value}")
        
        return "\n".join(lines)
    
    def reset_all_metrics(self):
        """Reset all metrics to initial state."""
        with self._lock:
            for metric in self._metrics.values():
                metric.reset()
    
    def register_export_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Register a callback for metric export.
        
        Args:
            callback: Function to call with metrics data
        """
        self._export_callbacks.append(callback)
    
    def trigger_export(self):
        """Trigger export callbacks with current metrics."""
        metrics = self.get_all_metrics()
        for callback in self._export_callbacks:
            try:
                callback(metrics)
            except Exception as e:
                # Log error but don't fail
                print(f"Export callback failed: {e}")


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector