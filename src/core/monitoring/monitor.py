"""
System monitoring components for the AI Detector.

Provides real-time monitoring of system health, performance,
and resource utilization.
"""

import time
import threading
import psutil
import asyncio
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import logging

from .metrics import get_metrics_collector, MetricsCollector
from .logger import get_logger


logger = get_logger(__name__)


class HealthStatus(Enum):
    """System health status levels."""
    
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class MonitoringConfig:
    """Configuration for system monitoring."""
    
    # Monitoring intervals
    health_check_interval_seconds: int = 30
    metrics_collection_interval_seconds: int = 10
    performance_check_interval_seconds: int = 60
    
    # Resource thresholds
    cpu_threshold_percent: float = 80.0
    memory_threshold_percent: float = 85.0
    disk_threshold_percent: float = 90.0
    
    # Performance thresholds
    response_time_threshold_ms: float = 2000
    error_rate_threshold_percent: float = 5.0
    
    # Monitoring features
    enable_health_checks: bool = True
    enable_performance_monitoring: bool = True
    enable_resource_monitoring: bool = True
    enable_alerts: bool = True
    
    # Export configuration
    export_metrics: bool = True
    export_interval_seconds: int = 300


@dataclass
class HealthCheck:
    """Individual health check result."""
    
    name: str
    status: HealthStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }


class SystemMonitor:
    """
    Main system monitoring component.
    
    Coordinates health checks, metrics collection, and performance monitoring.
    """
    
    def __init__(
        self,
        config: Optional[MonitoringConfig] = None,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        """
        Initialize system monitor.
        
        Args:
            config: Monitoring configuration
            metrics_collector: Metrics collector instance
        """
        self.config = config or MonitoringConfig()
        self.metrics = metrics_collector or get_metrics_collector()
        
        self._health_checks: Dict[str, Callable[[], HealthCheck]] = {}
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_health_status = HealthStatus.HEALTHY
        
        # Register default health checks
        self._register_default_health_checks()
    
    def _register_default_health_checks(self):
        """Register default system health checks."""
        self.register_health_check("cpu", self._check_cpu_health)
        self.register_health_check("memory", self._check_memory_health)
        self.register_health_check("disk", self._check_disk_health)
        self.register_health_check("api", self._check_api_health)
        self.register_health_check("database", self._check_database_health)
    
    def register_health_check(
        self,
        name: str,
        check_function: Callable[[], HealthCheck]
    ):
        """
        Register a health check function.
        
        Args:
            name: Health check name
            check_function: Function that returns HealthCheck
        """
        self._health_checks[name] = check_function
    
    def _check_cpu_health(self) -> HealthCheck:
        """Check CPU utilization health."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            
            if cpu_percent < self.config.cpu_threshold_percent:
                status = HealthStatus.HEALTHY
                message = f"CPU usage: {cpu_percent:.1f}%"
            elif cpu_percent < self.config.cpu_threshold_percent * 1.2:
                status = HealthStatus.DEGRADED
                message = f"High CPU usage: {cpu_percent:.1f}%"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Critical CPU usage: {cpu_percent:.1f}%"
            
            # Update metrics
            self.metrics.set_gauge("cpu_usage_percent", cpu_percent)
            
            return HealthCheck(
                name="cpu",
                status=status,
                message=message,
                details={
                    "cpu_percent": cpu_percent,
                    "cpu_count": psutil.cpu_count(),
                    "threshold": self.config.cpu_threshold_percent
                }
            )
        except Exception as e:
            logger.error(f"CPU health check failed: {e}")
            return HealthCheck(
                name="cpu",
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to check CPU: {e}"
            )
    
    def _check_memory_health(self) -> HealthCheck:
        """Check memory utilization health."""
        try:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            if memory_percent < self.config.memory_threshold_percent:
                status = HealthStatus.HEALTHY
                message = f"Memory usage: {memory_percent:.1f}%"
            elif memory_percent < self.config.memory_threshold_percent * 1.1:
                status = HealthStatus.DEGRADED
                message = f"High memory usage: {memory_percent:.1f}%"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Critical memory usage: {memory_percent:.1f}%"
            
            # Update metrics
            self.metrics.set_gauge("memory_usage_bytes", memory.used)
            
            return HealthCheck(
                name="memory",
                status=status,
                message=message,
                details={
                    "memory_percent": memory_percent,
                    "available_mb": memory.available / (1024 * 1024),
                    "total_mb": memory.total / (1024 * 1024),
                    "threshold": self.config.memory_threshold_percent
                }
            )
        except Exception as e:
            logger.error(f"Memory health check failed: {e}")
            return HealthCheck(
                name="memory",
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to check memory: {e}"
            )
    
    def _check_disk_health(self) -> HealthCheck:
        """Check disk utilization health."""
        try:
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            if disk_percent < self.config.disk_threshold_percent:
                status = HealthStatus.HEALTHY
                message = f"Disk usage: {disk_percent:.1f}%"
            elif disk_percent < self.config.disk_threshold_percent * 1.05:
                status = HealthStatus.DEGRADED
                message = f"High disk usage: {disk_percent:.1f}%"
            else:
                status = HealthStatus.CRITICAL
                message = f"Critical disk usage: {disk_percent:.1f}%"
            
            return HealthCheck(
                name="disk",
                status=status,
                message=message,
                details={
                    "disk_percent": disk_percent,
                    "free_gb": disk.free / (1024 ** 3),
                    "total_gb": disk.total / (1024 ** 3),
                    "threshold": self.config.disk_threshold_percent
                }
            )
        except Exception as e:
            logger.error(f"Disk health check failed: {e}")
            return HealthCheck(
                name="disk",
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to check disk: {e}"
            )
    
    def _check_api_health(self) -> HealthCheck:
        """Check API service health."""
        # This would check actual API availability
        # For now, return a placeholder
        return HealthCheck(
            name="api",
            status=HealthStatus.HEALTHY,
            message="API service is running"
        )
    
    def _check_database_health(self) -> HealthCheck:
        """Check database connection health."""
        # This would check actual database connectivity
        # For now, return a placeholder
        return HealthCheck(
            name="database",
            status=HealthStatus.HEALTHY,
            message="Database connection is healthy"
        )
    
    def run_health_checks(self) -> Dict[str, HealthCheck]:
        """
        Run all registered health checks.
        
        Returns:
            Dictionary of health check results
        """
        results = {}
        
        for name, check_function in self._health_checks.items():
            try:
                results[name] = check_function()
            except Exception as e:
                logger.error(f"Health check '{name}' failed: {e}")
                results[name] = HealthCheck(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed: {e}"
                )
        
        # Determine overall health status
        self._update_overall_health(results)
        
        return results
    
    def _update_overall_health(self, health_checks: Dict[str, HealthCheck]):
        """Update overall system health status."""
        statuses = [check.status for check in health_checks.values()]
        
        if all(s == HealthStatus.HEALTHY for s in statuses):
            overall_status = HealthStatus.HEALTHY
        elif any(s == HealthStatus.CRITICAL for s in statuses):
            overall_status = HealthStatus.CRITICAL
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall_status = HealthStatus.UNHEALTHY
        else:
            overall_status = HealthStatus.DEGRADED
        
        if overall_status != self._last_health_status:
            logger.info(f"System health status changed: {self._last_health_status.value} -> {overall_status.value}")
            self._last_health_status = overall_status
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get current system health status.
        
        Returns:
            Health status dictionary
        """
        health_checks = self.run_health_checks()
        
        return {
            "status": self._last_health_status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {
                name: check.to_dict()
                for name, check in health_checks.items()
            }
        }
    
    def start_monitoring(self):
        """Start background monitoring thread."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            logger.warning("Monitoring already running")
            return
        
        self._stop_event.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self._monitoring_thread.start()
        logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self._stop_event.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        logger.info("System monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop running in background thread."""
        last_health_check = time.time()
        last_metrics_collection = time.time()
        last_export = time.time()
        
        while not self._stop_event.is_set():
            now = time.time()
            
            # Run health checks
            if self.config.enable_health_checks:
                if now - last_health_check >= self.config.health_check_interval_seconds:
                    self.run_health_checks()
                    last_health_check = now
            
            # Collect metrics
            if self.config.enable_resource_monitoring:
                if now - last_metrics_collection >= self.config.metrics_collection_interval_seconds:
                    self._collect_system_metrics()
                    last_metrics_collection = now
            
            # Export metrics
            if self.config.export_metrics:
                if now - last_export >= self.config.export_interval_seconds:
                    self.metrics.trigger_export()
                    last_export = now
            
            # Sleep for a short interval
            time.sleep(1)
    
    def _collect_system_metrics(self):
        """Collect system resource metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.metrics.set_gauge("cpu_usage_percent", cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.metrics.set_gauge("memory_usage_bytes", memory.used)
            self.metrics.set_gauge("memory_available_bytes", memory.available)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            self.metrics.set_gauge("disk_usage_bytes", disk.used)
            self.metrics.set_gauge("disk_free_bytes", disk.free)
            
            # Network metrics
            net_io = psutil.net_io_counters()
            self.metrics.set_gauge("network_bytes_sent", net_io.bytes_sent)
            self.metrics.set_gauge("network_bytes_received", net_io.bytes_recv)
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")


class PerformanceMonitor:
    """
    Performance monitoring for application operations.
    
    Tracks operation latencies, throughput, and performance trends.
    """
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        """
        Initialize performance monitor.
        
        Args:
            metrics_collector: Metrics collector instance
        """
        self.metrics = metrics_collector or get_metrics_collector()
        self._operation_timers: Dict[str, float] = {}
    
    def start_operation(self, operation_name: str) -> str:
        """
        Start timing an operation.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Operation ID for tracking
        """
        operation_id = f"{operation_name}_{time.time()}"
        self._operation_timers[operation_id] = time.perf_counter()
        return operation_id
    
    def end_operation(
        self,
        operation_id: str,
        success: bool = True,
        labels: Optional[Dict[str, str]] = None
    ):
        """
        End timing an operation and record metrics.
        
        Args:
            operation_id: Operation ID from start_operation
            success: Whether operation succeeded
            labels: Additional labels
        """
        if operation_id not in self._operation_timers:
            logger.warning(f"Unknown operation ID: {operation_id}")
            return
        
        start_time = self._operation_timers.pop(operation_id)
        duration = time.perf_counter() - start_time
        
        # Extract operation name from ID
        operation_name = operation_id.rsplit('_', 1)[0]
        
        # Record metrics
        timer = self.metrics.get_metric(f"{operation_name}_duration_seconds")
        if timer:
            timer.observe(duration, labels)
        
        # Log performance
        logger.info(
            f"Operation completed",
            operation=operation_name,
            duration_ms=duration * 1000,
            success=success,
            labels=labels
        )
    
    def record_throughput(
        self,
        operation_name: str,
        count: int = 1,
        labels: Optional[Dict[str, str]] = None
    ):
        """
        Record throughput metrics.
        
        Args:
            operation_name: Name of the operation
            count: Number of operations
            labels: Additional labels
        """
        rate_metric = self.metrics.get_metric(f"{operation_name}_per_second")
        if rate_metric:
            rate_metric.record(count, labels)


class HealthMonitor:
    """
    Application health monitoring.
    
    Provides health endpoints and liveness/readiness checks.
    """
    
    def __init__(self, system_monitor: Optional[SystemMonitor] = None):
        """
        Initialize health monitor.
        
        Args:
            system_monitor: System monitor instance
        """
        self.system_monitor = system_monitor or SystemMonitor()
        self._readiness_checks: Dict[str, Callable[[], bool]] = {}
        self._liveness_checks: Dict[str, Callable[[], bool]] = {}
    
    def register_readiness_check(
        self,
        name: str,
        check_function: Callable[[], bool]
    ):
        """Register a readiness check."""
        self._readiness_checks[name] = check_function
    
    def register_liveness_check(
        self,
        name: str,
        check_function: Callable[[], bool]
    ):
        """Register a liveness check."""
        self._liveness_checks[name] = check_function
    
    def is_ready(self) -> bool:
        """
        Check if application is ready to serve requests.
        
        Returns:
            True if ready, False otherwise
        """
        for name, check in self._readiness_checks.items():
            try:
                if not check():
                    logger.warning(f"Readiness check failed: {name}")
                    return False
            except Exception as e:
                logger.error(f"Readiness check error: {name}: {e}")
                return False
        
        return True
    
    def is_alive(self) -> bool:
        """
        Check if application is alive.
        
        Returns:
            True if alive, False otherwise
        """
        for name, check in self._liveness_checks.items():
            try:
                if not check():
                    logger.warning(f"Liveness check failed: {name}")
                    return False
            except Exception as e:
                logger.error(f"Liveness check error: {name}: {e}")
                return False
        
        return True
    
    def get_health_report(self) -> Dict[str, Any]:
        """
        Get comprehensive health report.
        
        Returns:
            Health report dictionary
        """
        return {
            "ready": self.is_ready(),
            "alive": self.is_alive(),
            "system": self.system_monitor.get_health_status(),
            "timestamp": datetime.utcnow().isoformat()
        }


# Global monitor instances
_system_monitor: Optional[SystemMonitor] = None
_performance_monitor: Optional[PerformanceMonitor] = None
_health_monitor: Optional[HealthMonitor] = None


def get_system_monitor() -> SystemMonitor:
    """Get global system monitor instance."""
    global _system_monitor
    if _system_monitor is None:
        _system_monitor = SystemMonitor()
    return _system_monitor


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


def get_health_monitor() -> HealthMonitor:
    """Get global health monitor instance."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
    return _health_monitor