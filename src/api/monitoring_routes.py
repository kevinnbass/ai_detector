"""
API routes for monitoring and observability.

Provides endpoints for health checks, metrics, and system monitoring.
"""

from fastapi import APIRouter, HTTPException, Response, status
from typing import Dict, Any, Optional
import json

from src.core.monitoring import (
    get_system_monitor,
    get_health_monitor,
    get_metrics_collector,
    get_alert_manager,
    get_telemetry_client
)


router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Get system health status.
    
    Returns comprehensive health check results including
    system resources, service status, and readiness.
    """
    health_monitor = get_health_monitor()
    return health_monitor.get_health_report()


@router.get("/health/live")
async def liveness_probe() -> Dict[str, bool]:
    """
    Kubernetes liveness probe endpoint.
    
    Returns simple alive status for container orchestration.
    """
    health_monitor = get_health_monitor()
    is_alive = health_monitor.is_alive()
    
    if not is_alive:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is not alive"
        )
    
    return {"alive": True}


@router.get("/health/ready")
async def readiness_probe() -> Dict[str, bool]:
    """
    Kubernetes readiness probe endpoint.
    
    Returns readiness status for load balancer routing.
    """
    health_monitor = get_health_monitor()
    is_ready = health_monitor.is_ready()
    
    if not is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is not ready"
        )
    
    return {"ready": True}


@router.get("/metrics")
async def get_metrics(format: str = "json") -> Response:
    """
    Get current metrics.
    
    Args:
        format: Output format (json or prometheus)
    
    Returns:
        Metrics in requested format
    """
    metrics_collector = get_metrics_collector()
    
    if format == "prometheus":
        content = metrics_collector.export_metrics(format="prometheus")
        return Response(
            content=content,
            media_type="text/plain; version=0.0.4"
        )
    else:
        metrics_data = metrics_collector.get_all_metrics()
        return Response(
            content=json.dumps(metrics_data, indent=2, default=str),
            media_type="application/json"
        )


@router.get("/metrics/{metric_name}")
async def get_metric(metric_name: str) -> Dict[str, Any]:
    """
    Get specific metric by name.
    
    Args:
        metric_name: Name of the metric
    
    Returns:
        Metric details and current value
    """
    metrics_collector = get_metrics_collector()
    metric = metrics_collector.get_metric(metric_name)
    
    if not metric:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Metric '{metric_name}' not found"
        )
    
    return metric.to_dict()


@router.post("/metrics/reset")
async def reset_metrics() -> Dict[str, str]:
    """
    Reset all metrics to initial values.
    
    Returns:
        Confirmation message
    """
    metrics_collector = get_metrics_collector()
    metrics_collector.reset_all_metrics()
    
    return {"message": "All metrics have been reset"}


@router.get("/alerts")
async def get_alerts(
    active_only: bool = True,
    level: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get system alerts.
    
    Args:
        active_only: Only return active alerts
        level: Filter by alert level
    
    Returns:
        Alert information
    """
    alert_manager = get_alert_manager()
    
    if active_only:
        alerts = alert_manager.get_active_alerts()
    else:
        alerts = alert_manager.get_alert_history(limit=100)
    
    # Filter by level if specified
    if level:
        alerts = [a for a in alerts if a.level.value == level]
    
    return {
        "alerts": [alert.to_dict() for alert in alerts],
        "stats": alert_manager.get_alert_stats()
    }


@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str) -> Dict[str, str]:
    """
    Resolve an active alert.
    
    Args:
        alert_id: Alert identifier
    
    Returns:
        Confirmation message
    """
    alert_manager = get_alert_manager()
    alert_manager.resolve_alert(alert_id)
    
    return {"message": f"Alert {alert_id} resolved"}


@router.post("/alerts/{alert_name}/silence")
async def silence_alert(
    alert_name: str,
    duration_minutes: int = 60
) -> Dict[str, str]:
    """
    Silence an alert for specified duration.
    
    Args:
        alert_name: Name of alert to silence
        duration_minutes: Duration to silence in minutes
    
    Returns:
        Confirmation message
    """
    alert_manager = get_alert_manager()
    alert_manager.silence_alert(alert_name, duration_minutes)
    
    return {
        "message": f"Alert {alert_name} silenced for {duration_minutes} minutes"
    }


@router.get("/traces")
async def get_traces(
    trace_id: Optional[str] = None,
    limit: int = 100
) -> Dict[str, Any]:
    """
    Get distributed traces.
    
    Args:
        trace_id: Specific trace ID to retrieve
        limit: Maximum number of traces
    
    Returns:
        Trace information
    """
    telemetry_client = get_telemetry_client()
    
    if trace_id:
        traces = telemetry_client.get_trace(trace_id)
        return {
            "trace_id": trace_id,
            "spans": [span.to_dict() for span in traces]
        }
    else:
        # Return all traces (limited)
        traces_json = telemetry_client.export_traces(format="json")
        traces_data = json.loads(traces_json)
        
        # Limit traces
        limited_traces = dict(list(traces_data.items())[:limit])
        
        return {
            "traces": limited_traces,
            "total_traces": len(traces_data)
        }


@router.get("/traces/active")
async def get_active_spans() -> Dict[str, Any]:
    """
    Get currently active spans.
    
    Returns:
        List of active spans
    """
    telemetry_client = get_telemetry_client()
    active_spans = telemetry_client.get_active_spans()
    
    return {
        "active_spans": [span.to_dict() for span in active_spans],
        "count": len(active_spans)
    }


@router.get("/system")
async def get_system_info() -> Dict[str, Any]:
    """
    Get system information and resource usage.
    
    Returns:
        System metrics and configuration
    """
    system_monitor = get_system_monitor()
    health_status = system_monitor.get_health_status()
    
    return {
        "health": health_status,
        "config": {
            "monitoring_enabled": system_monitor.config.enable_health_checks,
            "performance_monitoring": system_monitor.config.enable_performance_monitoring,
            "resource_monitoring": system_monitor.config.enable_resource_monitoring
        }
    }


@router.post("/monitoring/start")
async def start_monitoring() -> Dict[str, str]:
    """
    Start background monitoring services.
    
    Returns:
        Confirmation message
    """
    system_monitor = get_system_monitor()
    alert_manager = get_alert_manager()
    
    system_monitor.start_monitoring()
    alert_manager.start_monitoring()
    
    return {"message": "Monitoring services started"}


@router.post("/monitoring/stop")
async def stop_monitoring() -> Dict[str, str]:
    """
    Stop background monitoring services.
    
    Returns:
        Confirmation message
    """
    system_monitor = get_system_monitor()
    alert_manager = get_alert_manager()
    
    system_monitor.stop_monitoring()
    alert_manager.stop_monitoring()
    
    return {"message": "Monitoring services stopped"}


def setup_monitoring_routes(app):
    """
    Set up monitoring routes in the FastAPI app.
    
    Args:
        app: FastAPI application instance
    """
    app.include_router(router)