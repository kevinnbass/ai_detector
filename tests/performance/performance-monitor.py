"""
Performance Monitoring and Alerting System
Real-time performance monitoring with alerts and reporting
"""

import asyncio
import time
import json
import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import psutil
import threading
from collections import deque, defaultdict
import aiohttp
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart


@dataclass
class PerformanceAlert:
    """Performance alert configuration"""
    metric_name: str
    threshold: float
    operator: str  # 'gt', 'lt', 'eq'
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    enabled: bool = True


@dataclass
class MetricReading:
    """Individual metric reading"""
    timestamp: datetime
    value: float
    metadata: Dict[str, Any] = None


@dataclass
class PerformanceReport:
    """Performance report data"""
    start_time: datetime
    end_time: datetime
    metrics_summary: Dict[str, Any]
    alerts_triggered: List[Dict[str, Any]]
    recommendations: List[str]
    system_health: str


class MetricsCollector:
    """Collects system and application metrics"""
    
    def __init__(self, max_readings: int = 1000):
        self.max_readings = max_readings
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_readings))
        self.collection_interval = 1.0  # seconds
        self.running = False
        self.collection_thread = None
        
    def start_collection(self):
        """Start background metrics collection"""
        if self.running:
            return
        
        self.running = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        logging.info("Metrics collection started")
    
    def stop_collection(self):
        """Stop background metrics collection"""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        logging.info("Metrics collection stopped")
    
    def _collection_loop(self):
        """Background collection loop"""
        while self.running:
            try:
                self._collect_system_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                logging.error(f"Error in metrics collection: {e}")
    
    def _collect_system_metrics(self):
        """Collect system-level metrics"""
        timestamp = datetime.now()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        self.record_metric("cpu_usage_percent", cpu_percent, timestamp)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        self.record_metric("memory_usage_percent", memory.percent, timestamp)
        self.record_metric("memory_available_mb", memory.available / 1024 / 1024, timestamp)
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        self.record_metric("disk_usage_percent", (disk.used / disk.total) * 100, timestamp)
        
        # Network metrics (if available)
        try:
            network = psutil.net_io_counters()
            self.record_metric("network_bytes_sent", network.bytes_sent, timestamp)
            self.record_metric("network_bytes_recv", network.bytes_recv, timestamp)
        except:
            pass  # Network metrics might not be available in all environments
    
    def record_metric(self, name: str, value: float, timestamp: datetime = None, metadata: Dict = None):
        """Record a metric reading"""
        if timestamp is None:
            timestamp = datetime.now()
        
        reading = MetricReading(timestamp=timestamp, value=value, metadata=metadata or {})
        self.metrics[name].append(reading)
    
    def get_metric_values(self, name: str, since: datetime = None) -> List[float]:
        """Get metric values, optionally filtered by time"""
        readings = self.metrics.get(name, [])
        
        if since:
            readings = [r for r in readings if r.timestamp >= since]
        
        return [r.value for r in readings]
    
    def get_metric_stats(self, name: str, since: datetime = None) -> Dict[str, float]:
        """Get statistical summary of a metric"""
        values = self.get_metric_values(name, since)
        
        if not values:
            return {
                "count": 0,
                "min": 0,
                "max": 0,
                "mean": 0,
                "median": 0,
                "std": 0
            }
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0
        }


class AlertEngine:
    """Performance alert engine"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alerts: List[PerformanceAlert] = []
        self.alert_history: List[Dict[str, Any]] = []
        self.alert_callbacks: List[Callable] = []
        self.checking_interval = 5.0  # seconds
        self.running = False
        self.alert_thread = None
    
    def add_alert(self, alert: PerformanceAlert):
        """Add a performance alert"""
        self.alerts.append(alert)
        logging.info(f"Added alert: {alert.metric_name} {alert.operator} {alert.threshold}")
    
    def add_callback(self, callback: Callable):
        """Add callback for alert notifications"""
        self.alert_callbacks.append(callback)
    
    def start_monitoring(self):
        """Start alert monitoring"""
        if self.running:
            return
        
        self.running = True
        self.alert_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.alert_thread.start()
        logging.info("Alert monitoring started")
    
    def stop_monitoring(self):
        """Stop alert monitoring"""
        self.running = False
        if self.alert_thread:
            self.alert_thread.join(timeout=5)
        logging.info("Alert monitoring stopped")
    
    def _monitoring_loop(self):
        """Background alert checking loop"""
        while self.running:
            try:
                self._check_alerts()
                time.sleep(self.checking_interval)
            except Exception as e:
                logging.error(f"Error in alert checking: {e}")
    
    def _check_alerts(self):
        """Check all alerts against current metrics"""
        current_time = datetime.now()
        check_window = current_time - timedelta(minutes=1)  # Check last minute
        
        for alert in self.alerts:
            if not alert.enabled:
                continue
            
            try:
                values = self.metrics_collector.get_metric_values(alert.metric_name, check_window)
                
                if not values:
                    continue
                
                # Use latest value for checking
                latest_value = values[-1]
                
                if self._evaluate_condition(latest_value, alert.threshold, alert.operator):
                    self._trigger_alert(alert, latest_value, current_time)
                    
            except Exception as e:
                logging.error(f"Error checking alert {alert.metric_name}: {e}")
    
    def _evaluate_condition(self, value: float, threshold: float, operator: str) -> bool:
        """Evaluate alert condition"""
        if operator == 'gt':
            return value > threshold
        elif operator == 'lt':
            return value < threshold
        elif operator == 'eq':
            return abs(value - threshold) < 0.001
        elif operator == 'gte':
            return value >= threshold
        elif operator == 'lte':
            return value <= threshold
        else:
            logging.warning(f"Unknown operator: {operator}")
            return False
    
    def _trigger_alert(self, alert: PerformanceAlert, value: float, timestamp: datetime):
        """Trigger an alert"""
        # Check if similar alert was recently triggered (debouncing)
        recent_alerts = [
            a for a in self.alert_history 
            if a['metric_name'] == alert.metric_name and 
               (timestamp - datetime.fromisoformat(a['timestamp'])).total_seconds() < 300  # 5 minutes
        ]
        
        if recent_alerts:
            return  # Skip duplicate alerts within 5 minutes
        
        alert_data = {
            "metric_name": alert.metric_name,
            "description": alert.description,
            "severity": alert.severity,
            "threshold": alert.threshold,
            "actual_value": value,
            "operator": alert.operator,
            "timestamp": timestamp.isoformat()
        }
        
        self.alert_history.append(alert_data)
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                logging.error(f"Error in alert callback: {e}")
        
        logging.warning(f"ALERT: {alert.description} - {alert.metric_name} = {value} (threshold: {alert.threshold})")


class PerformanceMonitor:
    """Main performance monitoring system"""
    
    def __init__(self, config_file: str = None):
        self.metrics_collector = MetricsCollector()
        self.alert_engine = AlertEngine(self.metrics_collector)
        self.config = self._load_config(config_file)
        self.reports: List[PerformanceReport] = []
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config.get('log_level', 'INFO')),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Setup default alerts
        self._setup_default_alerts()
        
        # Setup alert callbacks
        self._setup_alert_callbacks()
    
    def _load_config(self, config_file: str = None) -> Dict[str, Any]:
        """Load configuration from file"""
        default_config = {
            "log_level": "INFO",
            "metrics_retention_hours": 24,
            "alert_email": None,
            "webhook_url": None,
            "default_alerts": {
                "cpu_high": {"threshold": 80, "enabled": True},
                "memory_high": {"threshold": 85, "enabled": True},
                "disk_high": {"threshold": 90, "enabled": True},
                "response_time_high": {"threshold": 2000, "enabled": True}  # ms
            }
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logging.error(f"Error loading config file: {e}")
        
        return default_config
    
    def _setup_default_alerts(self):
        """Setup default performance alerts"""
        default_alerts = [
            PerformanceAlert(
                metric_name="cpu_usage_percent",
                threshold=self.config["default_alerts"]["cpu_high"]["threshold"],
                operator="gt",
                severity="medium",
                description="High CPU usage detected",
                enabled=self.config["default_alerts"]["cpu_high"]["enabled"]
            ),
            PerformanceAlert(
                metric_name="memory_usage_percent", 
                threshold=self.config["default_alerts"]["memory_high"]["threshold"],
                operator="gt",
                severity="medium",
                description="High memory usage detected",
                enabled=self.config["default_alerts"]["memory_high"]["enabled"]
            ),
            PerformanceAlert(
                metric_name="disk_usage_percent",
                threshold=self.config["default_alerts"]["disk_high"]["threshold"],
                operator="gt",
                severity="high",
                description="High disk usage detected",
                enabled=self.config["default_alerts"]["disk_high"]["enabled"]
            ),
            PerformanceAlert(
                metric_name="api_response_time_ms",
                threshold=self.config["default_alerts"]["response_time_high"]["threshold"],
                operator="gt",
                severity="medium",
                description="API response time too high",
                enabled=self.config["default_alerts"]["response_time_high"]["enabled"]
            )
        ]
        
        for alert in default_alerts:
            self.alert_engine.add_alert(alert)
    
    def _setup_alert_callbacks(self):
        """Setup alert notification callbacks"""
        # Console logging callback
        def log_alert(alert_data):
            logging.warning(f"Performance Alert: {alert_data['description']}")
        
        self.alert_engine.add_callback(log_alert)
        
        # Email callback (if configured)
        if self.config.get('alert_email'):
            def email_alert(alert_data):
                self._send_email_alert(alert_data)
            self.alert_engine.add_callback(email_alert)
        
        # Webhook callback (if configured)
        if self.config.get('webhook_url'):
            def webhook_alert(alert_data):
                asyncio.create_task(self._send_webhook_alert(alert_data))
            self.alert_engine.add_callback(webhook_alert)
    
    def start(self):
        """Start performance monitoring"""
        self.metrics_collector.start_collection()
        self.alert_engine.start_monitoring()
        logging.info("Performance monitoring started")
    
    def stop(self):
        """Stop performance monitoring"""
        self.metrics_collector.stop_collection()
        self.alert_engine.stop_monitoring()
        logging.info("Performance monitoring stopped")
    
    def record_application_metric(self, name: str, value: float, metadata: Dict = None):
        """Record an application-specific metric"""
        self.metrics_collector.record_metric(name, value, metadata=metadata)
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current system status"""
        current_time = datetime.now()
        last_hour = current_time - timedelta(hours=1)
        
        status = {
            "timestamp": current_time.isoformat(),
            "system_metrics": {},
            "recent_alerts": [],
            "health_score": 100
        }
        
        # System metrics
        for metric_name in ["cpu_usage_percent", "memory_usage_percent", "disk_usage_percent"]:
            stats = self.metrics_collector.get_metric_stats(metric_name, last_hour)
            status["system_metrics"][metric_name] = stats
            
            # Reduce health score based on high resource usage
            if stats["mean"] > 80:
                status["health_score"] -= 10
            elif stats["mean"] > 90:
                status["health_score"] -= 20
        
        # Recent alerts
        recent_alerts = [
            alert for alert in self.alert_engine.alert_history
            if (current_time - datetime.fromisoformat(alert['timestamp'])).total_seconds() < 3600
        ]
        status["recent_alerts"] = recent_alerts[-10:]  # Last 10 alerts
        
        # Reduce health score for recent alerts
        critical_alerts = [a for a in recent_alerts if a['severity'] == 'critical']
        high_alerts = [a for a in recent_alerts if a['severity'] == 'high']
        
        status["health_score"] -= len(critical_alerts) * 15
        status["health_score"] -= len(high_alerts) * 10
        status["health_score"] = max(0, status["health_score"])
        
        return status
    
    def generate_report(self, start_time: datetime = None, end_time: datetime = None) -> PerformanceReport:
        """Generate a performance report"""
        if end_time is None:
            end_time = datetime.now()
        if start_time is None:
            start_time = end_time - timedelta(hours=1)
        
        # Collect metrics summary
        metrics_summary = {}
        for metric_name in self.metrics_collector.metrics.keys():
            stats = self.metrics_collector.get_metric_stats(metric_name, start_time)
            metrics_summary[metric_name] = stats
        
        # Collect alerts in time range
        alerts_in_range = [
            alert for alert in self.alert_engine.alert_history
            if start_time <= datetime.fromisoformat(alert['timestamp']) <= end_time
        ]
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics_summary, alerts_in_range)
        
        # Determine system health
        health_status = self._determine_health_status(metrics_summary, alerts_in_range)
        
        report = PerformanceReport(
            start_time=start_time,
            end_time=end_time,
            metrics_summary=metrics_summary,
            alerts_triggered=alerts_in_range,
            recommendations=recommendations,
            system_health=health_status
        )
        
        self.reports.append(report)
        return report
    
    def _generate_recommendations(self, metrics_summary: Dict, alerts: List) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        # CPU recommendations
        cpu_stats = metrics_summary.get("cpu_usage_percent", {})
        if cpu_stats.get("mean", 0) > 70:
            recommendations.append("Consider scaling up CPU resources or optimizing CPU-intensive operations")
        
        # Memory recommendations
        memory_stats = metrics_summary.get("memory_usage_percent", {})
        if memory_stats.get("mean", 0) > 80:
            recommendations.append("Monitor memory usage patterns and consider increasing available memory")
        
        # Response time recommendations
        response_stats = metrics_summary.get("api_response_time_ms", {})
        if response_stats.get("mean", 0) > 1000:
            recommendations.append("Optimize API response times through caching, database indexing, or code optimization")
        
        # Alert-based recommendations
        critical_alerts = [a for a in alerts if a['severity'] == 'critical']
        if critical_alerts:
            recommendations.append("Address critical performance alerts immediately to prevent system degradation")
        
        return recommendations
    
    def _determine_health_status(self, metrics_summary: Dict, alerts: List) -> str:
        """Determine overall system health status"""
        critical_alerts = [a for a in alerts if a['severity'] == 'critical']
        high_alerts = [a for a in alerts if a['severity'] == 'high']
        
        # Check for critical issues
        if critical_alerts:
            return "Critical"
        
        # Check resource usage
        cpu_mean = metrics_summary.get("cpu_usage_percent", {}).get("mean", 0)
        memory_mean = metrics_summary.get("memory_usage_percent", {}).get("mean", 0)
        
        if cpu_mean > 90 or memory_mean > 90 or high_alerts:
            return "Warning"
        elif cpu_mean > 70 or memory_mean > 70:
            return "Degraded"
        else:
            return "Healthy"
    
    def _send_email_alert(self, alert_data: Dict):
        """Send email alert notification"""
        try:
            # This is a basic implementation - configure SMTP settings as needed
            msg = MimeMultipart()
            msg['From'] = "performance-monitor@aidetector.com"
            msg['To'] = self.config['alert_email']
            msg['Subject'] = f"Performance Alert: {alert_data['description']}"
            
            body = f"""
            Performance Alert Triggered
            
            Metric: {alert_data['metric_name']}
            Description: {alert_data['description']}
            Severity: {alert_data['severity']}
            Threshold: {alert_data['threshold']}
            Actual Value: {alert_data['actual_value']}
            Time: {alert_data['timestamp']}
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            # Note: Configure SMTP server settings
            # server = smtplib.SMTP('localhost', 587)
            # server.send_message(msg)
            # server.quit()
            
            logging.info(f"Email alert sent for {alert_data['metric_name']}")
            
        except Exception as e:
            logging.error(f"Failed to send email alert: {e}")
    
    async def _send_webhook_alert(self, alert_data: Dict):
        """Send webhook alert notification"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config['webhook_url'],
                    json=alert_data,
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    if response.status == 200:
                        logging.info(f"Webhook alert sent for {alert_data['metric_name']}")
                    else:
                        logging.error(f"Webhook alert failed with status {response.status}")
                        
        except Exception as e:
            logging.error(f"Failed to send webhook alert: {e}")
    
    def export_report(self, report: PerformanceReport, filename: str = None) -> str:
        """Export performance report to file"""
        if filename is None:
            filename = f"performance_report_{report.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        
        output_path = Path("test-results") / filename
        output_path.parent.mkdir(exist_ok=True)
        
        # Convert report to dict for JSON serialization
        report_dict = asdict(report)
        report_dict['start_time'] = report.start_time.isoformat()
        report_dict['end_time'] = report.end_time.isoformat()
        
        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        logging.info(f"Performance report exported to {output_path}")
        return str(output_path)


# Usage example and testing
if __name__ == "__main__":
    # Example usage
    monitor = PerformanceMonitor()
    
    try:
        monitor.start()
        
        # Simulate application metrics
        for i in range(10):
            # Simulate API response time
            response_time = 500 + (i * 100)  # Increasing response time
            monitor.record_application_metric("api_response_time_ms", response_time)
            
            # Simulate detection processing time
            detection_time = 50 + (i * 5)
            monitor.record_application_metric("detection_time_ms", detection_time)
            
            time.sleep(2)
        
        # Get current status
        status = monitor.get_current_status()
        print("Current Status:")
        print(json.dumps(status, indent=2))
        
        # Generate report
        report = monitor.generate_report()
        report_file = monitor.export_report(report)
        print(f"\nReport generated: {report_file}")
        
    finally:
        monitor.stop()