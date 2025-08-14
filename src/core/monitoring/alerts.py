"""
Alerting system for the AI Detector.

Provides configurable alerts based on metrics, health checks,
and system events.
"""

import time
import threading
from typing import Dict, Any, Optional, List, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import logging
from collections import defaultdict

from .logger import get_logger
from .metrics import get_metrics_collector, MetricsCollector
from .monitor import HealthStatus


logger = get_logger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertState(Enum):
    """Alert state in lifecycle."""
    
    PENDING = "pending"
    FIRING = "firing"
    RESOLVED = "resolved"
    SILENCED = "silenced"


@dataclass
class Alert:
    """
    Represents a system alert.
    
    Contains alert details, state, and metadata.
    """
    
    id: str
    name: str
    level: AlertLevel
    message: str
    state: AlertState = AlertState.PENDING
    details: Dict[str, Any] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    fired_at: Optional[datetime] = None
    silence_until: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary format."""
        return {
            "id": self.id,
            "name": self.name,
            "level": self.level.value,
            "message": self.message,
            "state": self.state.value,
            "details": self.details,
            "labels": self.labels,
            "annotations": self.annotations,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "fired_at": self.fired_at.isoformat() if self.fired_at else None,
            "silence_until": self.silence_until.isoformat() if self.silence_until else None
        }
    
    def is_silenced(self) -> bool:
        """Check if alert is currently silenced."""
        if self.silence_until:
            return datetime.utcnow() < self.silence_until
        return False
    
    def fire(self):
        """Mark alert as firing."""
        self.state = AlertState.FIRING
        self.fired_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def resolve(self):
        """Mark alert as resolved."""
        self.state = AlertState.RESOLVED
        self.resolved_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def silence(self, duration_minutes: int):
        """
        Silence alert for specified duration.
        
        Args:
            duration_minutes: Minutes to silence alert
        """
        self.state = AlertState.SILENCED
        self.silence_until = datetime.utcnow() + timedelta(minutes=duration_minutes)
        self.updated_at = datetime.utcnow()


@dataclass
class AlertRule:
    """
    Rule for generating alerts based on conditions.
    
    Defines when and how alerts should be triggered.
    """
    
    name: str
    condition: Callable[[], bool]
    level: AlertLevel
    message_template: str
    cooldown_minutes: int = 5
    auto_resolve: bool = True
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    
    def evaluate(self) -> Optional[Alert]:
        """
        Evaluate the alert rule.
        
        Returns:
            Alert if condition is met, None otherwise
        """
        try:
            if self.condition():
                return Alert(
                    id=f"{self.name}_{int(time.time())}",
                    name=self.name,
                    level=self.level,
                    message=self.message_template,
                    labels=self.labels.copy(),
                    annotations=self.annotations.copy()
                )
        except Exception as e:
            logger.error(f"Alert rule evaluation failed for {self.name}: {e}")
        
        return None


class AlertChannel:
    """
    Base class for alert notification channels.
    
    Defines how alerts are sent to external systems.
    """
    
    def send(self, alert: Alert) -> bool:
        """
        Send alert through this channel.
        
        Args:
            alert: Alert to send
            
        Returns:
            True if sent successfully
        """
        raise NotImplementedError


class LogAlertChannel(AlertChannel):
    """Alert channel that logs alerts."""
    
    def send(self, alert: Alert) -> bool:
        """Log the alert."""
        log_method = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.ERROR: logger.error,
            AlertLevel.CRITICAL: logger.critical
        }.get(alert.level, logger.info)
        
        log_method(
            f"ALERT: {alert.message}",
            alert_id=alert.id,
            alert_name=alert.name,
            alert_level=alert.level.value,
            alert_details=alert.details
        )
        
        return True


class WebhookAlertChannel(AlertChannel):
    """Alert channel that sends to webhooks."""
    
    def __init__(self, webhook_url: str):
        """
        Initialize webhook channel.
        
        Args:
            webhook_url: URL to send alerts to
        """
        self.webhook_url = webhook_url
    
    def send(self, alert: Alert) -> bool:
        """Send alert to webhook."""
        try:
            import requests
            
            response = requests.post(
                self.webhook_url,
                json=alert.to_dict(),
                timeout=5
            )
            
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to send alert to webhook: {e}")
            return False


class AlertManager:
    """
    Central alert management system.
    
    Manages alert rules, channels, and alert lifecycle.
    """
    
    def __init__(
        self,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        """
        Initialize alert manager.
        
        Args:
            metrics_collector: Metrics collector for rule evaluation
        """
        self.metrics = metrics_collector or get_metrics_collector()
        self._rules: Dict[str, AlertRule] = {}
        self._channels: List[AlertChannel] = []
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: List[Alert] = []
        self._rule_cooldowns: Dict[str, datetime] = {}
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Register default alert channel
        self._channels.append(LogAlertChannel())
        
        # Register default rules
        self._register_default_rules()
    
    def _register_default_rules(self):
        """Register default alert rules."""
        # High CPU usage
        self.register_rule(
            AlertRule(
                name="high_cpu_usage",
                condition=lambda: self._check_metric("cpu_usage_percent", ">", 80),
                level=AlertLevel.WARNING,
                message_template="CPU usage is above 80%",
                cooldown_minutes=10,
                labels={"category": "resource"}
            )
        )
        
        # High memory usage
        self.register_rule(
            AlertRule(
                name="high_memory_usage",
                condition=lambda: self._check_metric("memory_usage_percent", ">", 85),
                level=AlertLevel.WARNING,
                message_template="Memory usage is above 85%",
                cooldown_minutes=10,
                labels={"category": "resource"}
            )
        )
        
        # High error rate
        self.register_rule(
            AlertRule(
                name="high_error_rate",
                condition=lambda: self._check_metric("errors_per_minute", ">", 10),
                level=AlertLevel.ERROR,
                message_template="Error rate exceeds 10 errors per minute",
                cooldown_minutes=5,
                labels={"category": "application"}
            )
        )
        
        # Slow response time
        self.register_rule(
            AlertRule(
                name="slow_response_time",
                condition=lambda: self._check_histogram_percentile(
                    "request_duration_seconds",
                    "p95",
                    ">",
                    2.0
                ),
                level=AlertLevel.WARNING,
                message_template="95th percentile response time exceeds 2 seconds",
                cooldown_minutes=15,
                labels={"category": "performance"}
            )
        )
    
    def register_rule(self, rule: AlertRule):
        """
        Register an alert rule.
        
        Args:
            rule: Alert rule to register
        """
        self._rules[rule.name] = rule
        logger.info(f"Registered alert rule: {rule.name}")
    
    def register_channel(self, channel: AlertChannel):
        """
        Register an alert channel.
        
        Args:
            channel: Alert channel to register
        """
        self._channels.append(channel)
    
    def _check_metric(
        self,
        metric_name: str,
        operator: str,
        threshold: float
    ) -> bool:
        """
        Check if a metric meets a condition.
        
        Args:
            metric_name: Name of the metric
            operator: Comparison operator (>, <, ==, >=, <=)
            threshold: Threshold value
            
        Returns:
            True if condition is met
        """
        metric = self.metrics.get_metric(metric_name)
        if not metric:
            return False
        
        value = metric.get_value()
        
        if operator == ">":
            return value > threshold
        elif operator == "<":
            return value < threshold
        elif operator == ">=":
            return value >= threshold
        elif operator == "<=":
            return value <= threshold
        elif operator == "==":
            return value == threshold
        else:
            return False
    
    def _check_histogram_percentile(
        self,
        metric_name: str,
        percentile: str,
        operator: str,
        threshold: float
    ) -> bool:
        """
        Check if a histogram percentile meets a condition.
        
        Args:
            metric_name: Name of the histogram metric
            percentile: Percentile to check (p50, p95, etc.)
            operator: Comparison operator
            threshold: Threshold value
            
        Returns:
            True if condition is met
        """
        metric = self.metrics.get_metric(metric_name)
        if not metric:
            return False
        
        value = metric.get_value()
        if isinstance(value, dict) and 'percentiles' in value:
            percentile_value = value['percentiles'].get(percentile, 0)
            
            if operator == ">":
                return percentile_value > threshold
            elif operator == "<":
                return percentile_value < threshold
            elif operator == ">=":
                return percentile_value >= threshold
            elif operator == "<=":
                return percentile_value <= threshold
            elif operator == "==":
                return percentile_value == threshold
        
        return False
    
    def evaluate_rules(self):
        """Evaluate all alert rules and generate alerts."""
        now = datetime.utcnow()
        
        for rule_name, rule in self._rules.items():
            # Check cooldown
            if rule_name in self._rule_cooldowns:
                if now < self._rule_cooldowns[rule_name]:
                    continue
            
            # Check if alert already active
            if rule_name in self._active_alerts:
                alert = self._active_alerts[rule_name]
                
                # Check if condition still met
                if rule.auto_resolve and not rule.condition():
                    self.resolve_alert(alert.id)
                continue
            
            # Evaluate rule
            alert = rule.evaluate()
            if alert:
                self.trigger_alert(alert)
                
                # Set cooldown
                self._rule_cooldowns[rule_name] = now + timedelta(minutes=rule.cooldown_minutes)
    
    def trigger_alert(self, alert: Alert):
        """
        Trigger a new alert.
        
        Args:
            alert: Alert to trigger
        """
        # Check if already silenced
        if alert.is_silenced():
            logger.info(f"Alert {alert.name} is silenced until {alert.silence_until}")
            return
        
        # Mark as firing
        alert.fire()
        
        # Store active alert
        self._active_alerts[alert.name] = alert
        self._alert_history.append(alert)
        
        # Send through channels
        for channel in self._channels:
            try:
                channel.send(alert)
            except Exception as e:
                logger.error(f"Failed to send alert through channel: {e}")
        
        # Update metrics
        self.metrics.increment_counter(
            "alerts_triggered_total",
            labels={
                "alert_name": alert.name,
                "alert_level": alert.level.value
            }
        )
    
    def resolve_alert(self, alert_id: str):
        """
        Resolve an active alert.
        
        Args:
            alert_id: Alert ID to resolve
        """
        # Find alert by ID
        alert = None
        for name, active_alert in self._active_alerts.items():
            if active_alert.id == alert_id:
                alert = active_alert
                break
        
        if not alert:
            logger.warning(f"Alert {alert_id} not found")
            return
        
        # Mark as resolved
        alert.resolve()
        
        # Remove from active alerts
        self._active_alerts.pop(alert.name, None)
        
        # Log resolution
        logger.info(f"Alert resolved: {alert.name}")
        
        # Update metrics
        self.metrics.increment_counter(
            "alerts_resolved_total",
            labels={
                "alert_name": alert.name,
                "alert_level": alert.level.value
            }
        )
    
    def silence_alert(self, alert_name: str, duration_minutes: int):
        """
        Silence an alert for a specified duration.
        
        Args:
            alert_name: Name of alert to silence
            duration_minutes: Minutes to silence
        """
        if alert_name in self._active_alerts:
            alert = self._active_alerts[alert_name]
            alert.silence(duration_minutes)
            logger.info(f"Alert {alert_name} silenced for {duration_minutes} minutes")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get list of active alerts."""
        return list(self._active_alerts.values())
    
    def get_alert_history(
        self,
        limit: int = 100,
        level: Optional[AlertLevel] = None
    ) -> List[Alert]:
        """
        Get alert history.
        
        Args:
            limit: Maximum number of alerts to return
            level: Filter by alert level
            
        Returns:
            List of historical alerts
        """
        alerts = self._alert_history[-limit:]
        
        if level:
            alerts = [a for a in alerts if a.level == level]
        
        return alerts
    
    def start_monitoring(self, interval_seconds: int = 30):
        """
        Start alert monitoring in background.
        
        Args:
            interval_seconds: Evaluation interval
        """
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            logger.warning("Alert monitoring already running")
            return
        
        self._stop_event.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self._monitoring_thread.start()
        logger.info("Alert monitoring started")
    
    def stop_monitoring(self):
        """Stop alert monitoring."""
        self._stop_event.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        logger.info("Alert monitoring stopped")
    
    def _monitoring_loop(self, interval_seconds: int):
        """
        Background monitoring loop.
        
        Args:
            interval_seconds: Evaluation interval
        """
        while not self._stop_event.is_set():
            try:
                self.evaluate_rules()
            except Exception as e:
                logger.error(f"Alert evaluation error: {e}")
            
            self._stop_event.wait(interval_seconds)
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics."""
        active_by_level = defaultdict(int)
        for alert in self._active_alerts.values():
            active_by_level[alert.level.value] += 1
        
        history_by_level = defaultdict(int)
        for alert in self._alert_history:
            history_by_level[alert.level.value] += 1
        
        return {
            "active_alerts": len(self._active_alerts),
            "active_by_level": dict(active_by_level),
            "total_triggered": len(self._alert_history),
            "history_by_level": dict(history_by_level),
            "rules_registered": len(self._rules),
            "channels_registered": len(self._channels)
        }


# Global alert manager
_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Get global alert manager instance."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager