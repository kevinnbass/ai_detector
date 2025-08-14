"""
WebSocket API Module for AI Detector System
Real-time communication via WebSocket connections
"""

from .connection_manager import (
    WebSocketConnection, ConnectionManager, WebSocketHandler,
    connection_manager, websocket_handler,
    start_connection_cleanup, stop_connection_cleanup
)

from .routes import (
    websocket_endpoint, websocket_admin_endpoint, websocket_health_check,
    WebSocketNotificationService, notification_service
)

__all__ = [
    # Connection management
    'WebSocketConnection', 'ConnectionManager', 'WebSocketHandler',
    'connection_manager', 'websocket_handler',
    'start_connection_cleanup', 'stop_connection_cleanup',
    
    # Routes and services
    'websocket_endpoint', 'websocket_admin_endpoint', 'websocket_health_check',
    'WebSocketNotificationService', 'notification_service'
]