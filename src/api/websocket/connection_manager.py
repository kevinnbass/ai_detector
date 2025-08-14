"""
WebSocket Connection Manager for Real-time AI Detection
Handles WebSocket connections for real-time text analysis
"""

from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, List, Optional, Any
import json
import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

from src.core.dependency_injection import create_configured_container
from src.core.abstractions.presentation_layer import IDetectionController
from src.api.rest.schemas import WebSocketMessage, DetectionRequestModel

logger = logging.getLogger(__name__)


@dataclass
class WebSocketConnection:
    """WebSocket connection information"""
    websocket: WebSocket
    client_id: str
    user_id: Optional[str] = None
    connected_at: datetime = None
    last_activity: datetime = None
    
    def __post_init__(self):
        if self.connected_at is None:
            self.connected_at = datetime.utcnow()
        if self.last_activity is None:
            self.last_activity = datetime.utcnow()
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.utcnow()


class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocketConnection] = {}
        self.user_connections: Dict[str, List[str]] = {}  # user_id -> [client_ids]
        self.container = create_configured_container()
        
    async def connect(self, websocket: WebSocket, client_id: str = None, user_id: str = None) -> str:
        """Accept WebSocket connection"""
        await websocket.accept()
        
        if not client_id:
            client_id = str(uuid.uuid4())
        
        connection = WebSocketConnection(
            websocket=websocket,
            client_id=client_id,
            user_id=user_id
        )
        
        self.active_connections[client_id] = connection
        
        # Track user connections
        if user_id:
            if user_id not in self.user_connections:
                self.user_connections[user_id] = []
            self.user_connections[user_id].append(client_id)
        
        logger.info(f"WebSocket connected: client_id={client_id}, user_id={user_id}")
        
        # Send connection confirmation
        await self.send_message(client_id, {
            "type": "connection_established",
            "data": {
                "client_id": client_id,
                "server_time": datetime.utcnow().isoformat()
            }
        })
        
        return client_id
    
    def disconnect(self, client_id: str):
        """Disconnect WebSocket"""
        if client_id in self.active_connections:
            connection = self.active_connections[client_id]
            user_id = connection.user_id
            
            del self.active_connections[client_id]
            
            # Remove from user connections
            if user_id and user_id in self.user_connections:
                self.user_connections[user_id] = [
                    cid for cid in self.user_connections[user_id] if cid != client_id
                ]
                if not self.user_connections[user_id]:
                    del self.user_connections[user_id]
            
            logger.info(f"WebSocket disconnected: client_id={client_id}, user_id={user_id}")
    
    async def send_message(self, client_id: str, message: Dict[str, Any]) -> bool:
        """Send message to specific client"""
        if client_id not in self.active_connections:
            return False
        
        try:
            connection = self.active_connections[client_id]
            message_with_timestamp = {
                **message,
                "timestamp": datetime.utcnow().isoformat(),
                "client_id": client_id
            }
            
            await connection.websocket.send_text(json.dumps(message_with_timestamp))
            connection.update_activity()
            return True
            
        except Exception as e:
            logger.error(f"Error sending message to {client_id}: {e}")
            self.disconnect(client_id)
            return False
    
    async def broadcast_to_user(self, user_id: str, message: Dict[str, Any]) -> int:
        """Broadcast message to all connections of a user"""
        if user_id not in self.user_connections:
            return 0
        
        sent_count = 0
        client_ids = self.user_connections[user_id].copy()  # Copy to avoid modification during iteration
        
        for client_id in client_ids:
            if await self.send_message(client_id, message):
                sent_count += 1
        
        return sent_count
    
    async def broadcast_to_all(self, message: Dict[str, Any]) -> int:
        """Broadcast message to all connected clients"""
        sent_count = 0
        client_ids = list(self.active_connections.keys())  # Copy to avoid modification
        
        for client_id in client_ids:
            if await self.send_message(client_id, message):
                sent_count += 1
        
        return sent_count
    
    def get_connection(self, client_id: str) -> Optional[WebSocketConnection]:
        """Get connection by client ID"""
        return self.active_connections.get(client_id)
    
    def get_user_connections(self, user_id: str) -> List[WebSocketConnection]:
        """Get all connections for a user"""
        if user_id not in self.user_connections:
            return []
        
        connections = []
        for client_id in self.user_connections[user_id]:
            if client_id in self.active_connections:
                connections.append(self.active_connections[client_id])
        
        return connections
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        total_connections = len(self.active_connections)
        authenticated_connections = sum(
            1 for conn in self.active_connections.values() if conn.user_id
        )
        
        return {
            "total_connections": total_connections,
            "authenticated_connections": authenticated_connections,
            "anonymous_connections": total_connections - authenticated_connections,
            "unique_users": len(self.user_connections),
            "average_connections_per_user": (
                authenticated_connections / len(self.user_connections)
                if self.user_connections else 0
            )
        }
    
    async def cleanup_inactive_connections(self, timeout_minutes: int = 30):
        """Remove inactive connections"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=timeout_minutes)
        inactive_clients = []
        
        for client_id, connection in self.active_connections.items():
            if connection.last_activity < cutoff_time:
                inactive_clients.append(client_id)
        
        for client_id in inactive_clients:
            logger.info(f"Removing inactive WebSocket connection: {client_id}")
            try:
                connection = self.active_connections[client_id]
                await connection.websocket.close(code=1000, reason="Connection timeout")
            except:
                pass  # Connection might already be closed
            finally:
                self.disconnect(client_id)
        
        return len(inactive_clients)


# Global connection manager
connection_manager = ConnectionManager()


class WebSocketHandler:
    """Handles WebSocket message processing"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self.container = create_configured_container()
        
    async def handle_message(self, client_id: str, message: Dict[str, Any]) -> None:
        """Handle incoming WebSocket message"""
        try:
            message_type = message.get("type")
            
            if message_type == "ping":
                await self._handle_ping(client_id, message)
            elif message_type == "detection_request":
                await self._handle_detection_request(client_id, message)
            elif message_type == "subscribe":
                await self._handle_subscribe(client_id, message)
            elif message_type == "unsubscribe":
                await self._handle_unsubscribe(client_id, message)
            else:
                await self._send_error(client_id, f"Unknown message type: {message_type}")
                
        except Exception as e:
            logger.error(f"Error handling WebSocket message from {client_id}: {e}")
            await self._send_error(client_id, f"Message processing error: {str(e)}")
    
    async def _handle_ping(self, client_id: str, message: Dict[str, Any]):
        """Handle ping message"""
        await self.connection_manager.send_message(client_id, {
            "type": "pong",
            "data": {
                "original_timestamp": message.get("timestamp"),
                "server_timestamp": datetime.utcnow().isoformat()
            }
        })
    
    async def _handle_detection_request(self, client_id: str, message: Dict[str, Any]):
        """Handle text detection request"""
        try:
            # Extract detection request data
            request_data = message.get("data", {})
            
            # Validate request
            detection_request = DetectionRequestModel(**request_data)
            
            # Get detection controller from DI container
            with self.container.create_scope() as scope:
                controller = scope.get_service(IDetectionController)
                
                # Add WebSocket context
                request_dict = detection_request.dict()
                request_dict["source"] = "websocket"
                
                # Get connection info for user context
                connection = self.connection_manager.get_connection(client_id)
                if connection and connection.user_id:
                    request_dict["user_id"] = connection.user_id
                
                # Perform detection
                response = await controller.detect_text(request_dict)
                
                # Send result back
                await self.connection_manager.send_message(client_id, {
                    "type": "detection_result",
                    "data": response.data if response.success else None,
                    "success": response.success,
                    "message": response.message,
                    "errors": response.errors,
                    "correlation_id": message.get("correlation_id")
                })
                
        except Exception as e:
            await self._send_error(client_id, f"Detection failed: {str(e)}")
    
    async def _handle_subscribe(self, client_id: str, message: Dict[str, Any]):
        """Handle subscription to events"""
        # This would implement subscription to specific events
        # For now, just acknowledge
        await self.connection_manager.send_message(client_id, {
            "type": "subscription_acknowledged",
            "data": {
                "subscriptions": message.get("data", {}).get("events", [])
            }
        })
    
    async def _handle_unsubscribe(self, client_id: str, message: Dict[str, Any]):
        """Handle unsubscription from events"""
        await self.connection_manager.send_message(client_id, {
            "type": "unsubscription_acknowledged",
            "data": {}
        })
    
    async def _send_error(self, client_id: str, error_message: str):
        """Send error message to client"""
        await self.connection_manager.send_message(client_id, {
            "type": "error",
            "data": {
                "message": error_message,
                "timestamp": datetime.utcnow().isoformat()
            }
        })


# Global WebSocket handler
websocket_handler = WebSocketHandler(connection_manager)


# Background task for connection cleanup
async def connection_cleanup_task():
    """Background task to clean up inactive connections"""
    while True:
        try:
            removed_count = await connection_manager.cleanup_inactive_connections()
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} inactive WebSocket connections")
        except Exception as e:
            logger.error(f"Error in connection cleanup task: {e}")
        
        # Sleep for 10 minutes
        await asyncio.sleep(600)


# Start cleanup task
cleanup_task = None


def start_connection_cleanup():
    """Start the connection cleanup background task"""
    global cleanup_task
    if cleanup_task is None:
        cleanup_task = asyncio.create_task(connection_cleanup_task())


def stop_connection_cleanup():
    """Stop the connection cleanup background task"""
    global cleanup_task
    if cleanup_task:
        cleanup_task.cancel()
        cleanup_task = None


__all__ = [
    'WebSocketConnection', 'ConnectionManager', 'WebSocketHandler',
    'connection_manager', 'websocket_handler',
    'start_connection_cleanup', 'stop_connection_cleanup'
]