"""
WebSocket Routes for AI Detector API
Real-time text analysis via WebSocket connections
"""

from fastapi import WebSocket, WebSocketDisconnect, Depends, Query
from typing import Optional
import json
import logging

from .connection_manager import connection_manager, websocket_handler
from src.api.rest.auth import get_user_from_token, AuthenticationError

logger = logging.getLogger(__name__)


async def websocket_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
    client_id: Optional[str] = Query(None)
):
    """
    WebSocket endpoint for real-time AI detection
    
    Query parameters:
    - token: JWT authentication token (optional)
    - client_id: Custom client identifier (optional)
    
    Message format:
    {
        "type": "message_type",
        "data": {...},
        "correlation_id": "optional_id"
    }
    
    Supported message types:
    - ping: Health check
    - detection_request: Analyze text for AI content
    - subscribe: Subscribe to events
    - unsubscribe: Unsubscribe from events
    """
    
    user_id = None
    
    # Authenticate if token provided
    if token:
        try:
            user = get_user_from_token(token)
            user_id = user.user_id
        except AuthenticationError:
            logger.warning("Invalid WebSocket authentication token")
            # Don't reject connection, but user will have limited access
    
    # Connect to WebSocket
    try:
        client_id = await connection_manager.connect(websocket, client_id, user_id)
        
        # Send welcome message
        await connection_manager.send_message(client_id, {
            "type": "welcome",
            "data": {
                "authenticated": user_id is not None,
                "client_id": client_id,
                "server_info": {
                    "name": "AI Detector WebSocket API",
                    "version": "1.0.0"
                }
            }
        })
        
        # Message handling loop
        while True:
            try:
                # Receive message
                message_text = await websocket.receive_text()
                
                try:
                    message = json.loads(message_text)
                except json.JSONDecodeError:
                    await connection_manager.send_message(client_id, {
                        "type": "error",
                        "data": {
                            "message": "Invalid JSON format",
                            "error_code": "INVALID_JSON"
                        }
                    })
                    continue
                
                # Handle message
                await websocket_handler.handle_message(client_id, message)
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket client {client_id} disconnected")
                break
                
            except Exception as e:
                logger.error(f"Error handling WebSocket message for {client_id}: {e}")
                await connection_manager.send_message(client_id, {
                    "type": "error", 
                    "data": {
                        "message": f"Message handling error: {str(e)}",
                        "error_code": "MESSAGE_HANDLING_ERROR"
                    }
                })
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected during setup")
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    finally:
        if 'client_id' in locals():
            connection_manager.disconnect(client_id)


async def websocket_admin_endpoint(
    websocket: WebSocket,
    token: str = Query(...)
):
    """
    Admin WebSocket endpoint for monitoring and management
    
    Requires admin authentication token
    """
    
    try:
        # Authenticate admin
        user = get_user_from_token(token)
        if not user.is_admin():
            await websocket.close(code=1008, reason="Admin access required")
            return
        
        user_id = user.user_id
        
    except AuthenticationError:
        await websocket.close(code=1008, reason="Invalid authentication token")
        return
    
    # Connect admin WebSocket
    try:
        client_id = await connection_manager.connect(websocket, None, user_id)
        
        # Send admin welcome message
        stats = connection_manager.get_connection_stats()
        await connection_manager.send_message(client_id, {
            "type": "admin_welcome",
            "data": {
                "admin_user": user.username,
                "connection_stats": stats,
                "permissions": ["monitor", "broadcast", "disconnect"]
            }
        })
        
        # Admin message handling loop
        while True:
            try:
                message_text = await websocket.receive_text()
                message = json.loads(message_text)
                
                await handle_admin_message(client_id, message)
                
            except WebSocketDisconnect:
                logger.info(f"Admin WebSocket {client_id} disconnected")
                break
                
            except json.JSONDecodeError:
                await connection_manager.send_message(client_id, {
                    "type": "error",
                    "data": {"message": "Invalid JSON format"}
                })
                
            except Exception as e:
                logger.error(f"Error in admin WebSocket {client_id}: {e}")
                await connection_manager.send_message(client_id, {
                    "type": "error",
                    "data": {"message": f"Error: {str(e)}"}
                })
    
    except Exception as e:
        logger.error(f"Admin WebSocket error: {e}")
    finally:
        if 'client_id' in locals():
            connection_manager.disconnect(client_id)


async def handle_admin_message(client_id: str, message: dict):
    """Handle admin WebSocket messages"""
    
    message_type = message.get("type")
    data = message.get("data", {})
    
    if message_type == "get_stats":
        # Get connection statistics
        stats = connection_manager.get_connection_stats()
        await connection_manager.send_message(client_id, {
            "type": "stats_response",
            "data": stats
        })
    
    elif message_type == "broadcast":
        # Broadcast message to all connections
        broadcast_message = data.get("message", {})
        sent_count = await connection_manager.broadcast_to_all(broadcast_message)
        
        await connection_manager.send_message(client_id, {
            "type": "broadcast_result", 
            "data": {"sent_to": sent_count}
        })
    
    elif message_type == "disconnect_client":
        # Disconnect specific client
        target_client_id = data.get("client_id")
        if target_client_id and target_client_id in connection_manager.active_connections:
            target_connection = connection_manager.active_connections[target_client_id]
            await target_connection.websocket.close(code=1000, reason="Disconnected by admin")
            connection_manager.disconnect(target_client_id)
            
            await connection_manager.send_message(client_id, {
                "type": "disconnect_result",
                "data": {"disconnected": target_client_id}
            })
        else:
            await connection_manager.send_message(client_id, {
                "type": "error",
                "data": {"message": "Client not found"}
            })
    
    elif message_type == "get_connections":
        # Get list of all connections
        connections_info = []
        for cid, conn in connection_manager.active_connections.items():
            connections_info.append({
                "client_id": cid,
                "user_id": conn.user_id,
                "connected_at": conn.connected_at.isoformat(),
                "last_activity": conn.last_activity.isoformat()
            })
        
        await connection_manager.send_message(client_id, {
            "type": "connections_response",
            "data": {"connections": connections_info}
        })
    
    else:
        await connection_manager.send_message(client_id, {
            "type": "error",
            "data": {"message": f"Unknown admin command: {message_type}"}
        })


# Health check for WebSocket
async def websocket_health_check():
    """WebSocket health check function"""
    stats = connection_manager.get_connection_stats()
    return {
        "websocket_enabled": True,
        "active_connections": stats["total_connections"],
        "authenticated_connections": stats["authenticated_connections"]
    }


# Notification system for broadcasting events
class WebSocketNotificationService:
    """Service for sending notifications via WebSocket"""
    
    def __init__(self):
        self.connection_manager = connection_manager
    
    async def notify_user(self, user_id: str, notification: dict) -> int:
        """Send notification to specific user"""
        message = {
            "type": "notification",
            "data": notification
        }
        return await self.connection_manager.broadcast_to_user(user_id, message)
    
    async def notify_all_users(self, notification: dict) -> int:
        """Send notification to all connected users"""
        message = {
            "type": "notification",
            "data": notification
        }
        return await self.connection_manager.broadcast_to_all(message)
    
    async def notify_detection_complete(self, user_id: str, detection_result: dict):
        """Notify user about completed detection"""
        notification = {
            "type": "detection_complete",
            "result": detection_result,
            "timestamp": detection_result.get("timestamp")
        }
        return await self.notify_user(user_id, notification)
    
    async def notify_training_complete(self, user_id: str, training_result: dict):
        """Notify user about completed training"""
        notification = {
            "type": "training_complete", 
            "result": training_result,
            "timestamp": training_result.get("timestamp")
        }
        return await self.notify_user(user_id, notification)
    
    async def notify_system_maintenance(self, message: str, scheduled_time: str = None):
        """Notify all users about system maintenance"""
        notification = {
            "type": "system_maintenance",
            "message": message,
            "scheduled_time": scheduled_time
        }
        return await self.notify_all_users(notification)


# Global notification service
notification_service = WebSocketNotificationService()


__all__ = [
    'websocket_endpoint', 'websocket_admin_endpoint', 'websocket_health_check',
    'WebSocketNotificationService', 'notification_service'
]