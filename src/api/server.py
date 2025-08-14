"""
API Server for AI Detector System
Complete FastAPI application with REST and WebSocket endpoints
"""

import os
import logging
import asyncio
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

# Import main REST API app
from .rest.routes import app as rest_app
from .websocket.routes import websocket_endpoint, websocket_admin_endpoint
from .websocket.connection_manager import start_connection_cleanup, stop_connection_cleanup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting AI Detector API Server...")
    
    # Start WebSocket connection cleanup
    start_connection_cleanup()
    
    # Validate environment
    required_env_vars = ['JWT_SECRET_KEY']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        logger.warning(f"Missing environment variables: {missing_vars}")
    
    logger.info("API Server started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Detector API Server...")
    
    # Stop WebSocket cleanup
    stop_connection_cleanup()
    
    logger.info("API Server shutdown complete")


# Create the complete application
def create_app() -> FastAPI:
    """Create and configure the complete FastAPI application"""
    
    # Use the existing REST app as base
    app = rest_app
    app.lifespan = lifespan
    
    # Add WebSocket routes
    app.add_websocket_route("/ws", websocket_endpoint)
    app.add_websocket_route("/ws/admin", websocket_admin_endpoint)
    
    # Add additional middleware if needed
    # (CORS and other middleware already configured in rest/routes.py)
    
    return app


# Create the application instance
app = create_app()


# Additional endpoints for server management
@app.get("/api/v1/server/info", tags=["Server"])
async def get_server_info():
    """Get server information"""
    from .websocket import connection_manager
    
    ws_stats = connection_manager.get_connection_stats()
    
    return {
        "server": {
            "name": "AI Detector API",
            "version": "1.0.0",
            "environment": os.getenv("ENVIRONMENT", "development"),
            "debug": os.getenv("DEBUG", "false").lower() == "true"
        },
        "websocket": {
            "enabled": True,
            "connections": ws_stats
        },
        "features": {
            "rest_api": True,
            "websocket_api": True,
            "authentication": True,
            "rate_limiting": True,
            "caching": True
        }
    }


@app.get("/api/v1/server/metrics", tags=["Server"])
async def get_server_metrics():
    """Get server metrics"""
    from .rest.middleware import metrics_middleware
    from .websocket import connection_manager
    
    api_metrics = metrics_middleware.get_metrics()
    ws_stats = connection_manager.get_connection_stats()
    
    return {
        "api": api_metrics,
        "websocket": ws_stats,
        "timestamp": api_metrics.get("timestamp")
    }


# Development server configuration
def run_dev_server():
    """Run development server with hot reload"""
    uvicorn.run(
        "src.api.server:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )


def run_production_server():
    """Run production server"""
    # Production configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    workers = int(os.getenv("WORKERS", "4"))
    
    uvicorn.run(
        "src.api.server:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
        access_log=True,
        loop="uvloop"  # Use uvloop for better performance
    )


if __name__ == "__main__":
    # Check environment and run appropriate server
    environment = os.getenv("ENVIRONMENT", "development")
    
    if environment == "production":
        run_production_server()
    else:
        run_dev_server()