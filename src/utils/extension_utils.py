"""
Chrome Extension Utilities - Shared Functions for Extension Development
"""

import json
from typing import Any, Dict, List, Optional, Union, Callable
import asyncio
import logging

logger = logging.getLogger(__name__)


# ============================================
# Message Protocol Utilities
# ============================================

class MessageType:
    """Standard message types for extension communication"""
    DETECT_TEXT = "DETECT_TEXT"
    ANALYSIS_RESULT = "ANALYSIS_RESULT"
    GET_SETTINGS = "GET_SETTINGS"
    SET_SETTINGS = "SET_SETTINGS"
    ERROR = "ERROR"
    PING = "PING"
    PONG = "PONG"
    COLLECT_DATA = "COLLECT_DATA"
    TRAIN_MODEL = "TRAIN_MODEL"
    UPDATE_UI = "UPDATE_UI"


class Message:
    """Standard message structure"""
    
    def __init__(self, type_: str, data: Any = None, request_id: Optional[str] = None):
        self.type = type_
        self.data = data
        self.request_id = request_id
        self.timestamp = int(asyncio.get_event_loop().time() * 1000) if hasattr(asyncio, 'get_event_loop') else 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary"""
        return {
            'type': self.type,
            'data': self.data,
            'requestId': self.request_id,
            'timestamp': self.timestamp
        }
    
    def to_json(self) -> str:
        """Convert message to JSON string"""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary"""
        return cls(
            type_=data.get('type', ''),
            data=data.get('data'),
            request_id=data.get('requestId')
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Message':
        """Create message from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)


# ============================================
# Response Utilities
# ============================================

class Response:
    """Standard response structure"""
    
    def __init__(self, success: bool, data: Any = None, error: Optional[str] = None, 
                 request_id: Optional[str] = None):
        self.success = success
        self.data = data
        self.error = error
        self.request_id = request_id
        self.timestamp = int(asyncio.get_event_loop().time() * 1000) if hasattr(asyncio, 'get_event_loop') else 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary"""
        return {
            'success': self.success,
            'data': self.data,
            'error': self.error,
            'requestId': self.request_id,
            'timestamp': self.timestamp
        }
    
    def to_json(self) -> str:
        """Convert response to JSON string"""
        return json.dumps(self.to_dict())
    
    @classmethod
    def success(cls, data: Any = None, request_id: Optional[str] = None) -> 'Response':
        """Create success response"""
        return cls(True, data=data, request_id=request_id)
    
    @classmethod
    def error(cls, error: str, request_id: Optional[str] = None) -> 'Response':
        """Create error response"""
        return cls(False, error=error, request_id=request_id)


# ============================================
# Settings Management
# ============================================

class ExtensionSettings:
    """Extension settings manager"""
    
    def __init__(self, defaults: Optional[Dict[str, Any]] = None):
        self.defaults = defaults or self.get_default_settings()
        self.settings = self.defaults.copy()
    
    @staticmethod
    def get_default_settings() -> Dict[str, Any]:
        """Get default extension settings"""
        return {
            'enabled': True,
            'auto_detect': True,
            'show_confidence': True,
            'highlight_ai_text': True,
            'api_key': '',
            'model': 'gemini-pro',
            'confidence_threshold': 0.7,
            'detection_mode': 'hybrid',  # pattern, llm, hybrid
            'ui_theme': 'light',
            'notifications': True,
            'data_collection': False,
            'debug_mode': False
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get setting value"""
        return self.settings.get(key, default if default is not None else self.defaults.get(key))
    
    def set(self, key: str, value: Any) -> None:
        """Set setting value"""
        self.settings[key] = value
    
    def update(self, settings: Dict[str, Any]) -> None:
        """Update multiple settings"""
        self.settings.update(settings)
    
    def reset(self, key: Optional[str] = None) -> None:
        """Reset setting(s) to default"""
        if key:
            if key in self.defaults:
                self.settings[key] = self.defaults[key]
        else:
            self.settings = self.defaults.copy()
    
    def validate(self) -> List[str]:
        """Validate settings and return list of errors"""
        errors = []
        
        # Validate confidence threshold
        threshold = self.get('confidence_threshold')
        if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
            errors.append("Confidence threshold must be between 0 and 1")
        
        # Validate detection mode
        mode = self.get('detection_mode')
        if mode not in ['pattern', 'llm', 'hybrid']:
            errors.append("Detection mode must be 'pattern', 'llm', or 'hybrid'")
        
        # Validate API key for LLM mode
        if mode in ['llm', 'hybrid'] and not self.get('api_key'):
            errors.append("API key required for LLM detection mode")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return self.settings.copy()
    
    def to_json(self) -> str:
        """Convert settings to JSON string"""
        return json.dumps(self.settings, indent=2)


# ============================================
# Storage Utilities
# ============================================

class StorageManager:
    """Chrome extension storage manager simulation"""
    
    def __init__(self):
        self.data = {}
    
    def get(self, keys: Union[str, List[str]], default: Any = None) -> Dict[str, Any]:
        """Get values from storage"""
        if isinstance(keys, str):
            keys = [keys]
        
        result = {}
        for key in keys:
            result[key] = self.data.get(key, default)
        
        return result
    
    def set(self, data: Dict[str, Any]) -> bool:
        """Set values in storage"""
        try:
            self.data.update(data)
            return True
        except Exception as e:
            logger.error(f"Error setting storage data: {e}")
            return False
    
    def remove(self, keys: Union[str, List[str]]) -> bool:
        """Remove keys from storage"""
        if isinstance(keys, str):
            keys = [keys]
        
        try:
            for key in keys:
                self.data.pop(key, None)
            return True
        except Exception as e:
            logger.error(f"Error removing storage keys: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all storage data"""
        try:
            self.data.clear()
            return True
        except Exception as e:
            logger.error(f"Error clearing storage: {e}")
            return False


# ============================================
# UI State Management
# ============================================

class UIState:
    """UI state management"""
    
    def __init__(self):
        self.state = {
            'is_detecting': False,
            'last_result': None,
            'error_message': None,
            'settings_open': False,
            'data_collection_open': False,
            'notifications': []
        }
        self.subscribers = []
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state"""
        return self.state.copy()
    
    def update_state(self, updates: Dict[str, Any]) -> None:
        """Update state and notify subscribers"""
        old_state = self.state.copy()
        self.state.update(updates)
        
        # Notify subscribers of changes
        for subscriber in self.subscribers:
            try:
                subscriber(self.state, old_state)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {e}")
    
    def subscribe(self, callback: Callable[[Dict[str, Any], Dict[str, Any]], None]) -> None:
        """Subscribe to state changes"""
        self.subscribers.append(callback)
    
    def add_notification(self, message: str, type_: str = 'info', 
                        duration: int = 5000) -> None:
        """Add notification to state"""
        notification = {
            'id': f"notification_{len(self.state['notifications'])}",
            'message': message,
            'type': type_,
            'duration': duration,
            'timestamp': int(asyncio.get_event_loop().time() * 1000) if hasattr(asyncio, 'get_event_loop') else 0
        }
        
        notifications = self.state['notifications'][:]
        notifications.append(notification)
        self.update_state({'notifications': notifications})
    
    def remove_notification(self, notification_id: str) -> None:
        """Remove notification from state"""
        notifications = [n for n in self.state['notifications'] 
                        if n.get('id') != notification_id]
        self.update_state({'notifications': notifications})


# ============================================
# Content Script Utilities
# ============================================

def extract_text_from_element(element_data: Dict[str, Any]) -> Optional[str]:
    """Extract and clean text from DOM element data"""
    text = element_data.get('textContent', '').strip()
    
    if not text:
        return None
    
    # Clean up common issues
    text = ' '.join(text.split())  # Normalize whitespace
    text = text.replace('\u200b', '')  # Remove zero-width spaces
    
    return text if len(text) > 10 else None  # Minimum length filter


def create_highlight_data(text: str, confidence: float, 
                         ai_indicators: List[str]) -> Dict[str, Any]:
    """Create data for highlighting AI text"""
    return {
        'text': text,
        'confidence': confidence,
        'indicators': ai_indicators,
        'timestamp': int(asyncio.get_event_loop().time() * 1000) if hasattr(asyncio, 'get_event_loop') else 0,
        'highlight_class': get_confidence_class(confidence)
    }


def get_confidence_class(confidence: float) -> str:
    """Get CSS class based on confidence level"""
    if confidence >= 0.8:
        return 'ai-detected-high'
    elif confidence >= 0.6:
        return 'ai-detected-medium'
    elif confidence >= 0.4:
        return 'ai-detected-low'
    else:
        return 'ai-detected-very-low'


# ============================================
# Analytics Utilities
# ============================================

class AnalyticsCollector:
    """Analytics data collector"""
    
    def __init__(self):
        self.events = []
        self.session_id = f"session_{int(asyncio.get_event_loop().time() * 1000)}" if hasattr(asyncio, 'get_event_loop') else "session_0"
    
    def track_event(self, event_type: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Track an event"""
        event = {
            'type': event_type,
            'data': data or {},
            'timestamp': int(asyncio.get_event_loop().time() * 1000) if hasattr(asyncio, 'get_event_loop') else 0,
            'session_id': self.session_id
        }
        
        self.events.append(event)
        
        # Keep only last 1000 events
        if len(self.events) > 1000:
            self.events = self.events[-1000:]
    
    def get_events(self, event_type: Optional[str] = None, 
                   limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get tracked events"""
        events = self.events
        
        if event_type:
            events = [e for e in events if e['type'] == event_type]
        
        if limit:
            events = events[-limit:]
        
        return events
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get analytics statistics"""
        if not self.events:
            return {}
        
        event_types = {}
        for event in self.events:
            event_type = event['type']
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        return {
            'total_events': len(self.events),
            'event_types': event_types,
            'session_id': self.session_id,
            'first_event': self.events[0]['timestamp'] if self.events else None,
            'last_event': self.events[-1]['timestamp'] if self.events else None
        }


# ============================================
# Error Handling Utilities
# ============================================

class ExtensionError(Exception):
    """Base extension error"""
    def __init__(self, message: str, error_code: Optional[str] = None):
        self.message = message
        self.error_code = error_code
        super().__init__(message)


def create_error_response(error: Union[str, Exception], 
                         request_id: Optional[str] = None) -> Response:
    """Create standardized error response"""
    if isinstance(error, ExtensionError):
        error_msg = f"{error.error_code}: {error.message}" if error.error_code else error.message
    else:
        error_msg = str(error)
    
    return Response.error(error_msg, request_id)


def handle_async_errors(func):
    """Decorator for handling async function errors"""
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            return create_error_response(e)
    return wrapper


# Export all utilities
__all__ = [
    # Message Protocol
    'MessageType', 'Message', 'Response',
    
    # Settings
    'ExtensionSettings',
    
    # Storage
    'StorageManager',
    
    # UI State
    'UIState',
    
    # Content Script
    'extract_text_from_element', 'create_highlight_data', 'get_confidence_class',
    
    # Analytics
    'AnalyticsCollector',
    
    # Error Handling
    'ExtensionError', 'create_error_response', 'handle_async_errors'
]