"""
Extension Interface Definitions
Interfaces for Chrome extension components
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime
from enum import Enum

from .base_interfaces import IInitializable, IConfigurable, IDisposable


class ExtensionContext(Enum):
    """Extension context enumeration"""
    BACKGROUND = "background"
    CONTENT = "content"
    POPUP = "popup"
    OPTIONS = "options"
    DEVTOOLS = "devtools"


class MessageType(Enum):
    """Message type enumeration"""
    DETECTION_REQUEST = "detection_request"
    DETECTION_RESPONSE = "detection_response"
    SETTINGS_UPDATE = "settings_update"
    STATUS_UPDATE = "status_update"
    ERROR_REPORT = "error_report"
    HEALTH_CHECK = "health_check"


class StorageType(Enum):
    """Storage type enumeration"""
    LOCAL = "local"
    SYNC = "sync"
    SESSION = "session"
    MEMORY = "memory"


@dataclass
class ExtensionMessage:
    """Standardized extension message"""
    id: str
    type: MessageType
    data: Any
    source: ExtensionContext
    target: Optional[ExtensionContext] = None
    timestamp: Optional[datetime] = None
    correlation_id: Optional[str] = None


@dataclass
class ExtensionSettings:
    """Extension settings data structure"""
    enabled: bool = True
    detection_threshold: float = 0.7
    show_indicators: bool = True
    auto_analyze: bool = True
    api_endpoint: str = ""
    api_key: str = ""
    cache_enabled: bool = True
    debug_mode: bool = False


class IMessageHandler(IInitializable, ABC):
    """Interface for message handlers"""
    
    @abstractmethod
    async def handle_message(self, message: ExtensionMessage) -> Optional[Any]:
        """Handle incoming message"""
        pass
    
    @abstractmethod
    def can_handle(self, message_type: MessageType) -> bool:
        """Check if can handle message type"""
        pass
    
    @abstractmethod
    def get_supported_types(self) -> List[MessageType]:
        """Get supported message types"""
        pass
    
    @abstractmethod
    def register_handler(self, message_type: MessageType, handler: Callable) -> str:
        """Register message handler"""
        pass
    
    @abstractmethod
    def unregister_handler(self, handler_id: str) -> bool:
        """Unregister message handler"""
        pass


class IMessageBus(IInitializable, ABC):
    """Interface for message bus system"""
    
    @abstractmethod
    async def send(self, message: ExtensionMessage) -> bool:
        """Send message"""
        pass
    
    @abstractmethod
    async def broadcast(self, message: ExtensionMessage) -> Dict[str, bool]:
        """Broadcast message to all contexts"""
        pass
    
    @abstractmethod
    def subscribe(self, message_type: MessageType, callback: Callable) -> str:
        """Subscribe to messages"""
        pass
    
    @abstractmethod
    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from messages"""
        pass
    
    @abstractmethod
    def get_message_stats(self) -> Dict[str, Any]:
        """Get message statistics"""
        pass


class IExtensionStorage(ABC):
    """Interface for extension storage"""
    
    @abstractmethod
    async def get(self, key: str, storage_type: StorageType = StorageType.LOCAL) -> Any:
        """Get value from storage"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, 
                 storage_type: StorageType = StorageType.LOCAL) -> bool:
        """Set value in storage"""
        pass
    
    @abstractmethod
    async def remove(self, key: str, 
                    storage_type: StorageType = StorageType.LOCAL) -> bool:
        """Remove value from storage"""
        pass
    
    @abstractmethod
    async def clear(self, storage_type: StorageType = StorageType.LOCAL) -> bool:
        """Clear storage"""
        pass
    
    @abstractmethod
    def listen_for_changes(self, callback: Callable) -> str:
        """Listen for storage changes"""
        pass
    
    @abstractmethod
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        pass


class IBackgroundScript(IInitializable, IConfigurable, IDisposable, ABC):
    """Interface for background script"""
    
    @abstractmethod
    async def handle_extension_event(self, event_type: str, data: Any) -> None:
        """Handle extension events"""
        pass
    
    @abstractmethod
    async def handle_tab_update(self, tab_id: int, change_info: Dict[str, Any], 
                               tab: Dict[str, Any]) -> None:
        """Handle tab updates"""
        pass
    
    @abstractmethod
    async def handle_web_navigation(self, details: Dict[str, Any]) -> None:
        """Handle web navigation"""
        pass
    
    @abstractmethod
    def get_active_tabs(self) -> List[Dict[str, Any]]:
        """Get active tabs"""
        pass
    
    @abstractmethod
    async def inject_content_script(self, tab_id: int) -> bool:
        """Inject content script"""
        pass


class IContentScript(IInitializable, IConfigurable, ABC):
    """Interface for content script"""
    
    @abstractmethod
    async def scan_page(self, selectors: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Scan page for text elements"""
        pass
    
    @abstractmethod
    async def analyze_element(self, element_id: str) -> Optional[Dict[str, Any]]:
        """Analyze specific element"""
        pass
    
    @abstractmethod
    def add_ui_indicator(self, element_id: str, result: Dict[str, Any]) -> None:
        """Add UI indicator to element"""
        pass
    
    @abstractmethod
    def remove_ui_indicator(self, element_id: str) -> None:
        """Remove UI indicator from element"""
        pass
    
    @abstractmethod
    def observe_dom_changes(self, callback: Callable) -> str:
        """Observe DOM changes"""
        pass
    
    @abstractmethod
    def get_page_info(self) -> Dict[str, Any]:
        """Get page information"""
        pass


class IPopupHandler(IInitializable, IConfigurable, ABC):
    """Interface for popup handler"""
    
    @abstractmethod
    async def update_ui(self, data: Dict[str, Any]) -> None:
        """Update popup UI"""
        pass
    
    @abstractmethod
    def get_current_tab_info(self) -> Optional[Dict[str, Any]]:
        """Get current tab information"""
        pass
    
    @abstractmethod
    async def trigger_analysis(self, tab_id: Optional[int] = None) -> Dict[str, Any]:
        """Trigger analysis"""
        pass
    
    @abstractmethod
    def show_settings(self) -> None:
        """Show settings panel"""
        pass
    
    @abstractmethod
    def show_statistics(self) -> None:
        """Show statistics panel"""
        pass


class IOptionsHandler(IInitializable, IConfigurable, ABC):
    """Interface for options page handler"""
    
    @abstractmethod
    async def load_settings(self) -> ExtensionSettings:
        """Load extension settings"""
        pass
    
    @abstractmethod
    async def save_settings(self, settings: ExtensionSettings) -> bool:
        """Save extension settings"""
        pass
    
    @abstractmethod
    def validate_settings(self, settings: ExtensionSettings) -> tuple[bool, List[str]]:
        """Validate settings"""
        pass
    
    @abstractmethod
    async def test_api_connection(self, endpoint: str, api_key: str) -> bool:
        """Test API connection"""
        pass
    
    @abstractmethod
    def export_settings(self) -> str:
        """Export settings to JSON"""
        pass
    
    @abstractmethod
    async def import_settings(self, settings_json: str) -> bool:
        """Import settings from JSON"""
        pass


class IPermissionManager(ABC):
    """Interface for permission management"""
    
    @abstractmethod
    async def request_permission(self, permission: str) -> bool:
        """Request permission"""
        pass
    
    @abstractmethod
    def has_permission(self, permission: str) -> bool:
        """Check if has permission"""
        pass
    
    @abstractmethod
    def get_all_permissions(self) -> List[str]:
        """Get all granted permissions"""
        pass
    
    @abstractmethod
    async def remove_permission(self, permission: str) -> bool:
        """Remove permission"""
        pass


class IExtensionAnalytics(ABC):
    """Interface for extension analytics"""
    
    @abstractmethod
    def track_event(self, event_name: str, properties: Optional[Dict[str, Any]] = None) -> None:
        """Track analytics event"""
        pass
    
    @abstractmethod
    def track_detection(self, result: Dict[str, Any]) -> None:
        """Track detection event"""
        pass
    
    @abstractmethod
    def track_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """Track error event"""
        pass
    
    @abstractmethod
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        pass
    
    @abstractmethod
    def set_user_property(self, property_name: str, value: Any) -> None:
        """Set user property"""
        pass


class IExtensionUpdater(ABC):
    """Interface for extension updates"""
    
    @abstractmethod
    async def check_for_updates(self) -> Optional[Dict[str, Any]]:
        """Check for updates"""
        pass
    
    @abstractmethod
    def get_current_version(self) -> str:
        """Get current version"""
        pass
    
    @abstractmethod
    async def apply_update(self, update_info: Dict[str, Any]) -> bool:
        """Apply update"""
        pass
    
    @abstractmethod
    def get_update_history(self) -> List[Dict[str, Any]]:
        """Get update history"""
        pass


class IContextMenuHandler(ABC):
    """Interface for context menu handling"""
    
    @abstractmethod
    def create_menu_items(self) -> None:
        """Create context menu items"""
        pass
    
    @abstractmethod
    async def handle_menu_click(self, info: Dict[str, Any], tab: Dict[str, Any]) -> None:
        """Handle context menu click"""
        pass
    
    @abstractmethod
    def update_menu_items(self, enabled: bool) -> None:
        """Update menu items state"""
        pass


class IWebRequestHandler(ABC):
    """Interface for web request handling"""
    
    @abstractmethod
    def on_before_request(self, details: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle before request"""
        pass
    
    @abstractmethod
    def on_before_send_headers(self, details: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle before send headers"""
        pass
    
    @abstractmethod
    def on_response_started(self, details: Dict[str, Any]) -> None:
        """Handle response started"""
        pass
    
    @abstractmethod
    def on_completed(self, details: Dict[str, Any]) -> None:
        """Handle request completed"""
        pass


class IExtensionLogger(ABC):
    """Interface for extension logging"""
    
    @abstractmethod
    def log(self, level: str, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Log message"""
        pass
    
    @abstractmethod
    def get_logs(self, level: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get logs"""
        pass
    
    @abstractmethod
    def clear_logs(self) -> None:
        """Clear logs"""
        pass
    
    @abstractmethod
    def set_log_level(self, level: str) -> None:
        """Set log level"""
        pass