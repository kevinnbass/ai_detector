"""
Shared Utilities Package
Consolidates common functions used across the AI Detector project
"""

# Import all utilities from submodules
from .common import *
from .ml_utils import *
from .api_utils import *
from .extension_utils import *

__version__ = "1.0.0"
__author__ = "AI Detector Project"

# Convenience imports for most commonly used utilities
from .common import (
    load_json, save_json, normalize_text, clean_text, 
    ensure_directory, Timer, SimpleCache, Config
)

from .ml_utils import (
    evaluate_model, prepare_train_test_split, 
    extract_text_features, save_model, load_model
)

from .api_utils import (
    APIResponse, APIConfig, BaseAPIClient, 
    SyncAPIClient, LLMAPIClient
)

from .extension_utils import (
    Message, Response, MessageType, ExtensionSettings,
    UIState, AnalyticsCollector
)

__all__ = [
    # Most commonly used utilities (convenience imports)
    'load_json', 'save_json', 'normalize_text', 'clean_text',
    'ensure_directory', 'Timer', 'SimpleCache', 'Config',
    'evaluate_model', 'prepare_train_test_split', 'extract_text_features',
    'save_model', 'load_model',
    'APIResponse', 'APIConfig', 'BaseAPIClient', 'SyncAPIClient', 'LLMAPIClient',
    'Message', 'Response', 'MessageType', 'ExtensionSettings',
    'UIState', 'AnalyticsCollector'
]