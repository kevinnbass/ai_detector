"""
Common Utilities Library - Shared Functions Across All Modules
Consolidates duplicate utility functions from across the codebase
"""

import json
import os
import re
import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
import logging

# Configure logging
logger = logging.getLogger(__name__)


# ============================================
# File I/O Utilities
# ============================================

def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if not"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(filepath: Union[str, Path], default: Any = None) -> Any:
    """Load JSON file with error handling"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Error loading {filepath}: {e}")
        return default if default is not None else {}


def save_json(data: Any, filepath: Union[str, Path], indent: int = 2) -> bool:
    """Save data to JSON file with error handling"""
    try:
        ensure_directory(Path(filepath).parent)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Error saving to {filepath}: {e}")
        return False


def read_file(filepath: Union[str, Path], encoding: str = 'utf-8') -> Optional[str]:
    """Read text file with error handling"""
    try:
        with open(filepath, 'r', encoding=encoding) as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading {filepath}: {e}")
        return None


def write_file(content: str, filepath: Union[str, Path], encoding: str = 'utf-8') -> bool:
    """Write text file with error handling"""
    try:
        ensure_directory(Path(filepath).parent)
        with open(filepath, 'w', encoding=encoding) as f:
            f.write(content)
        return True
    except Exception as e:
        logger.error(f"Error writing to {filepath}: {e}")
        return False


# ============================================
# Text Processing Utilities
# ============================================

def normalize_text(text: str) -> str:
    """Normalize text for consistent processing"""
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    # Remove zero-width characters
    text = text.replace('\u200b', '').replace('\ufeff', '')
    return text.strip()


def clean_text(text: str, remove_urls: bool = True, remove_mentions: bool = False) -> str:
    """Clean text for analysis"""
    # Remove URLs
    if remove_urls:
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove mentions (Twitter/X style)
    if remove_mentions:
        text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags symbols but keep text
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove multiple spaces
    text = ' '.join(text.split())
    
    return text.strip()


def extract_sentences(text: str) -> List[str]:
    """Extract sentences from text"""
    # Simple sentence splitting
    sentences = re.split(r'[.!?]+', text)
    # Clean and filter
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def tokenize(text: str, lowercase: bool = True) -> List[str]:
    """Simple word tokenization"""
    if lowercase:
        text = text.lower()
    # Extract words (alphanumeric)
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


def calculate_text_hash(text: str) -> str:
    """Calculate hash of text for deduplication"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


# ============================================
# Validation Utilities
# ============================================

def validate_json_schema(data: Any, required_fields: List[str]) -> Tuple[bool, List[str]]:
    """Validate if data has required fields"""
    errors = []
    
    if not isinstance(data, dict):
        return False, ["Data must be a dictionary"]
    
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
    
    return len(errors) == 0, errors


def validate_range(value: Union[int, float], min_val: Optional[float] = None, 
                  max_val: Optional[float] = None, name: str = "value") -> Tuple[bool, Optional[str]]:
    """Validate if value is within range"""
    if min_val is not None and value < min_val:
        return False, f"{name} must be >= {min_val}"
    if max_val is not None and value > max_val:
        return False, f"{name} must be <= {max_val}"
    return True, None


def validate_probability(value: float, name: str = "probability") -> Tuple[bool, Optional[str]]:
    """Validate probability value (0-1)"""
    return validate_range(value, 0.0, 1.0, name)


# ============================================
# Data Processing Utilities
# ============================================

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split list into chunks"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """Flatten nested dictionary"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple dictionaries"""
    result = {}
    for d in dicts:
        result.update(d)
    return result


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default for zero denominator"""
    if denominator == 0:
        return default
    return numerator / denominator


# ============================================
# Time Utilities
# ============================================

def get_timestamp(format_str: str = "%Y%m%d_%H%M%S") -> str:
    """Get formatted timestamp"""
    return datetime.now().strftime(format_str)


def measure_time(func):
    """Decorator to measure function execution time"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.debug(f"{func.__name__} took {elapsed:.2f} seconds")
        return result
    return wrapper


class Timer:
    """Context manager for timing code blocks"""
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start = None
        
    def __enter__(self):
        self.start = time.time()
        return self
        
    def __exit__(self, *args):
        elapsed = time.time() - self.start
        logger.info(f"{self.name} took {elapsed:.2f} seconds")


# ============================================
# Caching Utilities
# ============================================

class SimpleCache:
    """Simple in-memory cache with TTL"""
    def __init__(self, ttl_seconds: int = 900):
        self.cache = {}
        self.ttl = ttl_seconds
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired"""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            else:
                del self.cache[key]
        return None
        
    def set(self, key: str, value: Any) -> None:
        """Set value in cache"""
        self.cache[key] = (value, time.time())
        
    def clear(self) -> None:
        """Clear cache"""
        self.cache.clear()


# ============================================
# Rate Limiting Utilities
# ============================================

class RateLimiter:
    """Simple rate limiter"""
    def __init__(self, max_calls: int, time_window: int):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
        
    def allow(self) -> bool:
        """Check if operation is allowed"""
        now = time.time()
        # Remove old calls outside window
        self.calls = [t for t in self.calls if now - t < self.time_window]
        
        if len(self.calls) < self.max_calls:
            self.calls.append(now)
            return True
        return False
        
    def wait_time(self) -> float:
        """Get time to wait before next allowed call"""
        if len(self.calls) < self.max_calls:
            return 0
        oldest = self.calls[0]
        wait = self.time_window - (time.time() - oldest)
        return max(0, wait)


# ============================================
# Pattern Matching Utilities
# ============================================

def compile_patterns(patterns: List[str]) -> List[re.Pattern]:
    """Compile regex patterns with error handling"""
    compiled = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(pattern, re.IGNORECASE))
        except re.error as e:
            logger.warning(f"Invalid regex pattern '{pattern}': {e}")
    return compiled


def find_patterns(text: str, patterns: List[re.Pattern]) -> Dict[str, int]:
    """Find all pattern matches in text"""
    matches = {}
    for pattern in patterns:
        found = pattern.findall(text)
        if found:
            matches[pattern.pattern] = len(found)
    return matches


# ============================================
# Statistics Utilities
# ============================================

def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """Calculate basic statistics"""
    if not values:
        return {
            'count': 0,
            'mean': 0,
            'min': 0,
            'max': 0,
            'sum': 0
        }
    
    return {
        'count': len(values),
        'mean': sum(values) / len(values),
        'min': min(values),
        'max': max(values),
        'sum': sum(values)
    }


def calculate_percentile(values: List[float], percentile: float) -> float:
    """Calculate percentile value"""
    if not values:
        return 0
    sorted_values = sorted(values)
    index = int(len(sorted_values) * percentile / 100)
    return sorted_values[min(index, len(sorted_values) - 1)]


# ============================================
# Error Handling Utilities
# ============================================

def safe_execute(func, default=None, log_errors=True):
    """Execute function with error handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if log_errors:
                logger.error(f"Error in {func.__name__}: {e}")
            return default
    return wrapper


class RetryManager:
    """Retry failed operations with exponential backoff"""
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        
    def execute(self, func, *args, **kwargs):
        """Execute function with retries"""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                    time.sleep(delay)
        
        raise last_error


# ============================================
# Configuration Utilities
# ============================================

class Config:
    """Simple configuration manager"""
    def __init__(self, config_file: Optional[str] = None):
        self.config = {}
        if config_file:
            self.load(config_file)
            
    def load(self, config_file: str) -> None:
        """Load configuration from file"""
        self.config = load_json(config_file, {})
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value
        
    def set(self, key: str, value: Any) -> None:
        """Set configuration value"""
        keys = key.split('.')
        d = self.config
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value


# ============================================
# Export all utilities
# ============================================

__all__ = [
    # File I/O
    'ensure_directory', 'load_json', 'save_json', 'read_file', 'write_file',
    
    # Text Processing
    'normalize_text', 'clean_text', 'extract_sentences', 'tokenize', 'calculate_text_hash',
    
    # Validation
    'validate_json_schema', 'validate_range', 'validate_probability',
    
    # Data Processing
    'chunk_list', 'flatten_dict', 'merge_dicts', 'safe_divide',
    
    # Time
    'get_timestamp', 'measure_time', 'Timer',
    
    # Caching
    'SimpleCache',
    
    # Rate Limiting
    'RateLimiter',
    
    # Pattern Matching
    'compile_patterns', 'find_patterns',
    
    # Statistics
    'calculate_statistics', 'calculate_percentile',
    
    # Error Handling
    'safe_execute', 'RetryManager',
    
    # Configuration
    'Config'
]