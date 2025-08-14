"""
AI Detector - Core Package
"""

__version__ = "2.1.0"
__author__ = "AI Detector Team"

from . import core
from . import data
from . import training
from . import api
from . import integrations

__all__ = ['core', 'data', 'training', 'api', 'integrations']