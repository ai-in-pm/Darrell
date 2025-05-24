"""
Computer vision and UI detection components for Darrell Agent
Florence-2 integration, Zoom UI detection, and meeting analysis
"""

from .florence_integration import Florence2Integration
from .zoom_ui_detector import ZoomUIDetector
from .meeting_analyzer import MeetingAnalyzer

__all__ = [
    "Florence2Integration",
    "ZoomUIDetector", 
    "MeetingAnalyzer"
]
