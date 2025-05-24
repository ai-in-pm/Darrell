"""
Darrell - Sophisticated AI Agent for Autonomous Zoom Meeting Attendance
Advanced Voice Synthesis & Meeting Automation System

Copyright (c) 2025 - AI in PM
Licensed under Apache 2.0
"""

__version__ = "1.0.0"
__author__ = "AI in PM"
__description__ = "Sophisticated AI Agent for Autonomous Zoom Meeting Attendance with Voice Cloning & Natural Conversation"

from .core.darrell_core import DarrellAgent
from .core.meeting_coordinator import MeetingCoordinator
from .core.conversation_engine import ConversationEngine
from .voice.elevenlabs_integration import ElevenLabsVoiceCloner
from .voice.audio_processor import AudioProcessor
from .voice.speech_recognition import SpeechRecognitionEngine
from .vision.florence_integration import Florence2Integration
from .vision.zoom_ui_detector import ZoomUIDetector
from .vision.meeting_analyzer import MeetingAnalyzer
from .automation.zoom_controller import ZoomController
from .automation.meeting_navigator import MeetingNavigator
from .automation.credential_manager import CredentialManager
from .conversation.context_manager import MeetingContextManager
from .conversation.response_generator import ResponseGenerator
from .conversation.participant_tracker import ParticipantTracker
from .utils.config import DarrellConfig
from .utils.logger import DarrellLogger
from .utils.security import SecurityManager

__all__ = [
    "DarrellAgent",
    "MeetingCoordinator", 
    "ConversationEngine",
    "ElevenLabsVoiceCloner",
    "AudioProcessor",
    "SpeechRecognitionEngine",
    "Florence2Integration",
    "ZoomUIDetector",
    "MeetingAnalyzer",
    "ZoomController",
    "MeetingNavigator",
    "CredentialManager",
    "MeetingContextManager",
    "ResponseGenerator",
    "ParticipantTracker",
    "DarrellConfig",
    "DarrellLogger",
    "SecurityManager"
]

# Version information
VERSION_INFO = {
    "major": 1,
    "minor": 0,
    "patch": 0,
    "release": "stable"
}

def get_version():
    """Get the current version of Darrell Agent"""
    return f"{VERSION_INFO['major']}.{VERSION_INFO['minor']}.{VERSION_INFO['patch']}"

def get_build_info():
    """Get build information"""
    return {
        "version": get_version(),
        "release": VERSION_INFO["release"],
        "description": __description__,
        "author": __author__
    }
