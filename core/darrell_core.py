"""
Darrell Agent Core - Main orchestration system for autonomous Zoom meeting attendance
Integrates xLAM, Florence-2, ElevenLabs, and meeting automation capabilities
"""

import asyncio
import time
import threading
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

# Import xLAM components
try:
    from xLAM.client import xLAMChatCompletion, xLAMConfig
except ImportError:
    print("Warning: xLAM not available. Please install xLAM package.")
    xLAMChatCompletion = None
    xLAMConfig = None

from ..utils.config import DarrellConfig
from ..utils.logger import DarrellLogger, LogCategory
from ..utils.security import SecurityManager
from .meeting_coordinator import MeetingCoordinator
from .conversation_engine import ConversationEngine
from ..voice.elevenlabs_integration import ElevenLabsVoiceCloner
from ..voice.audio_processor import AudioProcessor
from ..voice.speech_recognition import SpeechRecognitionEngine
from ..vision.florence_integration import Florence2Integration
from ..vision.zoom_ui_detector import ZoomUIDetector
from ..automation.zoom_controller import ZoomController
from ..automation.credential_manager import CredentialManager as AutoCredentialManager
from ..conversation.context_manager import MeetingContextManager
from ..conversation.response_generator import ResponseGenerator


@dataclass
class MeetingSession:
    """Meeting session information"""
    meeting_id: str
    meeting_url: str
    meeting_password: Optional[str]
    start_time: datetime
    duration_minutes: int
    participants: List[str]
    agenda: Optional[str]
    session_id: str
    status: str = "pending"  # pending, active, completed, failed


@dataclass
class AgentState:
    """Current state of Darrell Agent"""
    is_active: bool = False
    current_meeting: Optional[MeetingSession] = None
    voice_ready: bool = False
    automation_ready: bool = False
    ai_models_ready: bool = False
    last_activity: Optional[datetime] = None
    performance_metrics: Dict[str, Any] = None


class DarrellAgent:
    """
    Darrell - Sophisticated AI Agent for Autonomous Zoom Meeting Attendance
    
    Main orchestrator for multi-modal AI agent with voice synthesis, meeting automation,
    and natural conversation capabilities using xLAM and Florence-2.
    """
    
    def __init__(self, config: DarrellConfig):
        """Initialize Darrell Agent with comprehensive configuration"""
        
        self.config = config
        self.logger = DarrellLogger("DarrellCore", 
                                   log_dir=config.get_full_path(config.logs_path),
                                   log_level=config.log_level)
        
        # Agent state
        self.state = AgentState()
        self.session_id = self._generate_session_id()
        self.is_shutting_down = False
        
        # Core components (initialized in initialize())
        self.security_manager = None
        self.xlam_client = None
        self.meeting_coordinator = None
        self.conversation_engine = None
        self.voice_cloner = None
        self.audio_processor = None
        self.speech_recognition = None
        self.florence_integration = None
        self.zoom_ui_detector = None
        self.zoom_controller = None
        self.credential_manager = None
        self.context_manager = None
        self.response_generator = None
        
        # Threading and synchronization
        self._main_loop_task = None
        self._shutdown_event = asyncio.Event()
        self._component_lock = asyncio.Lock()
        
        self.logger.info("Darrell Agent initialized", LogCategory.GENERAL,
                        performance_data={"session_id": self.session_id})
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        return f"darrell_{int(time.time())}_{id(self)}"
    
    async def initialize(self) -> bool:
        """Initialize all Darrell Agent subsystems"""
        
        try:
            self.logger.info("Initializing Darrell Agent subsystems...", LogCategory.GENERAL)
            
            # Validate configuration
            config_errors = self.config.validate()
            if config_errors:
                for error in config_errors:
                    self.logger.error(f"Configuration error: {error}", LogCategory.GENERAL)
                return False
            
            # Initialize security manager
            await self._initialize_security()
            
            # Initialize xLAM client
            await self._initialize_xlam()
            
            # Initialize AI models
            await self._initialize_ai_models()
            
            # Initialize voice systems
            await self._initialize_voice_systems()
            
            # Initialize automation systems
            await self._initialize_automation_systems()
            
            # Initialize conversation systems
            await self._initialize_conversation_systems()
            
            # Initialize coordinators
            await self._initialize_coordinators()
            
            # Start main processing loop
            self._main_loop_task = asyncio.create_task(self._main_processing_loop())
            
            self.state.is_active = True
            self.state.last_activity = datetime.now()
            
            self.logger.info("Darrell Agent initialization completed successfully", 
                           LogCategory.GENERAL,
                           performance_data={"initialization_time": time.time()})
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Darrell Agent: {e}", 
                            LogCategory.GENERAL, error=e)
            return False
    
    async def _initialize_security(self):
        """Initialize security manager"""
        self.security_manager = SecurityManager(self.config.security.to_dict())
        self.logger.info("Security manager initialized", LogCategory.SECURITY)
    
    async def _initialize_xlam(self):
        """Initialize xLAM client"""
        if xLAMConfig and xLAMChatCompletion:
            self.xlam_config = xLAMConfig(
                base_url=self.config.ai_models.xlam_base_url,
                model=self.config.ai_models.xlam_model
            )
            self.xlam_client = xLAMChatCompletion.from_config(self.xlam_config)
            self.logger.info("xLAM client initialized", LogCategory.AI_MODEL)
        else:
            self.logger.warning("xLAM not available - using fallback decision making", 
                              LogCategory.AI_MODEL)
    
    async def _initialize_ai_models(self):
        """Initialize AI models (Florence-2, etc.)"""
        try:
            # Initialize Florence-2 integration
            self.florence_integration = Florence2Integration(
                model_path=self.config.ai_models.florence_model_path,
                device=self.config.ai_models.florence_device,
                precision=self.config.ai_models.florence_precision
            )
            await self.florence_integration.initialize()
            
            # Initialize Zoom UI detector
            self.zoom_ui_detector = ZoomUIDetector(self.florence_integration)
            await self.zoom_ui_detector.initialize()
            
            self.state.ai_models_ready = True
            self.logger.info("AI models initialized", LogCategory.AI_MODEL)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize AI models: {e}", 
                            LogCategory.AI_MODEL, error=e)
            raise
    
    async def _initialize_voice_systems(self):
        """Initialize voice synthesis and recognition systems"""
        try:
            # Initialize ElevenLabs voice cloner
            self.voice_cloner = ElevenLabsVoiceCloner(
                api_key=self.config.voice.elevenlabs_api_key,
                voice_id=self.config.voice.voice_id,
                config=self.config.voice
            )
            await self.voice_cloner.initialize()
            
            # Initialize audio processor
            self.audio_processor = AudioProcessor(
                sample_rate=self.config.voice.sample_rate,
                chunk_size=self.config.voice.chunk_size,
                default_microphone=self.config.meeting.default_microphone,
                default_speaker=self.config.meeting.default_speaker
            )
            await self.audio_processor.initialize()
            
            # Initialize speech recognition
            self.speech_recognition = SpeechRecognitionEngine(
                model=self.config.ai_models.whisper_model,
                device=self.config.ai_models.florence_device
            )
            await self.speech_recognition.initialize()
            
            self.state.voice_ready = True
            self.logger.info("Voice systems initialized", LogCategory.VOICE)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize voice systems: {e}", 
                            LogCategory.VOICE, error=e)
            raise
    
    async def _initialize_automation_systems(self):
        """Initialize automation and control systems"""
        try:
            # Initialize credential manager
            self.credential_manager = AutoCredentialManager(
                storage_path=self.config.security.credential_storage_path,
                encrypt_credentials=self.config.security.encrypt_credentials
            )
            
            # Initialize Zoom controller
            self.zoom_controller = ZoomController(
                executable_path=self.config.meeting.zoom_executable_path,
                data_dir=self.config.meeting.zoom_data_dir,
                ui_detector=self.zoom_ui_detector,
                config=self.config.automation
            )
            await self.zoom_controller.initialize()
            
            self.state.automation_ready = True
            self.logger.info("Automation systems initialized", LogCategory.AUTOMATION)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize automation systems: {e}", 
                            LogCategory.AUTOMATION, error=e)
            raise
    
    async def _initialize_conversation_systems(self):
        """Initialize conversation and context management"""
        try:
            # Initialize context manager
            self.context_manager = MeetingContextManager(
                max_context_length=self.config.ai_models.max_context_length,
                retention_policy=self.config.security.data_retention_days
            )
            
            # Initialize response generator
            self.response_generator = ResponseGenerator(
                conversation_model=self.config.ai_models.conversation_model,
                api_key=self.config.ai_models.conversation_api_key,
                politeness_level=self.config.meeting.politeness_level,
                participation_frequency=self.config.meeting.participation_frequency
            )
            await self.response_generator.initialize()
            
            self.logger.info("Conversation systems initialized", LogCategory.CONVERSATION)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize conversation systems: {e}", 
                            LogCategory.CONVERSATION, error=e)
            raise
    
    async def _initialize_coordinators(self):
        """Initialize coordination engines"""
        try:
            # Initialize conversation engine
            self.conversation_engine = ConversationEngine(
                speech_recognition=self.speech_recognition,
                response_generator=self.response_generator,
                voice_cloner=self.voice_cloner,
                audio_processor=self.audio_processor,
                context_manager=self.context_manager
            )
            await self.conversation_engine.initialize()
            
            # Initialize meeting coordinator
            self.meeting_coordinator = MeetingCoordinator(
                zoom_controller=self.zoom_controller,
                conversation_engine=self.conversation_engine,
                credential_manager=self.credential_manager,
                config=self.config.meeting
            )
            await self.meeting_coordinator.initialize()
            
            self.logger.info("Coordinators initialized", LogCategory.GENERAL)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize coordinators: {e}", 
                            LogCategory.GENERAL, error=e)
            raise
    
    async def _main_processing_loop(self):
        """Main processing loop for continuous operation"""
        self.logger.info("Starting main processing loop", LogCategory.GENERAL)
        
        while not self._shutdown_event.is_set():
            try:
                # Update agent state
                self.state.last_activity = datetime.now()
                
                # Process any pending meetings
                if self.state.current_meeting:
                    await self._process_active_meeting()
                
                # Perform maintenance tasks
                await self._perform_maintenance()
                
                # Sleep briefly to prevent busy waiting
                await asyncio.sleep(1.0)
                
            except Exception as e:
                self.logger.error(f"Error in main processing loop: {e}", 
                                LogCategory.GENERAL, error=e)
                await asyncio.sleep(5.0)  # Wait longer on error
        
        self.logger.info("Main processing loop stopped", LogCategory.GENERAL)
    
    async def _process_active_meeting(self):
        """Process currently active meeting"""
        if not self.state.current_meeting:
            return
        
        meeting = self.state.current_meeting
        
        # Check if meeting should still be active
        if meeting.status == "active":
            # Let meeting coordinator handle the meeting
            await self.meeting_coordinator.process_meeting_tick(meeting)
        
        # Check if meeting has ended
        end_time = meeting.start_time + timedelta(minutes=meeting.duration_minutes)
        if datetime.now() > end_time:
            await self._end_meeting(meeting.meeting_id)
    
    async def _perform_maintenance(self):
        """Perform periodic maintenance tasks"""
        # Clean up old logs
        if hasattr(self.logger, 'cleanup_old_logs'):
            self.logger.cleanup_old_logs(self.config.security.data_retention_days)
        
        # Clean up security sessions
        if self.security_manager:
            self.security_manager.cleanup()
        
        # Update performance metrics
        await self._update_performance_metrics()
    
    async def _update_performance_metrics(self):
        """Update performance metrics"""
        metrics = {
            "uptime": (datetime.now() - self.state.last_activity).total_seconds() if self.state.last_activity else 0,
            "voice_ready": self.state.voice_ready,
            "automation_ready": self.state.automation_ready,
            "ai_models_ready": self.state.ai_models_ready,
            "active_meeting": self.state.current_meeting is not None
        }
        
        self.state.performance_metrics = metrics
        self.logger.log_performance_metrics(metrics)
    
    async def join_meeting(self, meeting_url: str, meeting_password: str = None,
                          duration_minutes: int = 60, agenda: str = None) -> bool:
        """
        Join a Zoom meeting
        
        Args:
            meeting_url: Zoom meeting URL or ID
            meeting_password: Meeting password if required
            duration_minutes: Expected meeting duration
            agenda: Meeting agenda for context
        
        Returns:
            Success status
        """
        try:
            # Create meeting session
            meeting_session = MeetingSession(
                meeting_id=self._extract_meeting_id(meeting_url),
                meeting_url=meeting_url,
                meeting_password=meeting_password,
                start_time=datetime.now(),
                duration_minutes=duration_minutes,
                participants=[],
                agenda=agenda,
                session_id=self.session_id,
                status="pending"
            )
            
            self.logger.log_meeting_event("Meeting join initiated", {
                "meeting_id": meeting_session.meeting_id,
                "duration": duration_minutes
            })
            
            # Use meeting coordinator to join
            success = await self.meeting_coordinator.join_meeting(meeting_session)
            
            if success:
                self.state.current_meeting = meeting_session
                meeting_session.status = "active"
                self.logger.log_meeting_event("Meeting joined successfully", {
                    "meeting_id": meeting_session.meeting_id
                })
            else:
                meeting_session.status = "failed"
                self.logger.log_meeting_event("Meeting join failed", {
                    "meeting_id": meeting_session.meeting_id
                })
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to join meeting: {e}", LogCategory.MEETING, error=e)
            return False
    
    def _extract_meeting_id(self, meeting_url: str) -> str:
        """Extract meeting ID from URL"""
        import re
        
        # Try to extract meeting ID from various Zoom URL formats
        patterns = [
            r'/j/(\d+)',  # zoom.us/j/123456789
            r'meeting_id=(\d+)',  # meeting_id=123456789
            r'/(\d{9,11})',  # Direct meeting ID
        ]
        
        for pattern in patterns:
            match = re.search(pattern, meeting_url)
            if match:
                return match.group(1)
        
        # If no pattern matches, use the URL as ID
        return meeting_url.replace('/', '_').replace(':', '_')
    
    async def _end_meeting(self, meeting_id: str):
        """End current meeting"""
        if self.state.current_meeting and self.state.current_meeting.meeting_id == meeting_id:
            await self.meeting_coordinator.leave_meeting(self.state.current_meeting)
            self.state.current_meeting.status = "completed"
            self.state.current_meeting = None
            
            self.logger.log_meeting_event("Meeting ended", {"meeting_id": meeting_id})
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            "is_active": self.state.is_active,
            "session_id": self.session_id,
            "current_meeting": self.state.current_meeting.meeting_id if self.state.current_meeting else None,
            "voice_ready": self.state.voice_ready,
            "automation_ready": self.state.automation_ready,
            "ai_models_ready": self.state.ai_models_ready,
            "last_activity": self.state.last_activity.isoformat() if self.state.last_activity else None,
            "performance_metrics": self.state.performance_metrics or {}
        }
    
    async def stop(self):
        """Stop Darrell Agent gracefully"""
        self.logger.info("Stopping Darrell Agent...", LogCategory.GENERAL)
        
        self.is_shutting_down = True
        self._shutdown_event.set()
        
        # End current meeting if active
        if self.state.current_meeting:
            await self._end_meeting(self.state.current_meeting.meeting_id)
        
        # Stop main loop
        if self._main_loop_task:
            self._main_loop_task.cancel()
            try:
                await self._main_loop_task
            except asyncio.CancelledError:
                pass
        
        # Cleanup components
        await self._cleanup_components()
        
        self.state.is_active = False
        self.logger.info("Darrell Agent stopped", LogCategory.GENERAL)
    
    async def _cleanup_components(self):
        """Cleanup all components"""
        components = [
            self.meeting_coordinator,
            self.conversation_engine,
            self.voice_cloner,
            self.audio_processor,
            self.speech_recognition,
            self.florence_integration,
            self.zoom_controller
        ]
        
        for component in components:
            if component and hasattr(component, 'cleanup'):
                try:
                    await component.cleanup()
                except Exception as e:
                    self.logger.error(f"Error cleaning up component {component}: {e}", 
                                    LogCategory.GENERAL, error=e)
