"""
Meeting Coordinator for Darrell Agent
Orchestrates meeting joining, participation, and management
"""

import asyncio
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from ..utils.logger import DarrellLogger, LogCategory
from ..utils.config import MeetingConfig


class MeetingState(Enum):
    """Meeting state enumeration"""
    IDLE = "idle"
    JOINING = "joining"
    ACTIVE = "active"
    LEAVING = "leaving"
    ERROR = "error"


@dataclass
class MeetingAction:
    """Meeting action to be performed"""
    action_type: str
    parameters: Dict[str, Any]
    priority: int = 1
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class MeetingCoordinator:
    """
    Coordinates all meeting-related activities for Darrell Agent
    Manages meeting lifecycle, participant interaction, and automation
    """
    
    def __init__(self, zoom_controller, conversation_engine, credential_manager, config: MeetingConfig):
        """
        Initialize meeting coordinator
        
        Args:
            zoom_controller: Zoom automation controller
            conversation_engine: Conversation management engine
            credential_manager: Credential management system
            config: Meeting configuration
        """
        self.zoom_controller = zoom_controller
        self.conversation_engine = conversation_engine
        self.credential_manager = credential_manager
        self.config = config
        self.logger = DarrellLogger("MeetingCoordinator")
        
        # Meeting state
        self.current_state = MeetingState.IDLE
        self.current_meeting = None
        self.meeting_start_time = None
        self.last_activity_time = None
        
        # Action queue
        self.action_queue = asyncio.Queue()
        self.action_processor_task = None
        
        # Participation tracking
        self.participation_stats = {
            'messages_heard': 0,
            'responses_given': 0,
            'questions_answered': 0,
            'total_speaking_time': 0.0,
            'last_participation': None
        }
        
        # Meeting context
        self.meeting_context = {
            'participants': [],
            'current_speaker': None,
            'meeting_topic': None,
            'agenda_items': [],
            'action_items': []
        }
        
        # Timing controls
        self.last_response_time = None
        self.response_cooldown = 30.0  # Minimum seconds between responses
        
        self.is_initialized = False
    
    async def initialize(self) -> bool:
        """Initialize meeting coordinator"""
        try:
            self.logger.info("Initializing meeting coordinator...", LogCategory.MEETING)
            
            # Start action processor
            self.action_processor_task = asyncio.create_task(self._process_actions())
            
            self.is_initialized = True
            self.logger.info("Meeting coordinator initialized", LogCategory.MEETING)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize meeting coordinator: {e}", 
                            LogCategory.MEETING, error=e)
            return False
    
    async def join_meeting(self, meeting_session) -> bool:
        """
        Join a Zoom meeting
        
        Args:
            meeting_session: Meeting session information
        
        Returns:
            Success status
        """
        try:
            self.logger.log_meeting_event("Starting meeting join process", {
                "meeting_id": meeting_session.meeting_id,
                "meeting_url": meeting_session.meeting_url
            })
            
            self.current_state = MeetingState.JOINING
            self.current_meeting = meeting_session
            
            # Step 1: Launch Zoom and navigate to meeting
            zoom_success = await self.zoom_controller.join_meeting(
                meeting_session.meeting_url,
                meeting_session.meeting_password
            )
            
            if not zoom_success:
                self.logger.error("Failed to join meeting via Zoom controller", LogCategory.MEETING)
                self.current_state = MeetingState.ERROR
                return False
            
            # Step 2: Configure audio/video settings
            await self._configure_meeting_settings()
            
            # Step 3: Initialize conversation monitoring
            await self.conversation_engine.start_meeting_monitoring()
            
            # Step 4: Set up meeting context
            await self._initialize_meeting_context(meeting_session)
            
            # Step 5: Begin participation
            await self._start_meeting_participation()
            
            self.current_state = MeetingState.ACTIVE
            self.meeting_start_time = datetime.now()
            self.last_activity_time = datetime.now()
            
            self.logger.log_meeting_event("Meeting joined successfully", {
                "meeting_id": meeting_session.meeting_id,
                "join_duration": (datetime.now() - meeting_session.start_time).total_seconds()
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to join meeting: {e}", LogCategory.MEETING, error=e)
            self.current_state = MeetingState.ERROR
            return False
    
    async def _configure_meeting_settings(self):
        """Configure meeting audio/video settings"""
        try:
            # Mute microphone initially
            await self.zoom_controller.mute_microphone()
            
            # Turn off video if configured
            if not self.config.auto_join_video:
                await self.zoom_controller.turn_off_video()
            
            # Join audio if configured
            if self.config.auto_join_audio:
                await self.zoom_controller.join_audio()
            
            self.logger.info("Meeting settings configured", LogCategory.MEETING)
            
        except Exception as e:
            self.logger.error(f"Failed to configure meeting settings: {e}", 
                            LogCategory.MEETING, error=e)
    
    async def _initialize_meeting_context(self, meeting_session):
        """Initialize meeting context and understanding"""
        try:
            # Set meeting context in conversation engine
            await self.conversation_engine.set_meeting_context({
                'meeting_id': meeting_session.meeting_id,
                'agenda': meeting_session.agenda,
                'expected_duration': meeting_session.duration_minutes,
                'participants': meeting_session.participants
            })
            
            # Initialize meeting topic understanding
            if meeting_session.agenda:
                self.meeting_context['meeting_topic'] = meeting_session.agenda
                self.meeting_context['agenda_items'] = self._parse_agenda(meeting_session.agenda)
            
            self.logger.info("Meeting context initialized", LogCategory.MEETING)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize meeting context: {e}", 
                            LogCategory.MEETING, error=e)
    
    def _parse_agenda(self, agenda: str) -> List[str]:
        """Parse meeting agenda into items"""
        # Simple agenda parsing - could be enhanced with NLP
        items = []
        for line in agenda.split('\n'):
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('*') or line.startswith('1.')):
                items.append(line)
        return items
    
    async def _start_meeting_participation(self):
        """Start active meeting participation"""
        try:
            # Begin listening for conversation
            await self.conversation_engine.start_listening()
            
            # Schedule initial greeting (after a brief delay)
            await asyncio.sleep(5.0)  # Wait for meeting to stabilize
            
            if self._should_participate():
                await self._queue_action(MeetingAction(
                    action_type="initial_greeting",
                    parameters={"message": "Hello everyone, I've joined the meeting."},
                    priority=1
                ))
            
            self.logger.info("Meeting participation started", LogCategory.MEETING)
            
        except Exception as e:
            self.logger.error(f"Failed to start meeting participation: {e}", 
                            LogCategory.MEETING, error=e)
    
    async def process_meeting_tick(self, meeting_session):
        """Process periodic meeting updates"""
        try:
            if self.current_state != MeetingState.ACTIVE:
                return
            
            # Update activity time
            self.last_activity_time = datetime.now()
            
            # Check for new conversation
            await self._process_conversation_updates()
            
            # Check if we should participate
            await self._evaluate_participation_opportunity()
            
            # Update meeting context
            await self._update_meeting_context()
            
            # Check meeting health
            await self._check_meeting_health()
            
        except Exception as e:
            self.logger.error(f"Error in meeting tick processing: {e}", 
                            LogCategory.MEETING, error=e)
    
    async def _process_conversation_updates(self):
        """Process new conversation from the meeting"""
        try:
            # Get recent conversation from conversation engine
            recent_messages = await self.conversation_engine.get_recent_messages()
            
            for message in recent_messages:
                await self._process_conversation_message(message)
                
        except Exception as e:
            self.logger.error(f"Error processing conversation updates: {e}", 
                            LogCategory.CONVERSATION, error=e)
    
    async def _process_conversation_message(self, message: Dict[str, Any]):
        """Process a single conversation message"""
        try:
            speaker = message.get('speaker', 'Unknown')
            text = message.get('text', '')
            timestamp = message.get('timestamp', datetime.now())
            
            # Update participation stats
            self.participation_stats['messages_heard'] += 1
            
            # Update current speaker
            self.meeting_context['current_speaker'] = speaker
            
            # Check if this is a question directed at us
            if self._is_question_for_us(text):
                await self._handle_direct_question(text, speaker)
            
            # Check if we should contribute to the conversation
            elif self._should_contribute_to_topic(text):
                await self._consider_contribution(text, speaker)
            
            # Log conversation activity
            self.logger.log_conversation_turn(speaker, len(text))
            
        except Exception as e:
            self.logger.error(f"Error processing conversation message: {e}", 
                            LogCategory.CONVERSATION, error=e)
    
    def _is_question_for_us(self, text: str) -> bool:
        """Determine if a question is directed at us"""
        # Look for direct mentions or questions that might be for us
        our_indicators = ['darrell', 'ai', 'agent', 'you', 'your opinion']
        question_indicators = ['?', 'what do you think', 'what about', 'how about']
        
        text_lower = text.lower()
        
        has_our_indicator = any(indicator in text_lower for indicator in our_indicators)
        has_question = any(indicator in text_lower for indicator in question_indicators)
        
        return has_our_indicator and has_question
    
    def _should_contribute_to_topic(self, text: str) -> bool:
        """Determine if we should contribute to the current topic"""
        # Check participation frequency setting
        import random
        if random.random() > self.config.participation_frequency:
            return False
        
        # Check cooldown period
        if self.last_response_time:
            time_since_last = (datetime.now() - self.last_response_time).total_seconds()
            if time_since_last < self.response_cooldown:
                return False
        
        # Check if topic is relevant to our capabilities
        relevant_topics = ['technology', 'ai', 'automation', 'productivity', 'meeting', 'project']
        text_lower = text.lower()
        
        return any(topic in text_lower for topic in relevant_topics)
    
    async def _handle_direct_question(self, question: str, speaker: str):
        """Handle a direct question addressed to us"""
        try:
            self.logger.info(f"Handling direct question from {speaker}", LogCategory.CONVERSATION)
            
            # Generate response using conversation engine
            response = await self.conversation_engine.generate_response(
                question, 
                context={'speaker': speaker, 'type': 'direct_question'}
            )
            
            if response:
                await self._queue_action(MeetingAction(
                    action_type="speak_response",
                    parameters={
                        "text": response,
                        "in_response_to": speaker,
                        "response_type": "direct_answer"
                    },
                    priority=1  # High priority for direct questions
                ))
                
                self.participation_stats['questions_answered'] += 1
            
        except Exception as e:
            self.logger.error(f"Error handling direct question: {e}", 
                            LogCategory.CONVERSATION, error=e)
    
    async def _consider_contribution(self, text: str, speaker: str):
        """Consider making a contribution to the conversation"""
        try:
            # Generate potential contribution
            contribution = await self.conversation_engine.generate_contribution(
                text,
                context={'speaker': speaker, 'meeting_context': self.meeting_context}
            )
            
            if contribution and len(contribution.strip()) > 0:
                await self._queue_action(MeetingAction(
                    action_type="speak_response",
                    parameters={
                        "text": contribution,
                        "response_type": "contribution"
                    },
                    priority=2  # Lower priority than direct questions
                ))
            
        except Exception as e:
            self.logger.error(f"Error considering contribution: {e}", 
                            LogCategory.CONVERSATION, error=e)
    
    async def _evaluate_participation_opportunity(self):
        """Evaluate if there's an opportunity to participate"""
        try:
            # Check if there's been silence for a while
            if self.meeting_context.get('current_speaker') is None:
                silence_duration = 10.0  # seconds
                if self.last_activity_time:
                    time_since_activity = (datetime.now() - self.last_activity_time).total_seconds()
                    if time_since_activity > silence_duration and self._should_participate():
                        await self._queue_action(MeetingAction(
                            action_type="break_silence",
                            parameters={"message": "Is there anything I can help clarify or contribute to?"},
                            priority=3
                        ))
            
        except Exception as e:
            self.logger.error(f"Error evaluating participation: {e}", 
                            LogCategory.MEETING, error=e)
    
    def _should_participate(self) -> bool:
        """Determine if we should participate based on various factors"""
        # Check if we're in an appropriate state
        if self.current_state != MeetingState.ACTIVE:
            return False
        
        # Check participation frequency
        import random
        return random.random() < self.config.participation_frequency
    
    async def _update_meeting_context(self):
        """Update meeting context with current information"""
        try:
            # Get participant list from Zoom
            participants = await self.zoom_controller.get_participant_list()
            if participants:
                self.meeting_context['participants'] = participants
            
            # Update meeting duration
            if self.meeting_start_time:
                duration = (datetime.now() - self.meeting_start_time).total_seconds() / 60
                self.meeting_context['duration_minutes'] = duration
            
        except Exception as e:
            self.logger.error(f"Error updating meeting context: {e}", 
                            LogCategory.MEETING, error=e)
    
    async def _check_meeting_health(self):
        """Check if the meeting is still healthy and active"""
        try:
            # Check if Zoom is still running
            zoom_active = await self.zoom_controller.is_meeting_active()
            if not zoom_active:
                self.logger.warning("Zoom meeting no longer active", LogCategory.MEETING)
                self.current_state = MeetingState.ERROR
                return
            
            # Check if we've exceeded maximum meeting duration
            if self.meeting_start_time:
                duration_minutes = (datetime.now() - self.meeting_start_time).total_seconds() / 60
                if duration_minutes > self.config.meeting_timeout / 60:
                    self.logger.info("Meeting timeout reached", LogCategory.MEETING)
                    await self.leave_meeting(self.current_meeting)
            
        except Exception as e:
            self.logger.error(f"Error checking meeting health: {e}", 
                            LogCategory.MEETING, error=e)
    
    async def _queue_action(self, action: MeetingAction):
        """Queue an action for processing"""
        await self.action_queue.put(action)
    
    async def _process_actions(self):
        """Process queued meeting actions"""
        self.logger.info("Starting action processor", LogCategory.MEETING)
        
        while True:
            try:
                # Get action from queue
                action = await self.action_queue.get()
                
                if action is None:  # Shutdown signal
                    break
                
                # Process the action
                await self._execute_action(action)
                
                # Mark task as done
                self.action_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error in action processor: {e}", LogCategory.MEETING, error=e)
                await asyncio.sleep(1.0)
    
    async def _execute_action(self, action: MeetingAction):
        """Execute a meeting action"""
        try:
            action_type = action.action_type
            params = action.parameters
            
            if action_type == "speak_response":
                await self._speak_response(params.get("text", ""), params)
            
            elif action_type == "initial_greeting":
                await self._speak_response(params.get("message", ""), params)
            
            elif action_type == "break_silence":
                await self._speak_response(params.get("message", ""), params)
            
            else:
                self.logger.warning(f"Unknown action type: {action_type}", LogCategory.MEETING)
            
        except Exception as e:
            self.logger.error(f"Error executing action {action.action_type}: {e}", 
                            LogCategory.MEETING, error=e)
    
    async def _speak_response(self, text: str, params: Dict[str, Any]):
        """Speak a response in the meeting"""
        try:
            # Add natural delay
            delay = self.config.response_delay_min + (
                self.config.response_delay_max - self.config.response_delay_min
            ) * __import__('random').random()
            
            await asyncio.sleep(delay)
            
            # Unmute microphone
            await self.zoom_controller.unmute_microphone()
            
            # Speak using conversation engine
            speaking_duration = await self.conversation_engine.speak_text(text)
            
            # Update participation stats
            self.participation_stats['responses_given'] += 1
            self.participation_stats['total_speaking_time'] += speaking_duration
            self.participation_stats['last_participation'] = datetime.now()
            self.last_response_time = datetime.now()
            
            # Mute microphone again
            await asyncio.sleep(1.0)  # Brief pause
            await self.zoom_controller.mute_microphone()
            
            self.logger.log_conversation_turn("Darrell", len(text), speaking_duration)
            
        except Exception as e:
            self.logger.error(f"Error speaking response: {e}", LogCategory.CONVERSATION, error=e)
    
    async def leave_meeting(self, meeting_session):
        """Leave the current meeting"""
        try:
            self.logger.log_meeting_event("Leaving meeting", {
                "meeting_id": meeting_session.meeting_id if meeting_session else "unknown"
            })
            
            self.current_state = MeetingState.LEAVING
            
            # Stop conversation monitoring
            await self.conversation_engine.stop_meeting_monitoring()
            
            # Leave Zoom meeting
            await self.zoom_controller.leave_meeting()
            
            # Clean up meeting state
            self.current_meeting = None
            self.meeting_start_time = None
            self.meeting_context = {
                'participants': [],
                'current_speaker': None,
                'meeting_topic': None,
                'agenda_items': [],
                'action_items': []
            }
            
            self.current_state = MeetingState.IDLE
            
            self.logger.log_meeting_event("Meeting left successfully", {
                "participation_stats": self.participation_stats
            })
            
        except Exception as e:
            self.logger.error(f"Error leaving meeting: {e}", LogCategory.MEETING, error=e)
            self.current_state = MeetingState.ERROR
    
    async def get_meeting_status(self) -> Dict[str, Any]:
        """Get current meeting status"""
        return {
            "state": self.current_state.value,
            "current_meeting": self.current_meeting.meeting_id if self.current_meeting else None,
            "meeting_duration": (datetime.now() - self.meeting_start_time).total_seconds() / 60 if self.meeting_start_time else 0,
            "participation_stats": self.participation_stats.copy(),
            "meeting_context": self.meeting_context.copy()
        }
    
    async def cleanup(self):
        """Cleanup meeting coordinator resources"""
        self.logger.info("Cleaning up meeting coordinator...", LogCategory.MEETING)
        
        # Stop action processor
        if self.action_processor_task:
            await self.action_queue.put(None)  # Shutdown signal
            self.action_processor_task.cancel()
            try:
                await self.action_processor_task
            except asyncio.CancelledError:
                pass
        
        # Leave current meeting if active
        if self.current_meeting and self.current_state == MeetingState.ACTIVE:
            await self.leave_meeting(self.current_meeting)
        
        self.is_initialized = False
        self.logger.info("Meeting coordinator cleanup completed", LogCategory.MEETING)
