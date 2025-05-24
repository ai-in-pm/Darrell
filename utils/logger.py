"""
Advanced Logging System for Darrell Agent
Provides comprehensive logging with security, performance monitoring, and audit trails
"""

import logging
import logging.handlers
import os
import json
import time
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum


class LogLevel(Enum):
    """Log level enumeration"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogCategory(Enum):
    """Log category enumeration"""
    GENERAL = "GENERAL"
    VOICE = "VOICE"
    MEETING = "MEETING"
    AUTOMATION = "AUTOMATION"
    SECURITY = "SECURITY"
    PERFORMANCE = "PERFORMANCE"
    CONVERSATION = "CONVERSATION"
    AI_MODEL = "AI_MODEL"


@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: str
    level: str
    category: str
    component: str
    message: str
    session_id: Optional[str] = None
    meeting_id: Optional[str] = None
    user_id: Optional[str] = None
    performance_data: Optional[Dict[str, Any]] = None
    security_context: Optional[Dict[str, Any]] = None
    error_details: Optional[Dict[str, Any]] = None


class DarrellLogger:
    """Advanced logging system for Darrell Agent"""
    
    def __init__(self, component_name: str, log_dir: str = "logs", 
                 log_level: str = "INFO", enable_console: bool = True,
                 enable_file: bool = True, enable_audit: bool = True):
        """
        Initialize Darrell Logger
        
        Args:
            component_name: Name of the component using this logger
            log_dir: Directory for log files
            log_level: Minimum log level to record
            enable_console: Enable console output
            enable_file: Enable file logging
            enable_audit: Enable audit logging
        """
        self.component_name = component_name
        self.log_dir = Path(log_dir)
        self.log_level = getattr(logging, log_level.upper())
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.enable_audit = enable_audit
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Session tracking
        self.session_id = self._generate_session_id()
        self.current_meeting_id = None
        self.current_user_id = None
        
        # Performance tracking
        self.performance_metrics = {}
        self.operation_timers = {}
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Initialize loggers
        self._setup_loggers()
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        return f"darrell_{int(time.time())}_{os.getpid()}"
    
    def _setup_loggers(self):
        """Setup logging infrastructure"""
        # Main logger
        self.logger = logging.getLogger(f"darrell.{self.component_name}")
        self.logger.setLevel(self.log_level)
        self.logger.handlers.clear()
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        if self.enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if self.enable_file:
            log_file = self.log_dir / f"darrell_{self.component_name}.log"
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=10*1024*1024, backupCount=5
            )
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Audit logger
        if self.enable_audit:
            self.audit_logger = logging.getLogger(f"darrell.audit.{self.component_name}")
            self.audit_logger.setLevel(logging.INFO)
            self.audit_logger.handlers.clear()
            
            audit_file = self.log_dir / f"darrell_audit_{self.component_name}.log"
            audit_handler = logging.handlers.RotatingFileHandler(
                audit_file, maxBytes=50*1024*1024, backupCount=10
            )
            audit_formatter = logging.Formatter('%(message)s')
            audit_handler.setFormatter(audit_formatter)
            self.audit_logger.addHandler(audit_handler)
    
    def set_meeting_context(self, meeting_id: str, user_id: str = None):
        """Set current meeting context"""
        with self._lock:
            self.current_meeting_id = meeting_id
            self.current_user_id = user_id
    
    def clear_meeting_context(self):
        """Clear current meeting context"""
        with self._lock:
            self.current_meeting_id = None
            self.current_user_id = None
    
    def _create_log_entry(self, level: str, category: LogCategory, message: str,
                         performance_data: Dict[str, Any] = None,
                         security_context: Dict[str, Any] = None,
                         error_details: Dict[str, Any] = None) -> LogEntry:
        """Create structured log entry"""
        return LogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level=level,
            category=category.value,
            component=self.component_name,
            message=message,
            session_id=self.session_id,
            meeting_id=self.current_meeting_id,
            user_id=self.current_user_id,
            performance_data=performance_data,
            security_context=security_context,
            error_details=error_details
        )
    
    def _log_structured(self, log_entry: LogEntry):
        """Log structured entry"""
        # Standard logging
        log_message = f"[{log_entry.category}] {log_entry.message}"
        if log_entry.performance_data:
            log_message += f" | Performance: {log_entry.performance_data}"
        
        getattr(self.logger, log_entry.level.lower())(log_message)
        
        # Audit logging
        if self.enable_audit:
            audit_entry = json.dumps(asdict(log_entry), default=str)
            self.audit_logger.info(audit_entry)
    
    def debug(self, message: str, category: LogCategory = LogCategory.GENERAL, **kwargs):
        """Log debug message"""
        log_entry = self._create_log_entry("DEBUG", category, message, **kwargs)
        self._log_structured(log_entry)
    
    def info(self, message: str, category: LogCategory = LogCategory.GENERAL, **kwargs):
        """Log info message"""
        log_entry = self._create_log_entry("INFO", category, message, **kwargs)
        self._log_structured(log_entry)
    
    def warning(self, message: str, category: LogCategory = LogCategory.GENERAL, **kwargs):
        """Log warning message"""
        log_entry = self._create_log_entry("WARNING", category, message, **kwargs)
        self._log_structured(log_entry)
    
    def error(self, message: str, category: LogCategory = LogCategory.GENERAL, 
              error: Exception = None, **kwargs):
        """Log error message"""
        error_details = None
        if error:
            error_details = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": str(error.__traceback__) if error.__traceback__ else None
            }
        
        log_entry = self._create_log_entry("ERROR", category, message, 
                                         error_details=error_details, **kwargs)
        self._log_structured(log_entry)
    
    def critical(self, message: str, category: LogCategory = LogCategory.GENERAL, 
                 error: Exception = None, **kwargs):
        """Log critical message"""
        error_details = None
        if error:
            error_details = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": str(error.__traceback__) if error.__traceback__ else None
            }
        
        log_entry = self._create_log_entry("CRITICAL", category, message, 
                                         error_details=error_details, **kwargs)
        self._log_structured(log_entry)
    
    def log_voice_activity(self, activity: str, details: Dict[str, Any] = None):
        """Log voice synthesis activity"""
        message = f"Voice activity: {activity}"
        self.info(message, LogCategory.VOICE, performance_data=details)
    
    def log_meeting_event(self, event: str, details: Dict[str, Any] = None):
        """Log meeting-related event"""
        message = f"Meeting event: {event}"
        self.info(message, LogCategory.MEETING, performance_data=details)
    
    def log_automation_action(self, action: str, success: bool, details: Dict[str, Any] = None):
        """Log automation action"""
        status = "SUCCESS" if success else "FAILED"
        message = f"Automation action: {action} - {status}"
        level = "info" if success else "warning"
        getattr(self, level)(message, LogCategory.AUTOMATION, performance_data=details)
    
    def log_security_event(self, event: str, severity: str = "INFO", 
                          context: Dict[str, Any] = None):
        """Log security-related event"""
        message = f"Security event: {event}"
        level = severity.lower()
        getattr(self, level)(message, LogCategory.SECURITY, security_context=context)
    
    def log_conversation_turn(self, speaker: str, message_length: int, 
                            response_time: float = None):
        """Log conversation turn"""
        details = {
            "speaker": speaker,
            "message_length": message_length,
            "response_time": response_time
        }
        message = f"Conversation turn: {speaker} ({message_length} chars)"
        self.info(message, LogCategory.CONVERSATION, performance_data=details)
    
    def log_ai_model_usage(self, model: str, operation: str, 
                          processing_time: float, tokens_used: int = None):
        """Log AI model usage"""
        details = {
            "model": model,
            "operation": operation,
            "processing_time": processing_time,
            "tokens_used": tokens_used
        }
        message = f"AI model usage: {model} - {operation}"
        self.info(message, LogCategory.AI_MODEL, performance_data=details)
    
    def start_timer(self, operation: str) -> str:
        """Start performance timer"""
        timer_id = f"{operation}_{int(time.time() * 1000)}"
        with self._lock:
            self.operation_timers[timer_id] = time.time()
        return timer_id
    
    def end_timer(self, timer_id: str, operation: str = None) -> float:
        """End performance timer and log result"""
        with self._lock:
            if timer_id not in self.operation_timers:
                self.warning(f"Timer {timer_id} not found")
                return 0.0
            
            start_time = self.operation_timers.pop(timer_id)
            duration = time.time() - start_time
            
            if operation:
                self.info(f"Operation completed: {operation}", 
                         LogCategory.PERFORMANCE, 
                         performance_data={"duration": duration})
            
            return duration
    
    def log_performance_metrics(self, metrics: Dict[str, Any]):
        """Log performance metrics"""
        with self._lock:
            self.performance_metrics.update(metrics)
        
        self.info("Performance metrics updated", LogCategory.PERFORMANCE, 
                 performance_data=metrics)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        with self._lock:
            return self.performance_metrics.copy()
    
    def export_logs(self, start_time: datetime = None, end_time: datetime = None,
                   categories: List[LogCategory] = None) -> List[Dict[str, Any]]:
        """Export logs for analysis"""
        # This would typically read from the audit log file
        # For now, return empty list as placeholder
        return []
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up old log files"""
        cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
        
        for log_file in self.log_dir.glob("*.log*"):
            if log_file.stat().st_mtime < cutoff_time:
                try:
                    log_file.unlink()
                    self.info(f"Deleted old log file: {log_file}")
                except Exception as e:
                    self.error(f"Failed to delete log file {log_file}: {e}")


# Global logger instance
_global_logger = None


def get_logger(component_name: str = "darrell", **kwargs) -> DarrellLogger:
    """Get or create logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = DarrellLogger(component_name, **kwargs)
    return _global_logger


def set_global_logger(logger: DarrellLogger):
    """Set global logger instance"""
    global _global_logger
    _global_logger = logger
