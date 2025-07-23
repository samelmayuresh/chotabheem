# utils/playback_error_handler.py - Robust error handling for music playback
import logging
import time
import os
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

from utils.enhanced_song_database import EnhancedSong

class ErrorType(Enum):
    """Types of playback errors"""
    SONG_LOAD_FAILED = "song_load_failed"
    NETWORK_ERROR = "network_error"
    AUDIO_DECODE_ERROR = "audio_decode_error"
    PERMISSION_ERROR = "permission_error"
    SERVICE_UNAVAILABLE = "service_unavailable"
    TIMEOUT_ERROR = "timeout_error"
    UNKNOWN_ERROR = "unknown_error"

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class PlaybackError:
    """Represents a playback error with metadata"""
    error_type: ErrorType
    severity: ErrorSeverity
    message: str
    song: Optional[EnhancedSong]
    timestamp: datetime
    retry_count: int = 0
    context: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_method: Optional[str] = None

@dataclass
class ErrorStatistics:
    """Error statistics for monitoring"""
    total_errors: int = 0
    errors_by_type: Dict[str, int] = field(default_factory=dict)
    errors_by_severity: Dict[str, int] = field(default_factory=dict)
    resolution_success_rate: float = 0.0
    average_resolution_time: float = 0.0
    most_problematic_songs: List[str] = field(default_factory=list)

class PlaybackErrorHandler:
    """Handles playback errors with recovery strategies"""
    
    def __init__(self, 
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 max_consecutive_failures: int = 5,
                 error_log_path: str = "playback_errors.json"):
        """
        Initialize error handler
        
        Args:
            max_retries: Maximum retry attempts per song
            retry_delay: Delay between retries in seconds
            max_consecutive_failures: Max consecutive failures before stopping
            error_log_path: Path to error log file
        """
        
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_consecutive_failures = max_consecutive_failures
        self.error_log_path = error_log_path
        
        # Error tracking
        self.error_history: List[PlaybackError] = []
        self.consecutive_failures = 0
        self.failed_songs: Dict[str, int] = {}  # Song URL -> failure count
        
        # Recovery strategies
        self.recovery_strategies = {
            ErrorType.SONG_LOAD_FAILED: self._handle_song_load_error,
            ErrorType.NETWORK_ERROR: self._handle_network_error,
            ErrorType.AUDIO_DECODE_ERROR: self._handle_audio_decode_error,
            ErrorType.PERMISSION_ERROR: self._handle_permission_error,
            ErrorType.SERVICE_UNAVAILABLE: self._handle_service_unavailable,
            ErrorType.TIMEOUT_ERROR: self._handle_timeout_error,
            ErrorType.UNKNOWN_ERROR: self._handle_unknown_error
        }
        
        # Load existing error history
        self.load_error_history()
        
        logging.info("PlaybackErrorHandler initialized")
    
    def handle_error(self, 
                    error: Exception, 
                    song: Optional[EnhancedSong] = None,
                    context: Dict[str, Any] = None) -> bool:
        """
        Handle a playback error with appropriate recovery strategy
        
        Args:
            error: The exception that occurred
            song: Song that caused the error (if applicable)
            context: Additional context about the error
            
        Returns:
            True if error was resolved, False otherwise
        """
        
        # Classify the error
        error_type, severity = self._classify_error(error)
        
        # Create error record
        playback_error = PlaybackError(
            error_type=error_type,
            severity=severity,
            message=str(error),
            song=song,
            timestamp=datetime.now(),
            context=context or {}
        )
        
        # Add to history
        self.error_history.append(playback_error)
        
        # Update failure tracking
        if song:
            song_key = f"{song.title}_{song.artist}"
            self.failed_songs[song_key] = self.failed_songs.get(song_key, 0) + 1
        
        # Log the error
        logging.error(f"Playback error: {error_type.value} - {error}")
        
        # Try to resolve the error
        success = self._attempt_recovery(playback_error)
        
        if success:
            playback_error.resolved = True
            self.consecutive_failures = 0
            logging.info(f"Successfully recovered from {error_type.value}")
        else:
            self.consecutive_failures += 1
            logging.warning(f"Failed to recover from {error_type.value}. Consecutive failures: {self.consecutive_failures}")
        
        # Save error history
        self.save_error_history()
        
        # Check if we should stop playback due to too many failures
        if self.consecutive_failures >= self.max_consecutive_failures:
            logging.critical(f"Too many consecutive failures ({self.consecutive_failures}). Stopping playback.")
            return False
        
        return success
    
    def _classify_error(self, error: Exception) -> tuple[ErrorType, ErrorSeverity]:
        """Classify error type and severity"""
        
        error_str = str(error).lower()
        error_type_name = type(error).__name__.lower()
        
        # Network-related errors
        if any(keyword in error_str for keyword in ['network', 'connection', 'timeout', 'dns']):
            return ErrorType.NETWORK_ERROR, ErrorSeverity.MEDIUM
        
        # Permission errors
        if any(keyword in error_str for keyword in ['permission', 'forbidden', '403', 'unauthorized', '401']):
            return ErrorType.PERMISSION_ERROR, ErrorSeverity.HIGH
        
        # Service unavailable
        if any(keyword in error_str for keyword in ['service unavailable', '503', 'server error', '500']):
            return ErrorType.SERVICE_UNAVAILABLE, ErrorSeverity.HIGH
        
        # Timeout errors
        if any(keyword in error_str for keyword in ['timeout', 'timed out']):
            return ErrorType.TIMEOUT_ERROR, ErrorSeverity.MEDIUM
        
        # Audio decode errors
        if any(keyword in error_str for keyword in ['decode', 'codec', 'format', 'audio']):
            return ErrorType.AUDIO_DECODE_ERROR, ErrorSeverity.MEDIUM
        
        # Song load failures
        if any(keyword in error_str for keyword in ['not found', '404', 'file not found', 'load']):
            return ErrorType.SONG_LOAD_FAILED, ErrorSeverity.MEDIUM
        
        # Default to unknown error
        return ErrorType.UNKNOWN_ERROR, ErrorSeverity.LOW
    
    def _attempt_recovery(self, error: PlaybackError) -> bool:
        """Attempt to recover from the error"""
        
        if error.retry_count >= self.max_retries:
            logging.warning(f"Max retries exceeded for {error.error_type.value}")
            return False
        
        # Get recovery strategy
        recovery_func = self.recovery_strategies.get(error.error_type, self._handle_unknown_error)
        
        # Attempt recovery
        try:
            success = recovery_func(error)
            if success:
                error.resolution_method = recovery_func.__name__
            return success
        except Exception as recovery_error:
            logging.error(f"Recovery strategy failed: {recovery_error}")
            return False
    
    def _handle_song_load_error(self, error: PlaybackError) -> bool:
        """Handle song loading failures"""
        
        logging.info(f"Attempting to recover from song load error: {error.message}")
        
        # Strategy 1: Retry after delay
        if error.retry_count < 2:
            time.sleep(self.retry_delay * (error.retry_count + 1))
            error.retry_count += 1
            return True  # Signal to retry
        
        # Strategy 2: Skip to next song
        logging.info("Skipping problematic song")
        return False  # Signal to skip song
    
    def _handle_network_error(self, error: PlaybackError) -> bool:
        """Handle network-related errors"""
        
        logging.info(f"Attempting to recover from network error: {error.message}")
        
        # Strategy 1: Exponential backoff retry
        if error.retry_count < self.max_retries:
            delay = self.retry_delay * (2 ** error.retry_count)
            time.sleep(min(delay, 30))  # Cap at 30 seconds
            error.retry_count += 1
            return True
        
        # Strategy 2: Switch to fallback source (if available)
        if error.song and hasattr(error.song, 'fallback_url') and error.song.fallback_url:
            logging.info("Switching to fallback URL")
            error.song.url = error.song.fallback_url
            return True
        
        return False
    
    def _handle_audio_decode_error(self, error: PlaybackError) -> bool:
        """Handle audio decoding errors"""
        
        logging.info(f"Attempting to recover from audio decode error: {error.message}")
        
        # Strategy 1: Try alternative format (if available)
        if error.song:
            # This would require song database to have alternative formats
            logging.info("Audio decode error - skipping song")
            return False
        
        return False
    
    def _handle_permission_error(self, error: PlaybackError) -> bool:
        """Handle permission/authorization errors"""
        
        logging.info(f"Handling permission error: {error.message}")
        
        # Strategy 1: Skip song (can't resolve permission issues automatically)
        logging.warning("Permission denied - skipping song")
        return False
    
    def _handle_service_unavailable(self, error: PlaybackError) -> bool:
        """Handle service unavailability"""
        
        logging.info(f"Handling service unavailable error: {error.message}")
        
        # Strategy 1: Retry with exponential backoff
        if error.retry_count < 2:
            delay = 5 * (2 ** error.retry_count)
            time.sleep(delay)
            error.retry_count += 1
            return True
        
        return False
    
    def _handle_timeout_error(self, error: PlaybackError) -> bool:
        """Handle timeout errors"""
        
        logging.info(f"Handling timeout error: {error.message}")
        
        # Strategy 1: Retry with longer timeout
        if error.retry_count < 2:
            time.sleep(self.retry_delay * 2)
            error.retry_count += 1
            return True
        
        return False
    
    def _handle_unknown_error(self, error: PlaybackError) -> bool:
        """Handle unknown errors"""
        
        logging.info(f"Handling unknown error: {error.message}")
        
        # Strategy 1: Simple retry
        if error.retry_count < 1:
            time.sleep(self.retry_delay)
            error.retry_count += 1
            return True
        
        return False
    
    def get_error_statistics(self) -> ErrorStatistics:
        """Get comprehensive error statistics"""
        
        if not self.error_history:
            return ErrorStatistics()
        
        total_errors = len(self.error_history)
        
        # Count by type
        errors_by_type = {}
        for error in self.error_history:
            error_type = error.error_type.value
            errors_by_type[error_type] = errors_by_type.get(error_type, 0) + 1
        
        # Count by severity
        errors_by_severity = {}
        for error in self.error_history:
            severity = error.severity.value
            errors_by_severity[severity] = errors_by_severity.get(severity, 0) + 1
        
        # Calculate resolution success rate
        resolved_errors = sum(1 for error in self.error_history if error.resolved)
        resolution_success_rate = (resolved_errors / total_errors) * 100 if total_errors > 0 else 0
        
        # Most problematic songs
        most_problematic = sorted(self.failed_songs.items(), key=lambda x: x[1], reverse=True)[:5]
        most_problematic_songs = [song for song, count in most_problematic]
        
        return ErrorStatistics(
            total_errors=total_errors,
            errors_by_type=errors_by_type,
            errors_by_severity=errors_by_severity,
            resolution_success_rate=resolution_success_rate,
            most_problematic_songs=most_problematic_songs
        )
    
    def get_recent_errors(self, hours: int = 24) -> List[PlaybackError]:
        """Get errors from the last N hours"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [error for error in self.error_history if error.timestamp > cutoff_time]
    
    def clear_error_history(self, older_than_days: int = 30):
        """Clear old error history"""
        
        cutoff_time = datetime.now() - timedelta(days=older_than_days)
        self.error_history = [error for error in self.error_history if error.timestamp > cutoff_time]
        self.save_error_history()
        
        logging.info(f"Cleared error history older than {older_than_days} days")
    
    def is_song_problematic(self, song: EnhancedSong, threshold: int = 3) -> bool:
        """Check if a song has failed too many times"""
        
        song_key = f"{song.title}_{song.artist}"
        return self.failed_songs.get(song_key, 0) >= threshold
    
    def get_problematic_songs(self, threshold: int = 2) -> List[tuple[str, int]]:
        """Get list of songs that have failed multiple times"""
        
        return [(song, count) for song, count in self.failed_songs.items() if count >= threshold]
    
    def reset_consecutive_failures(self):
        """Reset consecutive failure counter"""
        self.consecutive_failures = 0
    
    def save_error_history(self):
        """Save error history to file"""
        
        try:
            # Convert to serializable format
            history_data = []
            for error in self.error_history[-1000:]:  # Keep last 1000 errors
                error_dict = {
                    "error_type": error.error_type.value,
                    "severity": error.severity.value,
                    "message": error.message,
                    "song_title": error.song.title if error.song else None,
                    "song_artist": error.song.artist if error.song else None,
                    "timestamp": error.timestamp.isoformat(),
                    "retry_count": error.retry_count,
                    "context": error.context,
                    "resolved": error.resolved,
                    "resolution_method": error.resolution_method
                }
                history_data.append(error_dict)
            
            data = {
                "error_history": history_data,
                "failed_songs": self.failed_songs,
                "consecutive_failures": self.consecutive_failures
            }
            
            with open(self.error_log_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logging.error(f"Failed to save error history: {e}")
    
    def load_error_history(self):
        """Load error history from file"""
        
        try:
            if not os.path.exists(self.error_log_path):
                return
            
            with open(self.error_log_path, 'r') as f:
                data = json.load(f)
            
            # Load failed songs tracking
            self.failed_songs = data.get("failed_songs", {})
            self.consecutive_failures = data.get("consecutive_failures", 0)
            
            # Load error history (recent errors only)
            history_data = data.get("error_history", [])
            cutoff_time = datetime.now() - timedelta(days=7)  # Keep last 7 days
            
            for error_dict in history_data:
                try:
                    timestamp = datetime.fromisoformat(error_dict["timestamp"])
                    if timestamp > cutoff_time:
                        # Reconstruct song object if available
                        song = None
                        if error_dict.get("song_title") and error_dict.get("song_artist"):
                            song = EnhancedSong(
                                title=error_dict["song_title"],
                                artist=error_dict["song_artist"],
                                url="",  # URL not stored for privacy
                                duration=0,
                                genre="",
                                mood=""
                            )
                        
                        error = PlaybackError(
                            error_type=ErrorType(error_dict["error_type"]),
                            severity=ErrorSeverity(error_dict["severity"]),
                            message=error_dict["message"],
                            song=song,
                            timestamp=timestamp,
                            retry_count=error_dict.get("retry_count", 0),
                            context=error_dict.get("context", {}),
                            resolved=error_dict.get("resolved", False),
                            resolution_method=error_dict.get("resolution_method")
                        )
                        
                        self.error_history.append(error)
                        
                except Exception as e:
                    logging.warning(f"Failed to load error record: {e}")
                    continue
            
            logging.info(f"Loaded {len(self.error_history)} error records")
            
        except Exception as e:
            logging.error(f"Failed to load error history: {e}")
    
    def cleanup(self):
        """Cleanup error handler"""
        self.clear_error_history(older_than_days=30)
        self.save_error_history()
        logging.info("PlaybackErrorHandler cleaned up")