# utils/player_engine.py - Multi-song playback engine with auto-advance functionality
import time
import threading
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from utils.enhanced_song_database import EnhancedSong
from utils.playlist_manager import TherapeuticPlaylist

class PlaybackStatus(Enum):
    """Playback status enumeration"""
    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"
    LOADING = "loading"
    ERROR = "error"

@dataclass
class PlaybackState:
    """Tracks current playback status and position"""
    current_song: Optional[EnhancedSong] = None
    playlist: Optional[TherapeuticPlaylist] = None
    playlist_position: int = 0
    status: PlaybackStatus = PlaybackStatus.STOPPED
    auto_advance: bool = True
    repeat_mode: str = "none"  # none, song, playlist
    shuffle: bool = False
    volume: float = 1.0  # 0.0 to 1.0
    
    # Timing information
    song_start_time: Optional[datetime] = None
    song_position: float = 0.0  # Current position in seconds
    song_duration: float = 0.0  # Total song duration
    
    # Error handling
    error_count: int = 0
    last_error: Optional[str] = None
    consecutive_failures: int = 0
    
    # User interaction tracking
    user_interactions: List[Dict[str, Any]] = field(default_factory=list)
    session_start_time: Optional[datetime] = None
    
    def get_progress_percentage(self) -> float:
        """Get playback progress as percentage (0-100)"""
        if self.song_duration > 0:
            return min(100.0, (self.song_position / self.song_duration) * 100)
        return 0.0
    
    def get_remaining_time(self) -> float:
        """Get remaining time in current song"""
        if self.song_duration > 0:
            return max(0.0, self.song_duration - self.song_position)
        return 0.0
    
    def get_formatted_position(self) -> str:
        """Get formatted current position (MM:SS)"""
        minutes = int(self.song_position // 60)
        seconds = int(self.song_position % 60)
        return f"{minutes}:{seconds:02d}"
    
    def get_formatted_duration(self) -> str:
        """Get formatted total duration (MM:SS)"""
        minutes = int(self.song_duration // 60)
        seconds = int(self.song_duration % 60)
        return f"{minutes}:{seconds:02d}"
    
    def add_user_interaction(self, action: str, details: Dict[str, Any] = None):
        """Record user interaction"""
        interaction = {
            "timestamp": datetime.now(),
            "action": action,
            "song_position": self.song_position,
            "playlist_position": self.playlist_position,
            "details": details or {}
        }
        self.user_interactions.append(interaction)

class PlayerEngine:
    """Handles multi-song playback with auto-advancement and error recovery"""
    
    def __init__(self, 
                 on_song_change: Optional[Callable] = None,
                 on_playlist_complete: Optional[Callable] = None,
                 on_error: Optional[Callable] = None):
        """
        Initialize player engine
        
        Args:
            on_song_change: Callback when song changes
            on_playlist_complete: Callback when playlist completes
            on_error: Callback when error occurs
        """
        self.state = PlaybackState()
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_monitoring = threading.Event()
        
        # Callbacks
        self.on_song_change = on_song_change
        self.on_playlist_complete = on_playlist_complete
        self.on_error = on_error
        
        # Configuration
        self.auto_advance_delay = 2.0  # Seconds to wait before auto-advance
        self.max_consecutive_failures = 3
        self.song_load_timeout = 10.0  # Seconds
        
        logging.info("PlayerEngine initialized")
    
    def load_playlist(self, playlist: TherapeuticPlaylist, start_position: int = 0) -> bool:
        """Load a playlist for playback"""
        
        if not playlist or not playlist.songs:
            logging.error("Cannot load empty playlist")
            return False
        
        if start_position < 0 or start_position >= len(playlist.songs):
            logging.error(f"Invalid start position: {start_position}")
            return False
        
        # Stop current playback
        self.stop()
        
        # Load new playlist
        self.state.playlist = playlist
        self.state.playlist_position = start_position
        self.state.current_song = playlist.songs[start_position]
        self.state.session_start_time = datetime.now()
        self.state.error_count = 0
        self.state.consecutive_failures = 0
        self.state.user_interactions = []
        
        logging.info(f"Loaded playlist '{playlist.name}' with {len(playlist.songs)} songs")
        return True
    
    def play_song(self, song: Optional[EnhancedSong] = None, auto_advance: bool = True) -> bool:
        """
        Play a specific song or current song
        
        Args:
            song: Song to play (if None, plays current song)
            auto_advance: Whether to auto-advance to next song
            
        Returns:
            True if playback started successfully
        """
        
        # Use provided song or current song
        if song:
            self.state.current_song = song
        elif not self.state.current_song:
            logging.error("No song to play")
            return False
        
        current_song = self.state.current_song
        self.state.auto_advance = auto_advance
        self.state.status = PlaybackStatus.LOADING
        self.state.song_start_time = datetime.now()
        self.state.song_position = 0.0
        self.state.song_duration = float(current_song.duration)
        
        try:
            # Simulate song loading (in real implementation, this would load the audio)
            logging.info(f"Playing: {current_song.title} by {current_song.artist}")
            
            # Update state
            self.state.status = PlaybackStatus.PLAYING
            self.state.last_error = None
            self.state.consecutive_failures = 0
            
            # Record user interaction
            self.state.add_user_interaction("play", {
                "song_title": current_song.title,
                "song_artist": current_song.artist,
                "auto_advance": auto_advance
            })
            
            # Start monitoring if auto-advance is enabled
            if auto_advance and not self.is_monitoring:
                self._start_monitoring()
            
            # Notify callback
            if self.on_song_change:
                self.on_song_change(current_song, self.state)
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to play song '{current_song.title}': {e}")
            return self._handle_playback_error(e, current_song)
    
    def pause(self) -> bool:
        """Pause current playback"""
        
        if self.state.status != PlaybackStatus.PLAYING:
            return False
        
        self.state.status = PlaybackStatus.PAUSED
        self.state.add_user_interaction("pause")
        
        logging.info("Playback paused")
        return True
    
    def resume(self) -> bool:
        """Resume paused playback"""
        
        if self.state.status != PlaybackStatus.PAUSED:
            return False
        
        self.state.status = PlaybackStatus.PLAYING
        self.state.add_user_interaction("resume")
        
        # Restart monitoring if needed
        if self.state.auto_advance and not self.is_monitoring:
            self._start_monitoring()
        
        logging.info("Playback resumed")
        return True
    
    def stop(self) -> bool:
        """Stop playback"""
        
        self.state.status = PlaybackStatus.STOPPED
        self.state.song_position = 0.0
        self.state.add_user_interaction("stop")
        
        # Stop monitoring
        self._stop_monitoring()
        
        logging.info("Playback stopped")
        return True
    
    def skip_to_next(self) -> bool:
        """Skip to next song in playlist"""
        
        if not self.state.playlist or not self.state.playlist.songs:
            return False
        
        self.state.add_user_interaction("skip_next")
        
        # Calculate next position
        next_position = self._get_next_song_position()
        
        if next_position is None:
            # End of playlist
            return self._handle_playlist_complete()
        
        # Move to next song
        self.state.playlist_position = next_position
        self.state.current_song = self.state.playlist.songs[next_position]
        
        # Continue playing if we were playing
        if self.state.status == PlaybackStatus.PLAYING:
            return self.play_song(auto_advance=self.state.auto_advance)
        
        return True
    
    def skip_to_previous(self) -> bool:
        """Skip to previous song in playlist"""
        
        if not self.state.playlist or not self.state.playlist.songs:
            return False
        
        self.state.add_user_interaction("skip_previous")
        
        # If we're more than 3 seconds into the song, restart current song
        if self.state.song_position > 3.0:
            self.state.song_position = 0.0
            if self.state.status == PlaybackStatus.PLAYING:
                return self.play_song(auto_advance=self.state.auto_advance)
            return True
        
        # Otherwise, go to previous song
        prev_position = self._get_previous_song_position()
        
        if prev_position is None:
            # At beginning of playlist
            self.state.playlist_position = 0
            self.state.current_song = self.state.playlist.songs[0]
        else:
            self.state.playlist_position = prev_position
            self.state.current_song = self.state.playlist.songs[prev_position]
        
        # Continue playing if we were playing
        if self.state.status == PlaybackStatus.PLAYING:
            return self.play_song(auto_advance=self.state.auto_advance)
        
        return True
    
    def seek_to_position(self, position: float) -> bool:
        """Seek to specific position in current song"""
        
        if not self.state.current_song:
            return False
        
        # Clamp position to valid range
        position = max(0.0, min(position, self.state.song_duration))
        self.state.song_position = position
        
        self.state.add_user_interaction("seek", {"position": position})
        
        logging.info(f"Seeked to position {position:.1f}s")
        return True
    
    def set_volume(self, volume: float) -> bool:
        """Set playback volume (0.0 to 1.0)"""
        
        volume = max(0.0, min(1.0, volume))
        self.state.volume = volume
        
        self.state.add_user_interaction("volume_change", {"volume": volume})
        
        logging.info(f"Volume set to {volume:.1%}")
        return True
    
    def set_repeat_mode(self, mode: str) -> bool:
        """Set repeat mode: none, song, playlist"""
        
        if mode not in ["none", "song", "playlist"]:
            return False
        
        self.state.repeat_mode = mode
        self.state.add_user_interaction("repeat_mode_change", {"mode": mode})
        
        logging.info(f"Repeat mode set to: {mode}")
        return True
    
    def set_shuffle(self, enabled: bool) -> bool:
        """Enable or disable shuffle mode"""
        
        self.state.shuffle = enabled
        self.state.add_user_interaction("shuffle_change", {"enabled": enabled})
        
        logging.info(f"Shuffle {'enabled' if enabled else 'disabled'}")
        return True
    
    def get_playback_state(self) -> PlaybackState:
        """Get current playback state"""
        return self.state
    
    def _start_monitoring(self):
        """Start playback monitoring thread"""
        
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.stop_monitoring.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_playback, daemon=True)
        self.monitor_thread.start()
        
        logging.debug("Started playback monitoring")
    
    def _stop_monitoring(self):
        """Stop playback monitoring thread"""
        
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        self.stop_monitoring.set()
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
        
        logging.debug("Stopped playback monitoring")
    
    def _monitor_playback(self):
        """Monitor playback progress and handle auto-advance"""
        
        while self.is_monitoring and not self.stop_monitoring.is_set():
            try:
                if self.state.status == PlaybackStatus.PLAYING:
                    # Update song position
                    if self.state.song_start_time:
                        elapsed = (datetime.now() - self.state.song_start_time).total_seconds()
                        self.state.song_position = elapsed
                    
                    # Check if song is complete
                    if (self.state.song_position >= self.state.song_duration and 
                        self.state.auto_advance):
                        
                        logging.info(f"Song '{self.state.current_song.title}' completed")
                        
                        # Handle repeat mode
                        if self.state.repeat_mode == "song":
                            # Repeat current song
                            self.play_song(auto_advance=True)
                        else:
                            # Auto-advance to next song
                            time.sleep(self.auto_advance_delay)
                            if not self.skip_to_next():
                                # End of playlist
                                break
                
                # Sleep before next check
                time.sleep(0.5)
                
            except Exception as e:
                logging.error(f"Error in playback monitoring: {e}")
                time.sleep(1.0)
        
        logging.debug("Playback monitoring stopped")
    
    def _get_next_song_position(self) -> Optional[int]:
        """Get next song position considering shuffle and repeat modes"""
        
        if not self.state.playlist or not self.state.playlist.songs:
            return None
        
        playlist_length = len(self.state.playlist.songs)
        current_pos = self.state.playlist_position
        
        if self.state.shuffle:
            # Random next song (excluding current)
            import random
            available_positions = [i for i in range(playlist_length) if i != current_pos]
            if available_positions:
                return random.choice(available_positions)
            return None
        else:
            # Sequential next song
            next_pos = current_pos + 1
            
            if next_pos < playlist_length:
                return next_pos
            elif self.state.repeat_mode == "playlist":
                return 0  # Loop back to beginning
            else:
                return None  # End of playlist
    
    def _get_previous_song_position(self) -> Optional[int]:
        """Get previous song position"""
        
        if not self.state.playlist or not self.state.playlist.songs:
            return None
        
        current_pos = self.state.playlist_position
        
        if current_pos > 0:
            return current_pos - 1
        elif self.state.repeat_mode == "playlist":
            return len(self.state.playlist.songs) - 1  # Loop to end
        else:
            return None  # At beginning
    
    def _handle_playback_error(self, error: Exception, song: EnhancedSong) -> bool:
        """Handle playback errors with recovery logic"""
        
        self.state.error_count += 1
        self.state.consecutive_failures += 1
        self.state.last_error = str(error)
        self.state.status = PlaybackStatus.ERROR
        
        logging.error(f"Playback error for '{song.title}': {error}")
        
        # Record error interaction
        self.state.add_user_interaction("error", {
            "error_message": str(error),
            "song_title": song.title,
            "consecutive_failures": self.state.consecutive_failures
        })
        
        # Notify error callback
        if self.on_error:
            self.on_error(error, song, self.state)
        
        # Auto-recovery logic
        if (self.state.consecutive_failures < self.max_consecutive_failures and 
            self.state.auto_advance):
            
            logging.info(f"Attempting auto-recovery (attempt {self.state.consecutive_failures})")
            
            # Try to skip to next song
            time.sleep(1.0)  # Brief delay
            return self.skip_to_next()
        
        else:
            logging.error("Max consecutive failures reached, stopping playback")
            self.stop()
            return False
    
    def _handle_playlist_complete(self) -> bool:
        """Handle playlist completion"""
        
        logging.info("Playlist completed")
        
        # Record completion
        self.state.add_user_interaction("playlist_complete")
        
        # Handle repeat mode
        if self.state.repeat_mode == "playlist":
            # Restart playlist
            self.state.playlist_position = 0
            self.state.current_song = self.state.playlist.songs[0]
            return self.play_song(auto_advance=self.state.auto_advance)
        else:
            # Stop playback
            self.stop()
            
            # Notify callback
            if self.on_playlist_complete:
                self.on_playlist_complete(self.state.playlist, self.state)
            
            return True
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get session statistics and analytics"""
        
        if not self.state.session_start_time:
            return {}
        
        session_duration = (datetime.now() - self.state.session_start_time).total_seconds()
        
        # Count interaction types
        interaction_counts = {}
        for interaction in self.state.user_interactions:
            action = interaction["action"]
            interaction_counts[action] = interaction_counts.get(action, 0) + 1
        
        # Calculate completion rate
        completion_rate = 0.0
        if self.state.playlist and self.state.playlist.songs:
            completed_songs = self.state.playlist_position
            if self.state.status == PlaybackStatus.STOPPED:
                completed_songs += 1  # Include current song if stopped naturally
            completion_rate = (completed_songs / len(self.state.playlist.songs)) * 100
        
        return {
            "session_duration": session_duration,
            "songs_played": self.state.playlist_position + (1 if self.state.current_song else 0),
            "completion_rate": completion_rate,
            "error_count": self.state.error_count,
            "user_interactions": len(self.state.user_interactions),
            "interaction_breakdown": interaction_counts,
            "average_song_position": sum(i["song_position"] for i in self.state.user_interactions) / max(1, len(self.state.user_interactions))
        }
    
    def cleanup(self):
        """Clean up resources"""
        
        self.stop()
        self._stop_monitoring()
        logging.info("PlayerEngine cleaned up")