# utils/enhanced_music_controller.py - Enhanced music controller orchestrating all components
import logging
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import random

from utils.enhanced_song_database import EnhancedSong, EnhancedSongDatabase
from utils.playlist_manager import TherapeuticPlaylist, PlaylistManager
from utils.player_engine import PlayerEngine, PlaybackState, PlaybackStatus

@dataclass
class MusicSession:
    """Represents a complete music therapy session"""
    session_id: str
    emotion_focus: str
    secondary_emotions: List[str]
    therapy_mode: str  # "targeted", "full_session", "custom"
    playlist: TherapeuticPlaylist
    start_time: datetime
    user_preferences: Dict[str, Any]
    session_notes: str = ""

class EnhancedMusicController:
    """Enhanced music controller orchestrating playlist management, playback, and emotion integration"""
    
    def __init__(self, 
                 storage_path: str = "music_sessions.json",
                 on_session_start: Optional[Callable] = None,
                 on_session_complete: Optional[Callable] = None,
                 on_song_change: Optional[Callable] = None,
                 on_error: Optional[Callable] = None):
        """
        Initialize enhanced music controller
        
        Args:
            storage_path: Path for session persistence
            on_session_start: Callback when therapy session starts
            on_session_complete: Callback when session completes
            on_song_change: Callback when song changes
            on_error: Callback when error occurs
        """
        
        # Initialize components
        self.song_database = EnhancedSongDatabase()
        self.playlist_manager = PlaylistManager(storage_path)
        
        # Initialize player engine with callbacks
        self.player_engine = PlayerEngine(
            on_song_change=self._handle_song_change,
            on_playlist_complete=self._handle_playlist_complete,
            on_error=self._handle_error
        )
        
        # Session management
        self.current_session: Optional[MusicSession] = None
        self.session_history: List[MusicSession] = []
        
        # User callbacks
        self.on_session_start = on_session_start
        self.on_session_complete = on_session_complete
        self.on_song_change = on_song_change
        self.on_error = on_error
        
        # Default preferences
        self.default_preferences = {
            "vocal_ratio": 0.7,  # 70% vocal tracks
            "auto_advance": True,
            "volume": 1.0,
            "language_preference": "english",
            "content_rating": "clean",
            "session_length": "medium",  # short, medium, long
            "energy_preference": "balanced"  # low, balanced, high
        }
        
        logging.info("EnhancedMusicController initialized")
    
    def create_continuous_session(self, 
                                emotion: str, 
                                mode: str = "targeted", 
                                song_count: int = 5,
                                secondary_emotions: List[str] = None,
                                user_preferences: Dict[str, Any] = None) -> MusicSession:
        """
        Create a continuous multi-song therapy session
        
        Args:
            emotion: Primary emotion for therapy
            mode: Therapy mode ("targeted", "full_session", "custom")
            song_count: Number of songs in session
            secondary_emotions: Additional emotions to address
            user_preferences: User-specific preferences
            
        Returns:
            MusicSession object
        """
        
        # Merge user preferences with defaults
        preferences = self.default_preferences.copy()
        if user_preferences:
            preferences.update(user_preferences)
        
        secondary_emotions = secondary_emotions or []
        
        # Generate session ID
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create playlist based on mode
        if mode == "targeted":
            playlist = self._create_targeted_playlist(emotion, song_count, preferences)
        elif mode == "full_session":
            playlist = self._create_full_session_playlist(emotion, secondary_emotions, preferences)
        elif mode == "custom":
            playlist = self._create_custom_playlist(emotion, secondary_emotions, song_count, preferences)
        else:
            raise ValueError(f"Unknown therapy mode: {mode}")
        
        # Create session
        session = MusicSession(
            session_id=session_id,
            emotion_focus=emotion,
            secondary_emotions=secondary_emotions,
            therapy_mode=mode,
            playlist=playlist,
            start_time=datetime.now(),
            user_preferences=preferences
        )
        
        # Set as current session
        self.current_session = session
        
        # Load playlist into player
        self.player_engine.load_playlist(playlist)
        
        # Apply user preferences to player
        self.player_engine.set_volume(preferences.get("volume", 1.0))
        
        # Notify callback
        if self.on_session_start:
            self.on_session_start(session)
        
        logging.info(f"Created {mode} session for {emotion} with {len(playlist.songs)} songs")
        
        return session
    
    def get_enhanced_english_playlist(self, 
                                    emotion: str, 
                                    count: int = 5,
                                    vocal_ratio: float = 0.8) -> List[EnhancedSong]:
        """
        Get enhanced English playlist with high vocal content
        
        Args:
            emotion: Target emotion
            count: Number of songs
            vocal_ratio: Ratio of vocal to instrumental tracks
            
        Returns:
            List of enhanced songs with vocal focus
        """
        
        # Get vocal-focused songs
        vocal_songs = self.song_database.get_vocal_songs_for_emotion(emotion, count * 2)
        
        # Filter for English songs with vocals
        english_vocal_songs = [
            song for song in vocal_songs 
            if song.language == "english" and song.has_vocals
        ]
        
        # Get some instrumental songs for balance
        all_songs = self.song_database.get_songs_for_emotion(emotion, count * 2)
        instrumental_songs = [
            song for song in all_songs 
            if not song.has_vocals and song.language == "english"
        ]
        
        # Calculate counts
        vocal_count = int(count * vocal_ratio)
        instrumental_count = count - vocal_count
        
        # Select songs
        selected_vocal = random.sample(
            english_vocal_songs, 
            min(vocal_count, len(english_vocal_songs))
        )
        selected_instrumental = random.sample(
            instrumental_songs, 
            min(instrumental_count, len(instrumental_songs))
        )
        
        # Combine and shuffle
        playlist = selected_vocal + selected_instrumental
        random.shuffle(playlist)
        
        return playlist[:count]
    
    def start_playback(self, auto_advance: bool = True) -> bool:
        """Start playback of current session"""
        
        if not self.current_session:
            logging.error("No active session to start playback")
            return False
        
        success = self.player_engine.play_song(auto_advance=auto_advance)
        
        if success:
            logging.info(f"Started playback for session {self.current_session.session_id}")
        
        return success
    
    def handle_playback_control(self, action: str, **kwargs) -> bool:
        """
        Handle playback control actions
        
        Args:
            action: Control action (play, pause, resume, stop, skip_next, skip_previous, seek, volume)
            **kwargs: Additional parameters for specific actions
            
        Returns:
            True if action was successful
        """
        
        if action == "play":
            return self.start_playback(kwargs.get("auto_advance", True))
        
        elif action == "pause":
            return self.player_engine.pause()
        
        elif action == "resume":
            return self.player_engine.resume()
        
        elif action == "stop":
            return self.player_engine.stop()
        
        elif action == "skip_next":
            return self.player_engine.skip_to_next()
        
        elif action == "skip_previous":
            return self.player_engine.skip_to_previous()
        
        elif action == "seek":
            position = kwargs.get("position", 0.0)
            return self.player_engine.seek_to_position(position)
        
        elif action == "volume":
            volume = kwargs.get("volume", 1.0)
            return self.player_engine.set_volume(volume)
        
        elif action == "repeat":
            mode = kwargs.get("mode", "none")
            return self.player_engine.set_repeat_mode(mode)
        
        elif action == "shuffle":
            enabled = kwargs.get("enabled", False)
            return self.player_engine.set_shuffle(enabled)
        
        else:
            logging.error(f"Unknown playback action: {action}")
            return False
    
    def get_playback_state(self) -> PlaybackState:
        """Get current playback state"""
        return self.player_engine.get_playback_state()
    
    def get_session_info(self) -> Optional[Dict[str, Any]]:
        """Get current session information"""
        
        if not self.current_session:
            return None
        
        playback_state = self.player_engine.get_playback_state()
        session_stats = self.player_engine.get_session_statistics()
        
        return {
            "session_id": self.current_session.session_id,
            "emotion_focus": self.current_session.emotion_focus,
            "secondary_emotions": self.current_session.secondary_emotions,
            "therapy_mode": self.current_session.therapy_mode,
            "playlist_name": self.current_session.playlist.name,
            "total_songs": len(self.current_session.playlist.songs),
            "current_position": playback_state.playlist_position,
            "current_song": playback_state.current_song.title if playback_state.current_song else None,
            "playback_status": playback_state.status.value,
            "session_duration": session_stats.get("session_duration", 0),
            "completion_rate": session_stats.get("completion_rate", 0),
            "user_interactions": session_stats.get("user_interactions", 0)
        }
    
    def save_current_session(self) -> bool:
        """Save current session to playlist manager"""
        
        if not self.current_session:
            return False
        
        # Add session notes if available
        session_stats = self.player_engine.get_session_statistics()
        self.current_session.session_notes = f"Completed {session_stats.get('completion_rate', 0):.1f}% of playlist"
        
        # Save playlist
        success = self.playlist_manager.save_playlist(self.current_session.playlist)
        
        if success:
            # Add to session history
            self.session_history.append(self.current_session)
            logging.info(f"Saved session {self.current_session.session_id}")
        
        return success
    
    def load_saved_playlist(self, playlist_id: str) -> bool:
        """Load a saved playlist as new session"""
        
        playlist = self.playlist_manager.get_playlist(playlist_id)
        if not playlist:
            logging.error(f"Playlist {playlist_id} not found")
            return False
        
        # Create session from saved playlist
        session = MusicSession(
            session_id=f"loaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            emotion_focus=playlist.emotion_focus,
            secondary_emotions=playlist.secondary_emotions,
            therapy_mode="custom",
            playlist=playlist,
            start_time=datetime.now(),
            user_preferences=self.default_preferences.copy()
        )
        
        self.current_session = session
        self.player_engine.load_playlist(playlist)
        
        # Increment play count
        self.playlist_manager.increment_play_count(playlist_id)
        
        logging.info(f"Loaded saved playlist: {playlist.name}")
        return True
    
    def get_recommendations(self, emotion: str, limit: int = 5) -> Dict[str, List[Dict]]:
        """Get playlist recommendations based on emotion"""
        
        recommendations = {
            "popular_playlists": [],
            "similar_emotions": [],
            "therapeutic_benefits": []
        }
        
        # Get popular playlists for this emotion
        popular = self.playlist_manager.search_playlists("", emotion=emotion)
        popular_sorted = sorted(popular, key=lambda p: p.play_count, reverse=True)
        
        for playlist in popular_sorted[:limit]:
            recommendations["popular_playlists"].append({
                "id": playlist.id,
                "name": playlist.name,
                "play_count": playlist.play_count,
                "song_count": len(playlist.songs),
                "duration": playlist.get_duration_formatted()
            })
        
        # Get songs for similar emotions
        similar_emotions = self._get_similar_emotions(emotion)
        for similar_emotion in similar_emotions[:3]:
            songs = self.song_database.get_vocal_songs_for_emotion(similar_emotion, 3)
            recommendations["similar_emotions"].append({
                "emotion": similar_emotion,
                "songs": [{"title": s.title, "artist": s.artist} for s in songs]
            })
        
        # Get therapeutic benefit recommendations
        therapeutic_songs = self.song_database.get_therapeutic_songs("mood elevation", 3)
        recommendations["therapeutic_benefits"].append({
            "benefit": "mood elevation",
            "songs": [{"title": s.title, "artist": s.artist} for s in therapeutic_songs]
        })
        
        return recommendations
    
    def _create_targeted_playlist(self, 
                                emotion: str, 
                                song_count: int, 
                                preferences: Dict[str, Any]) -> TherapeuticPlaylist:
        """Create targeted therapy playlist focusing on single emotion"""
        
        vocal_ratio = preferences.get("vocal_ratio", 0.7)
        
        # Get songs with high vocal content for emotional connection
        songs = self.song_database.get_mixed_playlist(emotion, vocal_ratio, song_count)
        
        # Filter by content rating
        content_rating = preferences.get("content_rating", "clean")
        songs = self.song_database.filter_content(songs, content_rating)
        
        # Create playlist
        metadata = {
            "name": f"Targeted Therapy - {emotion.title()}",
            "emotion_focus": emotion,
            "therapeutic_notes": f"Focused therapy session for {emotion} with {len(songs)} carefully selected songs",
            "tags": ["targeted", emotion, "therapy"]
        }
        
        return self.playlist_manager.create_playlist(songs, metadata)
    
    def _create_full_session_playlist(self, 
                                    emotion: str, 
                                    secondary_emotions: List[str], 
                                    preferences: Dict[str, Any]) -> TherapeuticPlaylist:
        """Create comprehensive therapy session playlist"""
        
        songs = []
        
        # Primary emotion songs (40% of playlist)
        primary_count = max(3, int(0.4 * 10))  # Assume 10 song session
        primary_songs = self.song_database.get_vocal_songs_for_emotion(emotion, primary_count)
        songs.extend(primary_songs)
        
        # Secondary emotion songs (40% of playlist)
        if secondary_emotions:
            secondary_count = max(2, int(0.4 * 10 / len(secondary_emotions)))
            for sec_emotion in secondary_emotions:
                sec_songs = self.song_database.get_vocal_songs_for_emotion(sec_emotion, secondary_count)
                songs.extend(sec_songs)
        
        # Balancing/neutral songs (20% of playlist)
        neutral_count = max(2, int(0.2 * 10))
        neutral_songs = self.song_database.get_songs_for_emotion("neutral", neutral_count)
        songs.extend(neutral_songs)
        
        # Shuffle for better flow
        random.shuffle(songs)
        
        # Create playlist
        metadata = {
            "name": f"Full Session - {emotion.title()}",
            "emotion_focus": emotion,
            "secondary_emotions": secondary_emotions,
            "therapeutic_notes": f"Comprehensive therapy session addressing {emotion} and related emotions",
            "tags": ["full_session", emotion, "comprehensive", "therapy"]
        }
        
        return self.playlist_manager.create_playlist(songs[:10], metadata)
    
    def _create_custom_playlist(self, 
                              emotion: str, 
                              secondary_emotions: List[str], 
                              song_count: int, 
                              preferences: Dict[str, Any]) -> TherapeuticPlaylist:
        """Create custom playlist based on user preferences"""
        
        songs = []
        
        # Distribute songs across emotions
        all_emotions = [emotion] + secondary_emotions
        songs_per_emotion = max(1, song_count // len(all_emotions))
        
        for emo in all_emotions:
            emo_songs = self.song_database.get_vocal_songs_for_emotion(emo, songs_per_emotion)
            songs.extend(emo_songs)
        
        # Fill remaining slots with primary emotion
        while len(songs) < song_count:
            additional = self.song_database.get_vocal_songs_for_emotion(emotion, song_count)
            # Add songs that aren't already in the list
            for song in additional:
                if song not in songs and len(songs) < song_count:
                    songs.append(song)
            
            # If we still don't have enough, break to avoid infinite loop
            if len(songs) >= song_count or len(additional) == 0:
                break
        
        # Apply energy preference (but ensure we still have enough songs)
        energy_pref = preferences.get("energy_preference", "balanced")
        if energy_pref == "low":
            filtered_songs = [s for s in songs if s.energy_level <= 4]
            if len(filtered_songs) >= song_count // 2:  # Keep at least half the songs
                songs = filtered_songs
        elif energy_pref == "high":
            filtered_songs = [s for s in songs if s.energy_level >= 7]
            if len(filtered_songs) >= song_count // 2:  # Keep at least half the songs
                songs = filtered_songs
        
        # Shuffle
        random.shuffle(songs)
        
        # Ensure we have exactly the requested count
        songs = songs[:song_count]
        
        # Create playlist
        metadata = {
            "name": f"Custom Playlist - {emotion.title()}",
            "emotion_focus": emotion,
            "secondary_emotions": secondary_emotions,
            "therapeutic_notes": f"Custom playlist tailored to user preferences",
            "tags": ["custom", emotion, "personalized"]
        }
        
        return self.playlist_manager.create_playlist(songs, metadata)
    
    def _get_similar_emotions(self, emotion: str) -> List[str]:
        """Get emotions similar to the given emotion"""
        
        emotion_clusters = {
            "joy": ["happiness", "excitement", "optimism"],
            "sadness": ["grief", "disappointment", "melancholy"],
            "anger": ["frustration", "annoyance", "rage"],
            "fear": ["anxiety", "worry", "nervousness"],
            "love": ["affection", "caring", "romance"],
            "neutral": ["calm", "peaceful", "balanced"]
        }
        
        # Find cluster containing the emotion
        for cluster_emotion, similar in emotion_clusters.items():
            if emotion.lower() in [cluster_emotion] + similar:
                return similar
        
        return []
    
    def _handle_song_change(self, song: EnhancedSong, state: PlaybackState):
        """Handle song change events"""
        
        logging.info(f"Song changed to: {song.title} by {song.artist}")
        
        # Update session if active
        if self.current_session:
            # Could add session-specific logic here
            pass
        
        # Notify external callback
        if self.on_song_change:
            self.on_song_change(song, state)
    
    def _handle_playlist_complete(self, playlist: TherapeuticPlaylist, state: PlaybackState):
        """Handle playlist completion"""
        
        logging.info(f"Playlist completed: {playlist.name}")
        
        # Save session automatically
        if self.current_session:
            self.save_current_session()
        
        # Notify external callback
        if self.on_session_complete:
            self.on_session_complete(self.current_session, state)
    
    def _handle_error(self, error: Exception, song: EnhancedSong, state: PlaybackState):
        """Handle playback errors"""
        
        logging.error(f"Playback error with {song.title}: {error}")
        
        # Notify external callback
        if self.on_error:
            self.on_error(error, song, state)
    
    def cleanup(self):
        """Clean up resources"""
        
        # Save current session if active
        if self.current_session:
            self.save_current_session()
        
        # Cleanup player engine
        self.player_engine.cleanup()
        
        logging.info("EnhancedMusicController cleaned up")