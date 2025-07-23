# test_enhanced_music_controller.py - Unit tests for enhanced music controller
import unittest
import tempfile
import os
from unittest.mock import Mock, patch
from datetime import datetime

from utils.enhanced_music_controller import EnhancedMusicController, MusicSession
from utils.enhanced_song_database import EnhancedSong
from utils.playlist_manager import TherapeuticPlaylist

class TestMusicSession(unittest.TestCase):
    """Test cases for MusicSession dataclass"""
    
    def test_music_session_creation(self):
        """Test creating a music session"""
        
        test_songs = [
            EnhancedSong(
                title="Test Song", artist="Test Artist", url="https://example.com",
                duration=180, genre="Pop", mood="Happy"
            )
        ]
        
        test_playlist = TherapeuticPlaylist(
            id="test-playlist",
            name="Test Playlist",
            songs=test_songs,
            emotion_focus="joy"
        )
        
        session = MusicSession(
            session_id="test-session-123",
            emotion_focus="joy",
            secondary_emotions=["energy"],
            therapy_mode="targeted",
            playlist=test_playlist,
            start_time=datetime.now(),
            user_preferences={"volume": 0.8},
            session_notes="Test session"
        )
        
        self.assertEqual(session.session_id, "test-session-123")
        self.assertEqual(session.emotion_focus, "joy")
        self.assertIn("energy", session.secondary_emotions)
        self.assertEqual(session.therapy_mode, "targeted")
        self.assertEqual(session.playlist, test_playlist)
        self.assertEqual(session.user_preferences["volume"], 0.8)
        self.assertEqual(session.session_notes, "Test session")

class TestEnhancedMusicController(unittest.TestCase):
    """Test cases for EnhancedMusicController"""
    
    def setUp(self):
        """Set up test environment"""
        
        # Create temporary file for testing
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        self.temp_file.close()
        
        # Mock callbacks
        self.on_session_start = Mock()
        self.on_session_complete = Mock()
        self.on_song_change = Mock()
        self.on_error = Mock()
        
        # Create controller
        self.controller = EnhancedMusicController(
            storage_path=self.temp_file.name,
            on_session_start=self.on_session_start,
            on_session_complete=self.on_session_complete,
            on_song_change=self.on_song_change,
            on_error=self.on_error
        )
    
    def tearDown(self):
        """Clean up test environment"""
        self.controller.cleanup()
        try:
            os.unlink(self.temp_file.name)
        except:
            pass
    
    def test_controller_initialization(self):
        """Test controller initialization"""
        
        self.assertIsNotNone(self.controller.song_database)
        self.assertIsNotNone(self.controller.playlist_manager)
        self.assertIsNotNone(self.controller.player_engine)
        self.assertIsNone(self.controller.current_session)
        self.assertEqual(len(self.controller.session_history), 0)
        
        # Check default preferences
        self.assertIn("vocal_ratio", self.controller.default_preferences)
        self.assertIn("auto_advance", self.controller.default_preferences)
        self.assertEqual(self.controller.default_preferences["vocal_ratio"], 0.7)
    
    def test_create_targeted_session(self):
        """Test creating targeted therapy session"""
        
        session = self.controller.create_continuous_session(
            emotion="joy",
            mode="targeted",
            song_count=5
        )
        
        self.assertIsInstance(session, MusicSession)
        self.assertEqual(session.emotion_focus, "joy")
        self.assertEqual(session.therapy_mode, "targeted")
        self.assertIsNotNone(session.playlist)
        self.assertLessEqual(len(session.playlist.songs), 5)
        
        # Check that session is set as current
        self.assertEqual(self.controller.current_session, session)
        
        # Check callback was called
        self.on_session_start.assert_called_once_with(session)
    
    def test_create_full_session(self):
        """Test creating full therapy session"""
        
        session = self.controller.create_continuous_session(
            emotion="sadness",
            mode="full_session",
            secondary_emotions=["grief", "healing"]
        )
        
        self.assertEqual(session.emotion_focus, "sadness")
        self.assertEqual(session.therapy_mode, "full_session")
        self.assertIn("grief", session.secondary_emotions)
        self.assertIn("healing", session.secondary_emotions)
        self.assertGreater(len(session.playlist.songs), 5)  # Full session should have more songs
    
    def test_create_custom_session(self):
        """Test creating custom therapy session"""
        
        user_prefs = {
            "vocal_ratio": 0.9,
            "volume": 0.6,
            "energy_preference": "high"
        }
        
        session = self.controller.create_continuous_session(
            emotion="anger",
            mode="custom",
            song_count=7,
            secondary_emotions=["empowerment"],
            user_preferences=user_prefs
        )
        
        self.assertEqual(session.emotion_focus, "anger")
        self.assertEqual(session.therapy_mode, "custom")
        self.assertEqual(session.user_preferences["vocal_ratio"], 0.9)
        self.assertEqual(session.user_preferences["volume"], 0.6)
        self.assertEqual(len(session.playlist.songs), 7)
    
    def test_invalid_therapy_mode(self):
        """Test creating session with invalid therapy mode"""
        
        with self.assertRaises(ValueError):
            self.controller.create_continuous_session(
                emotion="joy",
                mode="invalid_mode"
            )
    
    def test_get_enhanced_english_playlist(self):
        """Test getting enhanced English playlist"""
        
        playlist = self.controller.get_enhanced_english_playlist(
            emotion="love",
            count=4,
            vocal_ratio=0.8
        )
        
        self.assertIsInstance(playlist, list)
        self.assertLessEqual(len(playlist), 4)
        
        # Check that most songs are vocal and English
        vocal_count = sum(1 for song in playlist if song.has_vocals)
        english_count = sum(1 for song in playlist if song.language == "english")
        
        self.assertGreaterEqual(vocal_count, len(playlist) * 0.6)  # At least 60% vocal
        self.assertEqual(english_count, len(playlist))  # All should be English
    
    def test_playback_controls(self):
        """Test playback control actions"""
        
        # Create session first
        session = self.controller.create_continuous_session("joy", "targeted", 3)
        
        # Test play
        success = self.controller.handle_playback_control("play")
        self.assertTrue(success)
        
        # Test pause
        success = self.controller.handle_playback_control("pause")
        self.assertTrue(success)
        
        # Test resume
        success = self.controller.handle_playback_control("resume")
        self.assertTrue(success)
        
        # Test volume
        success = self.controller.handle_playback_control("volume", volume=0.5)
        self.assertTrue(success)
        
        # Test skip next
        success = self.controller.handle_playback_control("skip_next")
        self.assertTrue(success)
        
        # Test skip previous
        success = self.controller.handle_playback_control("skip_previous")
        self.assertTrue(success)
        
        # Test seek
        success = self.controller.handle_playback_control("seek", position=30.0)
        self.assertTrue(success)
        
        # Test repeat mode
        success = self.controller.handle_playback_control("repeat", mode="playlist")
        self.assertTrue(success)
        
        # Test shuffle
        success = self.controller.handle_playback_control("shuffle", enabled=True)
        self.assertTrue(success)
        
        # Test stop
        success = self.controller.handle_playback_control("stop")
        self.assertTrue(success)
    
    def test_invalid_playback_control(self):
        """Test invalid playback control action"""
        
        success = self.controller.handle_playback_control("invalid_action")
        self.assertFalse(success)
    
    def test_playback_control_without_session(self):
        """Test playback control without active session"""
        
        # Try to play without session
        success = self.controller.handle_playback_control("play")
        self.assertFalse(success)
    
    def test_get_session_info(self):
        """Test getting session information"""
        
        # No session initially
        info = self.controller.get_session_info()
        self.assertIsNone(info)
        
        # Create session
        session = self.controller.create_continuous_session("fear", "targeted", 4)
        
        info = self.controller.get_session_info()
        self.assertIsNotNone(info)
        self.assertEqual(info["emotion_focus"], "fear")
        self.assertEqual(info["therapy_mode"], "targeted")
        self.assertEqual(info["total_songs"], len(session.playlist.songs))
        self.assertIn("session_id", info)
        self.assertIn("playback_status", info)
        self.assertIn("current_position", info)
    
    def test_save_current_session(self):
        """Test saving current session"""
        
        # No session to save initially
        success = self.controller.save_current_session()
        self.assertFalse(success)
        
        # Create and save session
        session = self.controller.create_continuous_session("anxiety", "targeted", 3)
        success = self.controller.save_current_session()
        self.assertTrue(success)
        
        # Check session was added to history
        self.assertIn(session, self.controller.session_history)
        self.assertIsNotNone(session.session_notes)
    
    def test_load_saved_playlist(self):
        """Test loading saved playlist"""
        
        # Create and save a session first
        original_session = self.controller.create_continuous_session("love", "targeted", 3)
        self.controller.save_current_session()
        
        playlist_id = original_session.playlist.id
        
        # Load the saved playlist
        success = self.controller.load_saved_playlist(playlist_id)
        self.assertTrue(success)
        
        # Check new session was created
        self.assertIsNotNone(self.controller.current_session)
        self.assertNotEqual(self.controller.current_session.session_id, original_session.session_id)
        self.assertEqual(self.controller.current_session.emotion_focus, "love")
    
    def test_load_nonexistent_playlist(self):
        """Test loading non-existent playlist"""
        
        success = self.controller.load_saved_playlist("nonexistent-id")
        self.assertFalse(success)
    
    def test_get_recommendations(self):
        """Test getting playlist recommendations"""
        
        recommendations = self.controller.get_recommendations("joy", limit=3)
        
        self.assertIsInstance(recommendations, dict)
        self.assertIn("popular_playlists", recommendations)
        self.assertIn("similar_emotions", recommendations)
        self.assertIn("therapeutic_benefits", recommendations)
        
        # Check structure
        self.assertIsInstance(recommendations["popular_playlists"], list)
        self.assertIsInstance(recommendations["similar_emotions"], list)
        self.assertIsInstance(recommendations["therapeutic_benefits"], list)
    
    def test_start_playback(self):
        """Test starting playback"""
        
        # No session - should fail
        success = self.controller.start_playback()
        self.assertFalse(success)
        
        # Create session and start playback
        session = self.controller.create_continuous_session("neutral", "targeted", 2)
        success = self.controller.start_playback()
        self.assertTrue(success)
        
        # Check playback state
        state = self.controller.get_playback_state()
        self.assertIsNotNone(state.current_song)
    
    def test_callback_integration(self):
        """Test that callbacks are properly integrated"""
        
        # Create session (should trigger on_session_start)
        session = self.controller.create_continuous_session("joy", "targeted", 2)
        self.on_session_start.assert_called_once()
        
        # Start playback and skip (should trigger on_song_change)
        self.controller.start_playback()
        self.controller.handle_playback_control("skip_next")
        
        # Note: on_song_change might be called multiple times due to internal player events
        self.assertGreater(self.on_song_change.call_count, 0)
    
    def test_user_preferences_application(self):
        """Test that user preferences are applied correctly"""
        
        user_prefs = {
            "volume": 0.3,
            "vocal_ratio": 0.9,
            "content_rating": "clean"
        }
        
        session = self.controller.create_continuous_session(
            emotion="happiness",
            mode="targeted",
            song_count=3,
            user_preferences=user_prefs
        )
        
        # Check preferences were stored
        self.assertEqual(session.user_preferences["volume"], 0.3)
        self.assertEqual(session.user_preferences["vocal_ratio"], 0.9)
        
        # Start playback to apply volume
        self.controller.start_playback()
        
        # Check volume was applied to player
        state = self.controller.get_playback_state()
        self.assertEqual(state.volume, 0.3)
    
    def test_cleanup(self):
        """Test controller cleanup"""
        
        # Create session
        session = self.controller.create_continuous_session("calm", "targeted", 2)
        self.controller.start_playback()
        
        # Cleanup should save session and stop playback
        self.controller.cleanup()
        
        # Check session was saved
        self.assertIn(session, self.controller.session_history)

if __name__ == "__main__":
    unittest.main()