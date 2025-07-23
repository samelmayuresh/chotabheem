# test_player_engine.py - Unit tests for multi-song playback engine
import unittest
import time
import threading
from unittest.mock import Mock, patch
from utils.player_engine import PlayerEngine, PlaybackState, PlaybackStatus
from utils.enhanced_song_database import EnhancedSong
from utils.playlist_manager import TherapeuticPlaylist

class TestPlaybackState(unittest.TestCase):
    """Test cases for PlaybackState dataclass"""
    
    def setUp(self):
        """Set up test data"""
        self.test_song = EnhancedSong(
            title="Test Song", artist="Test Artist", url="https://example.com",
            duration=180, genre="Pop", mood="Happy"
        )
    
    def test_playback_state_creation(self):
        """Test creating a playback state"""
        state = PlaybackState(
            current_song=self.test_song,
            playlist_position=1,
            status=PlaybackStatus.PLAYING,
            auto_advance=True
        )
        
        self.assertEqual(state.current_song, self.test_song)
        self.assertEqual(state.playlist_position, 1)
        self.assertEqual(state.status, PlaybackStatus.PLAYING)
        self.assertTrue(state.auto_advance)
        self.assertEqual(state.error_count, 0)
    
    def test_progress_calculation(self):
        """Test progress percentage calculation"""
        state = PlaybackState()
        state.song_duration = 180.0
        state.song_position = 90.0
        
        progress = state.get_progress_percentage()
        self.assertEqual(progress, 50.0)
        
        # Test edge cases
        state.song_position = 0.0
        self.assertEqual(state.get_progress_percentage(), 0.0)
        
        state.song_position = 180.0
        self.assertEqual(state.get_progress_percentage(), 100.0)
        
        state.song_position = 200.0  # Over duration
        self.assertEqual(state.get_progress_percentage(), 100.0)
    
    def test_remaining_time_calculation(self):
        """Test remaining time calculation"""
        state = PlaybackState()
        state.song_duration = 180.0
        state.song_position = 90.0
        
        remaining = state.get_remaining_time()
        self.assertEqual(remaining, 90.0)
        
        # Test edge cases
        state.song_position = 180.0
        self.assertEqual(state.get_remaining_time(), 0.0)
        
        state.song_position = 200.0  # Over duration
        self.assertEqual(state.get_remaining_time(), 0.0)
    
    def test_time_formatting(self):
        """Test time formatting methods"""
        state = PlaybackState()
        state.song_position = 125.0  # 2:05
        state.song_duration = 185.0   # 3:05
        
        self.assertEqual(state.get_formatted_position(), "2:05")
        self.assertEqual(state.get_formatted_duration(), "3:05")
        
        # Test single digit seconds
        state.song_position = 65.0  # 1:05
        self.assertEqual(state.get_formatted_position(), "1:05")
    
    def test_user_interaction_tracking(self):
        """Test user interaction recording"""
        state = PlaybackState()
        
        # Add interaction
        state.add_user_interaction("play", {"song": "test"})
        
        self.assertEqual(len(state.user_interactions), 1)
        interaction = state.user_interactions[0]
        self.assertEqual(interaction["action"], "play")
        self.assertEqual(interaction["details"]["song"], "test")
        self.assertIn("timestamp", interaction)

class TestPlayerEngine(unittest.TestCase):
    """Test cases for PlayerEngine"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_songs = [
            EnhancedSong(
                title="Song 1", artist="Artist 1", url="https://example.com/1",
                duration=180, genre="Pop", mood="Happy"
            ),
            EnhancedSong(
                title="Song 2", artist="Artist 2", url="https://example.com/2",
                duration=200, genre="Rock", mood="Energetic"
            ),
            EnhancedSong(
                title="Song 3", artist="Artist 3", url="https://example.com/3",
                duration=160, genre="Folk", mood="Calm"
            )
        ]
        
        self.test_playlist = TherapeuticPlaylist(
            id="test-playlist",
            name="Test Playlist",
            songs=self.test_songs,
            emotion_focus="mixed"
        )
        
        # Mock callbacks
        self.on_song_change = Mock()
        self.on_playlist_complete = Mock()
        self.on_error = Mock()
        
        self.engine = PlayerEngine(
            on_song_change=self.on_song_change,
            on_playlist_complete=self.on_playlist_complete,
            on_error=self.on_error
        )
    
    def tearDown(self):
        """Clean up after tests"""
        self.engine.cleanup()
    
    def test_engine_initialization(self):
        """Test player engine initialization"""
        self.assertIsInstance(self.engine.state, PlaybackState)
        self.assertEqual(self.engine.state.status, PlaybackStatus.STOPPED)
        self.assertFalse(self.engine.is_monitoring)
    
    def test_load_playlist(self):
        """Test loading a playlist"""
        success = self.engine.load_playlist(self.test_playlist)
        
        self.assertTrue(success)
        self.assertEqual(self.engine.state.playlist, self.test_playlist)
        self.assertEqual(self.engine.state.playlist_position, 0)
        self.assertEqual(self.engine.state.current_song, self.test_songs[0])
        self.assertIsNotNone(self.engine.state.session_start_time)
    
    def test_load_playlist_with_start_position(self):
        """Test loading playlist with custom start position"""
        success = self.engine.load_playlist(self.test_playlist, start_position=1)
        
        self.assertTrue(success)
        self.assertEqual(self.engine.state.playlist_position, 1)
        self.assertEqual(self.engine.state.current_song, self.test_songs[1])
    
    def test_load_empty_playlist(self):
        """Test loading empty playlist"""
        empty_playlist = TherapeuticPlaylist(
            id="empty", name="Empty", songs=[], emotion_focus="neutral"
        )
        
        success = self.engine.load_playlist(empty_playlist)
        self.assertFalse(success)
    
    def test_load_playlist_invalid_position(self):
        """Test loading playlist with invalid start position"""
        success = self.engine.load_playlist(self.test_playlist, start_position=10)
        self.assertFalse(success)
        
        success = self.engine.load_playlist(self.test_playlist, start_position=-1)
        self.assertFalse(success)
    
    def test_play_song(self):
        """Test playing a song"""
        self.engine.load_playlist(self.test_playlist)
        
        success = self.engine.play_song()
        
        self.assertTrue(success)
        self.assertEqual(self.engine.state.status, PlaybackStatus.PLAYING)
        self.assertEqual(self.engine.state.current_song, self.test_songs[0])
        self.assertIsNotNone(self.engine.state.song_start_time)
        self.assertEqual(self.engine.state.song_duration, 180.0)
        
        # Check callback was called
        self.on_song_change.assert_called_once()
    
    def test_play_specific_song(self):
        """Test playing a specific song"""
        specific_song = self.test_songs[1]
        
        success = self.engine.play_song(specific_song)
        
        self.assertTrue(success)
        self.assertEqual(self.engine.state.current_song, specific_song)
        self.assertEqual(self.engine.state.status, PlaybackStatus.PLAYING)
    
    def test_pause_and_resume(self):
        """Test pause and resume functionality"""
        self.engine.load_playlist(self.test_playlist)
        self.engine.play_song()
        
        # Test pause
        success = self.engine.pause()
        self.assertTrue(success)
        self.assertEqual(self.engine.state.status, PlaybackStatus.PAUSED)
        
        # Test resume
        success = self.engine.resume()
        self.assertTrue(success)
        self.assertEqual(self.engine.state.status, PlaybackStatus.PLAYING)
    
    def test_stop(self):
        """Test stop functionality"""
        self.engine.load_playlist(self.test_playlist)
        self.engine.play_song()
        
        success = self.engine.stop()
        
        self.assertTrue(success)
        self.assertEqual(self.engine.state.status, PlaybackStatus.STOPPED)
        self.assertEqual(self.engine.state.song_position, 0.0)
        self.assertFalse(self.engine.is_monitoring)
    
    def test_skip_to_next(self):
        """Test skipping to next song"""
        self.engine.load_playlist(self.test_playlist)
        self.engine.play_song()
        
        # Skip to next song
        success = self.engine.skip_to_next()
        
        self.assertTrue(success)
        self.assertEqual(self.engine.state.playlist_position, 1)
        self.assertEqual(self.engine.state.current_song, self.test_songs[1])
    
    def test_skip_to_next_at_end(self):
        """Test skipping to next at end of playlist"""
        self.engine.load_playlist(self.test_playlist, start_position=2)  # Last song
        self.engine.play_song()
        
        success = self.engine.skip_to_next()
        
        # Should handle playlist completion
        self.assertTrue(success)
        self.assertEqual(self.engine.state.status, PlaybackStatus.STOPPED)
    
    def test_skip_to_previous(self):
        """Test skipping to previous song"""
        self.engine.load_playlist(self.test_playlist, start_position=1)
        self.engine.play_song()
        
        success = self.engine.skip_to_previous()
        
        self.assertTrue(success)
        self.assertEqual(self.engine.state.playlist_position, 0)
        self.assertEqual(self.engine.state.current_song, self.test_songs[0])
    
    def test_skip_to_previous_restart_song(self):
        """Test skipping to previous when deep in song (should restart)"""
        self.engine.load_playlist(self.test_playlist)
        self.engine.play_song()
        
        # Simulate being 5 seconds into the song
        self.engine.state.song_position = 5.0
        
        success = self.engine.skip_to_previous()
        
        self.assertTrue(success)
        self.assertEqual(self.engine.state.song_position, 0.0)
        self.assertEqual(self.engine.state.playlist_position, 0)  # Same song
    
    def test_seek_to_position(self):
        """Test seeking to specific position"""
        self.engine.load_playlist(self.test_playlist)
        self.engine.play_song()
        
        success = self.engine.seek_to_position(90.0)
        
        self.assertTrue(success)
        self.assertEqual(self.engine.state.song_position, 90.0)
    
    def test_seek_position_clamping(self):
        """Test that seek position is clamped to valid range"""
        self.engine.load_playlist(self.test_playlist)
        self.engine.play_song()
        
        # Test negative position
        self.engine.seek_to_position(-10.0)
        self.assertEqual(self.engine.state.song_position, 0.0)
        
        # Test position beyond duration
        self.engine.seek_to_position(300.0)
        self.assertEqual(self.engine.state.song_position, 180.0)  # Song duration
    
    def test_set_volume(self):
        """Test volume control"""
        success = self.engine.set_volume(0.5)
        
        self.assertTrue(success)
        self.assertEqual(self.engine.state.volume, 0.5)
        
        # Test clamping
        self.engine.set_volume(-0.1)
        self.assertEqual(self.engine.state.volume, 0.0)
        
        self.engine.set_volume(1.5)
        self.assertEqual(self.engine.state.volume, 1.0)
    
    def test_set_repeat_mode(self):
        """Test repeat mode settings"""
        # Test valid modes
        for mode in ["none", "song", "playlist"]:
            success = self.engine.set_repeat_mode(mode)
            self.assertTrue(success)
            self.assertEqual(self.engine.state.repeat_mode, mode)
        
        # Test invalid mode
        success = self.engine.set_repeat_mode("invalid")
        self.assertFalse(success)
    
    def test_set_shuffle(self):
        """Test shuffle mode"""
        success = self.engine.set_shuffle(True)
        
        self.assertTrue(success)
        self.assertTrue(self.engine.state.shuffle)
        
        success = self.engine.set_shuffle(False)
        self.assertTrue(success)
        self.assertFalse(self.engine.state.shuffle)
    
    def test_repeat_playlist_mode(self):
        """Test playlist repeat functionality"""
        self.engine.load_playlist(self.test_playlist, start_position=2)  # Last song
        self.engine.set_repeat_mode("playlist")
        
        # Skip to next (should loop to beginning)
        success = self.engine.skip_to_next()
        
        self.assertTrue(success)
        self.assertEqual(self.engine.state.playlist_position, 0)
        self.assertEqual(self.engine.state.current_song, self.test_songs[0])
    
    def test_user_interaction_tracking(self):
        """Test that user interactions are tracked"""
        self.engine.load_playlist(self.test_playlist)
        
        # Perform various actions
        self.engine.play_song()
        self.engine.pause()
        self.engine.resume()
        self.engine.skip_to_next()
        
        # Check interactions were recorded
        interactions = self.engine.state.user_interactions
        self.assertGreater(len(interactions), 0)
        
        # Check specific actions
        actions = [i["action"] for i in interactions]
        self.assertIn("play", actions)
        self.assertIn("pause", actions)
        self.assertIn("resume", actions)
        self.assertIn("skip_next", actions)
    
    def test_session_statistics(self):
        """Test session statistics generation"""
        self.engine.load_playlist(self.test_playlist)
        self.engine.play_song()
        self.engine.skip_to_next()
        
        stats = self.engine.get_session_statistics()
        
        self.assertIn("session_duration", stats)
        self.assertIn("songs_played", stats)
        self.assertIn("completion_rate", stats)
        self.assertIn("error_count", stats)
        self.assertIn("user_interactions", stats)
        self.assertIn("interaction_breakdown", stats)
        
        self.assertGreater(stats["songs_played"], 0)
        self.assertGreater(stats["user_interactions"], 0)
    
    @patch('time.sleep')  # Mock sleep to speed up test
    def test_error_handling_and_recovery(self):
        """Test error handling and auto-recovery"""
        self.engine.load_playlist(self.test_playlist)
        
        # Simulate error during playback
        with patch.object(self.engine, 'play_song', side_effect=Exception("Test error")) as mock_play:
            # First call raises exception, subsequent calls succeed
            mock_play.side_effect = [Exception("Test error"), True, True]
            
            # This should trigger error handling
            success = self.engine.play_song()
            
            # Check error was handled
            self.assertGreater(self.engine.state.error_count, 0)
            self.assertIsNotNone(self.engine.state.last_error)
    
    def test_cleanup(self):
        """Test engine cleanup"""
        self.engine.load_playlist(self.test_playlist)
        self.engine.play_song()
        
        self.engine.cleanup()
        
        self.assertEqual(self.engine.state.status, PlaybackStatus.STOPPED)
        self.assertFalse(self.engine.is_monitoring)

if __name__ == "__main__":
    unittest.main()