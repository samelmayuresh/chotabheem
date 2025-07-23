# test_playlist_manager.py - Unit tests for playlist management system
import unittest
import tempfile
import os
from datetime import datetime
from utils.playlist_manager import TherapeuticPlaylist, PlaylistManager
from utils.enhanced_song_database import EnhancedSong

class TestTherapeuticPlaylist(unittest.TestCase):
    """Test cases for TherapeuticPlaylist dataclass"""
    
    def setUp(self):
        """Set up test data"""
        self.test_songs = [
            EnhancedSong(
                title="Happy Song", artist="Test Artist", url="https://example.com/1",
                duration=180, genre="Pop", mood="Happy", has_vocals=True,
                energy_level=8, therapeutic_benefits=["mood elevation"]
            ),
            EnhancedSong(
                title="Calm Song", artist="Test Artist 2", url="https://example.com/2",
                duration=240, genre="Ambient", mood="Calm", has_vocals=False,
                energy_level=3, therapeutic_benefits=["relaxation"]
            ),
            EnhancedSong(
                title="Empowering Song", artist="Test Artist 3", url="https://example.com/3",
                duration=200, genre="Rock", mood="Empowering", has_vocals=True,
                energy_level=9, therapeutic_benefits=["empowerment"], release_year=2020
            )
        ]
    
    def test_playlist_creation(self):
        """Test creating a therapeutic playlist"""
        playlist = TherapeuticPlaylist(
            id="test-123",
            name="Test Playlist",
            songs=self.test_songs,
            emotion_focus="joy",
            secondary_emotions=["confidence"],
            therapeutic_notes="Test therapy session"
        )
        
        self.assertEqual(playlist.id, "test-123")
        self.assertEqual(playlist.name, "Test Playlist")
        self.assertEqual(len(playlist.songs), 3)
        self.assertEqual(playlist.emotion_focus, "joy")
        self.assertIn("confidence", playlist.secondary_emotions)
        self.assertEqual(playlist.total_duration, 620)  # 180 + 240 + 200
    
    def test_duration_calculation(self):
        """Test automatic duration calculation"""
        playlist = TherapeuticPlaylist(
            id="test-duration",
            name="Duration Test",
            songs=self.test_songs,
            emotion_focus="neutral"
        )
        
        expected_duration = sum(song.duration for song in self.test_songs)
        self.assertEqual(playlist.total_duration, expected_duration)
    
    def test_vocal_ratio_calculation(self):
        """Test vocal/instrumental ratio calculation"""
        playlist = TherapeuticPlaylist(
            id="test-ratio",
            name="Ratio Test",
            songs=self.test_songs,
            emotion_focus="neutral"
        )
        
        # 2 vocal songs out of 3 total = 2/3 ≈ 0.67
        expected_ratio = 2/3
        self.assertAlmostEqual(playlist.vocal_instrumental_ratio, expected_ratio, places=2)
    
    def test_duration_formatting(self):
        """Test duration formatting"""
        # Test minutes:seconds format
        short_playlist = TherapeuticPlaylist(
            id="short",
            name="Short",
            songs=[self.test_songs[0]],  # 180 seconds = 3:00
            emotion_focus="neutral"
        )
        self.assertEqual(short_playlist.get_duration_formatted(), "3:00")
        
        # Test hours:minutes:seconds format
        long_songs = [EnhancedSong(
            title="Long Song", artist="Test", url="https://example.com",
            duration=3661, genre="Ambient", mood="Calm"  # 1:01:01
        )]
        long_playlist = TherapeuticPlaylist(
            id="long",
            name="Long",
            songs=long_songs,
            emotion_focus="neutral"
        )
        self.assertEqual(long_playlist.get_duration_formatted(), "1:01:01")
    
    def test_therapeutic_summary(self):
        """Test therapeutic summary generation"""
        playlist = TherapeuticPlaylist(
            id="summary-test",
            name="Summary Test",
            songs=self.test_songs,
            emotion_focus="mixed"
        )
        
        summary = playlist.get_therapeutic_summary()
        
        # Check summary structure
        self.assertIn("primary_benefits", summary)
        self.assertIn("average_energy_level", summary)
        self.assertIn("average_emotional_intensity", summary)
        self.assertIn("vocal_percentage", summary)
        self.assertIn("genre_distribution", summary)
        
        # Check values
        self.assertIsInstance(summary["primary_benefits"], list)
        self.assertIsInstance(summary["average_energy_level"], float)
        self.assertGreater(summary["vocal_percentage"], 0)

class TestPlaylistManager(unittest.TestCase):
    """Test cases for PlaylistManager"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary file for testing
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        self.temp_file.close()
        
        self.manager = PlaylistManager(storage_path=self.temp_file.name)
        
        # Test songs
        self.test_songs = [
            EnhancedSong(
                title="Test Song 1", artist="Artist 1", url="https://example.com/1",
                duration=180, genre="Pop", mood="Happy", has_vocals=True
            ),
            EnhancedSong(
                title="Test Song 2", artist="Artist 2", url="https://example.com/2",
                duration=200, genre="Rock", mood="Energetic", has_vocals=True
            )
        ]
    
    def tearDown(self):
        """Clean up test environment"""
        try:
            os.unlink(self.temp_file.name)
        except:
            pass
    
    def test_create_playlist(self):
        """Test playlist creation"""
        metadata = {
            "name": "Test Playlist",
            "emotion_focus": "joy",
            "secondary_emotions": ["energy"],
            "therapeutic_notes": "Test notes",
            "tags": ["test", "happy"]
        }
        
        playlist = self.manager.create_playlist(self.test_songs, metadata)
        
        self.assertIsInstance(playlist, TherapeuticPlaylist)
        self.assertEqual(playlist.name, "Test Playlist")
        self.assertEqual(playlist.emotion_focus, "joy")
        self.assertIn("energy", playlist.secondary_emotions)
        self.assertEqual(playlist.therapeutic_notes, "Test notes")
        self.assertIn("test", playlist.tags)
        self.assertEqual(len(playlist.songs), 2)
        
        # Check that playlist is stored in manager
        self.assertIn(playlist.id, self.manager.playlists)
    
    def test_modify_playlist_add_song(self):
        """Test adding song to playlist"""
        # Create initial playlist
        playlist = self.manager.create_playlist(self.test_songs[:1], {"name": "Test"})
        
        # Add a song
        new_song = self.test_songs[1]
        modified = self.manager.modify_playlist(
            playlist.id, "add_song", song=new_song, position=1
        )
        
        self.assertIsNotNone(modified)
        self.assertEqual(len(modified.songs), 2)
        self.assertEqual(modified.songs[1].title, "Test Song 2")
        self.assertTrue(modified.user_customized)
    
    def test_modify_playlist_remove_song(self):
        """Test removing song from playlist"""
        # Create playlist with 2 songs
        playlist = self.manager.create_playlist(self.test_songs, {"name": "Test"})
        
        # Remove first song
        modified = self.manager.modify_playlist(playlist.id, "remove_song", position=0)
        
        self.assertIsNotNone(modified)
        self.assertEqual(len(modified.songs), 1)
        self.assertEqual(modified.songs[0].title, "Test Song 2")
        self.assertTrue(modified.user_customized)
    
    def test_modify_playlist_reorder(self):
        """Test reordering songs in playlist"""
        playlist = self.manager.create_playlist(self.test_songs, {"name": "Test"})
        
        # Reverse order
        modified = self.manager.modify_playlist(
            playlist.id, "reorder_songs", new_order=[1, 0]
        )
        
        self.assertIsNotNone(modified)
        self.assertEqual(modified.songs[0].title, "Test Song 2")
        self.assertEqual(modified.songs[1].title, "Test Song 1")
        self.assertTrue(modified.user_customized)
    
    def test_modify_playlist_update_metadata(self):
        """Test updating playlist metadata"""
        playlist = self.manager.create_playlist(self.test_songs, {"name": "Original"})
        
        modified = self.manager.modify_playlist(
            playlist.id, "update_metadata",
            name="Updated Name",
            therapeutic_notes="Updated notes",
            user_rating=5
        )
        
        self.assertIsNotNone(modified)
        self.assertEqual(modified.name, "Updated Name")
        self.assertEqual(modified.therapeutic_notes, "Updated notes")
        self.assertEqual(modified.user_rating, 5)
        self.assertTrue(modified.user_customized)
    
    def test_save_and_load_playlist(self):
        """Test playlist persistence"""
        # Create and save playlist
        playlist = self.manager.create_playlist(self.test_songs, {"name": "Persistent"})
        success = self.manager.save_playlist(playlist)
        self.assertTrue(success)
        
        # Create new manager instance to test loading
        new_manager = PlaylistManager(storage_path=self.temp_file.name)
        
        # Check that playlist was loaded
        self.assertIn(playlist.id, new_manager.playlists)
        loaded_playlist = new_manager.playlists[playlist.id]
        self.assertEqual(loaded_playlist.name, "Persistent")
        self.assertEqual(len(loaded_playlist.songs), 2)
    
    def test_search_playlists(self):
        """Test playlist search functionality"""
        # Create test playlists
        playlist1 = self.manager.create_playlist(
            self.test_songs, 
            {"name": "Happy Songs", "emotion_focus": "joy", "tags": ["upbeat"]}
        )
        playlist2 = self.manager.create_playlist(
            self.test_songs, 
            {"name": "Sad Songs", "emotion_focus": "sadness", "tags": ["melancholy"]}
        )
        
        # Search by name
        results = self.manager.search_playlists("Happy")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, playlist1.id)
        
        # Search by emotion
        results = self.manager.search_playlists("", emotion="sadness")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, playlist2.id)
        
        # Search by tags
        results = self.manager.search_playlists("", tags=["upbeat"])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, playlist1.id)
    
    def test_playlist_validation(self):
        """Test playlist validation"""
        # Valid playlist
        valid_playlist = self.manager.create_playlist(
            self.test_songs, {"name": "Valid", "emotion_focus": "joy"}
        )
        validation = self.manager.validate_playlist(valid_playlist)
        self.assertTrue(validation["is_valid"])
        
        # Empty playlist (invalid)
        empty_playlist = TherapeuticPlaylist(
            id="empty", name="Empty", songs=[], emotion_focus="neutral"
        )
        validation = self.manager.validate_playlist(empty_playlist)
        self.assertFalse(validation["is_valid"])
        self.assertGreater(len(validation["errors"]), 0)
    
    def test_export_playlist(self):
        """Test playlist export functionality"""
        playlist = self.manager.create_playlist(
            self.test_songs, {"name": "Export Test", "emotion_focus": "joy"}
        )
        
        # Test JSON export
        json_export = self.manager.export_playlist(playlist.id, "json")
        self.assertIsNotNone(json_export)
        self.assertIn("Export Test", json_export)
        
        # Test text export
        text_export = self.manager.export_playlist(playlist.id, "text")
        self.assertIsNotNone(text_export)
        self.assertIn("# Export Test", text_export)
        self.assertIn("Test Song 1", text_export)
        
        # Test M3U export
        m3u_export = self.manager.export_playlist(playlist.id, "m3u")
        self.assertIsNotNone(m3u_export)
        self.assertIn("#EXTM3U", m3u_export)
        self.assertIn("https://example.com/1", m3u_export)
    
    def test_play_count_tracking(self):
        """Test play count increment"""
        playlist = self.manager.create_playlist(self.test_songs, {"name": "Play Test"})
        
        # Initial play count should be 0
        self.assertEqual(playlist.play_count, 0)
        
        # Increment play count
        success = self.manager.increment_play_count(playlist.id)
        self.assertTrue(success)
        
        # Check updated count
        updated_playlist = self.manager.get_playlist(playlist.id)
        self.assertEqual(updated_playlist.play_count, 1)
    
    def test_get_popular_playlists(self):
        """Test getting popular playlists"""
        # Create playlists with different play counts
        playlist1 = self.manager.create_playlist(self.test_songs, {"name": "Popular"})
        playlist2 = self.manager.create_playlist(self.test_songs, {"name": "Less Popular"})
        
        # Set play counts
        self.manager.increment_play_count(playlist1.id)
        self.manager.increment_play_count(playlist1.id)
        self.manager.increment_play_count(playlist2.id)
        
        # Get popular playlists
        popular = self.manager.get_popular_playlists(limit=2)
        
        self.assertEqual(len(popular), 2)
        self.assertEqual(popular[0].id, playlist1.id)  # Should be first (higher play count)
        self.assertEqual(popular[1].id, playlist2.id)

if __name__ == "__main__":
    unittest.main()