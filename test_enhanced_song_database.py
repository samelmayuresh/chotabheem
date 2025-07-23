# test_enhanced_song_database.py - Unit tests for enhanced song database
import unittest
from utils.enhanced_song_database import EnhancedSong, EnhancedSongDatabase

class TestEnhancedSong(unittest.TestCase):
    """Test cases for EnhancedSong dataclass"""
    
    def test_enhanced_song_creation(self):
        """Test creating an enhanced song with all fields"""
        song = EnhancedSong(
            title="Test Song",
            artist="Test Artist", 
            url="https://example.com",
            duration=180,
            genre="Pop",
            mood="Happy",
            has_vocals=True,
            lyrics_theme="joy",
            energy_level=8,
            therapeutic_benefits=["mood elevation", "energy boost"],
            content_rating="clean",
            popularity_score=85,
            language="english",
            release_year=2020,
            emotional_intensity=7
        )
        
        self.assertEqual(song.title, "Test Song")
        self.assertEqual(song.artist, "Test Artist")
        self.assertTrue(song.has_vocals)
        self.assertEqual(song.energy_level, 8)
        self.assertIn("mood elevation", song.therapeutic_benefits)
        self.assertEqual(song.content_rating, "clean")
        self.assertEqual(song.popularity_score, 85)
    
    def test_enhanced_song_defaults(self):
        """Test enhanced song with default values"""
        song = EnhancedSong(
            title="Simple Song",
            artist="Simple Artist",
            url="https://example.com",
            duration=200,
            genre="Folk",
            mood="Calm"
        )
        
        # Test defaults
        self.assertTrue(song.has_vocals)  # Default True
        self.assertEqual(song.energy_level, 5)  # Default 5
        self.assertEqual(song.therapeutic_benefits, [])  # Default empty list
        self.assertEqual(song.content_rating, "clean")  # Default clean
        self.assertEqual(song.popularity_score, 50)  # Default 50
        self.assertEqual(song.language, "english")  # Default english

class TestEnhancedSongDatabase(unittest.TestCase):
    """Test cases for EnhancedSongDatabase"""
    
    def setUp(self):
        """Set up test database"""
        self.db = EnhancedSongDatabase()
    
    def test_database_initialization(self):
        """Test that database initializes with emotion playlists"""
        self.assertIsInstance(self.db.emotion_playlists, dict)
        self.assertIn("joy", self.db.emotion_playlists)
        self.assertIn("sadness", self.db.emotion_playlists)
        self.assertIn("anger", self.db.emotion_playlists)
        self.assertIn("fear", self.db.emotion_playlists)
        self.assertIn("anxiety", self.db.emotion_playlists)
        self.assertIn("love", self.db.emotion_playlists)
        self.assertIn("neutral", self.db.emotion_playlists)
    
    def test_emotion_playlists_have_enhanced_songs(self):
        """Test that emotion playlists contain EnhancedSong objects"""
        for emotion, songs in self.db.emotion_playlists.items():
            self.assertIsInstance(songs, list)
            self.assertGreater(len(songs), 0, f"No songs found for emotion: {emotion}")
            
            for song in songs:
                self.assertIsInstance(song, EnhancedSong)
                self.assertIsInstance(song.has_vocals, bool)
                self.assertIsInstance(song.energy_level, int)
                self.assertGreaterEqual(song.energy_level, 1)
                self.assertLessEqual(song.energy_level, 10)
                self.assertIsInstance(song.therapeutic_benefits, list)
    
    def test_get_vocal_songs_for_emotion(self):
        """Test getting vocal songs for specific emotions"""
        # Test joy emotion
        joy_songs = self.db.get_vocal_songs_for_emotion("joy", 3)
        self.assertEqual(len(joy_songs), 3)
        
        # Should prioritize vocal songs
        vocal_count = sum(1 for song in joy_songs if song.has_vocals)
        self.assertGreaterEqual(vocal_count, 2, "Should prioritize vocal songs")
        
        # Test sadness emotion
        sad_songs = self.db.get_vocal_songs_for_emotion("sadness", 2)
        self.assertEqual(len(sad_songs), 2)
        
        for song in sad_songs:
            self.assertIsInstance(song, EnhancedSong)
    
    def test_emotion_mapping(self):
        """Test that emotion mapping works correctly"""
        # Test mapped emotions
        happiness_songs = self.db.get_vocal_songs_for_emotion("happiness", 2)
        joy_songs = self.db.get_vocal_songs_for_emotion("joy", 2)
        
        # Should get songs from joy category for happiness
        self.assertEqual(len(happiness_songs), 2)
        
        # Test depression -> sadness mapping
        depression_songs = self.db.get_vocal_songs_for_emotion("depression", 2)
        self.assertEqual(len(depression_songs), 2)
    
    def test_get_mixed_playlist(self):
        """Test getting mixed vocal/instrumental playlists"""
        # Test 70% vocal ratio
        mixed_playlist = self.db.get_mixed_playlist("joy", vocal_ratio=0.7, count=5)
        self.assertEqual(len(mixed_playlist), 5)
        
        vocal_count = sum(1 for song in mixed_playlist if song.has_vocals)
        # Should be approximately 70% vocal (3-4 out of 5)
        self.assertGreaterEqual(vocal_count, 3)
        
        # Test 100% vocal ratio
        all_vocal = self.db.get_mixed_playlist("joy", vocal_ratio=1.0, count=3)
        vocal_count = sum(1 for song in all_vocal if song.has_vocals)
        self.assertEqual(vocal_count, len(all_vocal))
    
    def test_content_filtering(self):
        """Test content rating filtering"""
        all_songs = self.db.get_songs_for_emotion("joy", 10)
        
        # Filter for clean content
        clean_songs = self.db.filter_content(all_songs, "clean")
        
        for song in clean_songs:
            self.assertEqual(song.content_rating, "clean")
        
        # Test "all" rating (no filtering)
        all_content = self.db.filter_content(all_songs, "all")
        self.assertEqual(len(all_content), len(all_songs))
    
    def test_get_songs_by_energy_level(self):
        """Test filtering songs by energy level"""
        # Get high energy songs (7-10)
        high_energy = self.db.get_songs_by_energy_level(7, 10, count=5)
        self.assertLessEqual(len(high_energy), 5)
        
        for song in high_energy:
            self.assertGreaterEqual(song.energy_level, 7)
            self.assertLessEqual(song.energy_level, 10)
        
        # Get low energy songs (1-3)
        low_energy = self.db.get_songs_by_energy_level(1, 3, count=3)
        
        for song in low_energy:
            self.assertGreaterEqual(song.energy_level, 1)
            self.assertLessEqual(song.energy_level, 3)
    
    def test_get_therapeutic_songs(self):
        """Test getting songs by therapeutic benefit"""
        # Test getting empowerment songs
        empowerment_songs = self.db.get_therapeutic_songs("empowerment", count=3)
        
        for song in empowerment_songs:
            benefits_lower = [b.lower() for b in song.therapeutic_benefits]
            self.assertIn("empowerment", benefits_lower)
        
        # Test getting calming songs
        calming_songs = self.db.get_therapeutic_songs("anxiety relief", count=2)
        
        for song in calming_songs:
            benefits_lower = [b.lower() for b in song.therapeutic_benefits]
            self.assertTrue(any("anxiety" in b for b in benefits_lower))
    
    def test_english_vocal_library(self):
        """Test that English vocal library is properly structured"""
        self.assertIsInstance(self.db.english_vocal_library, dict)
        
        # Check that it has therapeutic categories
        expected_categories = ["empowerment", "healing", "motivation"]
        for category in expected_categories:
            self.assertIn(category, self.db.english_vocal_library)
            
            songs = self.db.english_vocal_library[category]
            self.assertIsInstance(songs, list)
            self.assertGreater(len(songs), 0)
            
            for song in songs:
                self.assertIsInstance(song, EnhancedSong)
                self.assertTrue(song.has_vocals, f"Song in {category} should have vocals")
                self.assertEqual(song.language, "english")
    
    def test_song_metadata_quality(self):
        """Test that songs have proper therapeutic metadata"""
        for emotion, songs in self.db.emotion_playlists.items():
            for song in songs:
                # Check required fields
                self.assertIsInstance(song.title, str)
                self.assertGreater(len(song.title), 0)
                self.assertIsInstance(song.artist, str)
                self.assertGreater(len(song.artist), 0)
                
                # Check therapeutic metadata
                self.assertIsInstance(song.therapeutic_benefits, list)
                self.assertIsInstance(song.energy_level, int)
                self.assertIsInstance(song.emotional_intensity, int)
                
                # Check ranges
                self.assertGreaterEqual(song.energy_level, 1)
                self.assertLessEqual(song.energy_level, 10)
                self.assertGreaterEqual(song.emotional_intensity, 1)
                self.assertLessEqual(song.emotional_intensity, 10)
                self.assertGreaterEqual(song.popularity_score, 1)
                self.assertLessEqual(song.popularity_score, 100)

if __name__ == "__main__":
    unittest.main()