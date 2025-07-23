# test_emotion_music_integration.py - Tests for emotion-music integration
import unittest
import tempfile
import os
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime

from utils.emotion_music_integration import EmotionMusicIntegrator, EmotionMusicSession
from utils.enhanced_music_controller import EnhancedMusicController
from utils.enhanced_emotion_analyzer import EnhancedEmotionAnalyzer, EmotionResult

class TestEmotionMusicIntegration(unittest.TestCase):
    """Test cases for emotion-music integration"""
    
    def setUp(self):
        """Set up test environment"""
        
        # Create temporary file for music controller
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        self.temp_file.close()
        
        # Create integrator
        self.integrator = EmotionMusicIntegrator()
        
        # Mock emotion result for testing
        self.mock_emotion_result = EmotionResult(
            emotions=[
                {"label": "joy", "score": 0.8},
                {"label": "excitement", "score": 0.4},
                {"label": "optimism", "score": 0.3}
            ],
            confidence=0.8,
            analysis_type="text",
            processing_time=0.1,
            model_info={"model": "test"},
            raw_output={"test": "data"}
        )
    
    def tearDown(self):
        """Clean up test environment"""
        self.integrator.cleanup()
        try:
            os.unlink(self.temp_file.name)
        except:
            pass
    
    def test_integrator_initialization(self):
        """Test integrator initialization"""
        
        self.assertIsNotNone(self.integrator.music_controller)
        self.assertIsNotNone(self.integrator.emotion_analyzer)
        self.assertEqual(len(self.integrator.active_sessions), 0)
        self.assertEqual(len(self.integrator.session_history), 0)
        
        # Check default settings
        self.assertEqual(self.integrator.confidence_threshold, 0.6)
        self.assertEqual(self.integrator.secondary_emotion_threshold, 0.3)
        self.assertTrue(self.integrator.dynamic_adjustment_enabled)
    
    @patch.object(EnhancedEmotionAnalyzer, 'analyze_text')
    def test_analyze_and_create_session_text(self, mock_analyze):
        """Test creating session from text analysis"""
        
        # Mock emotion analysis
        mock_analyze.return_value = self.mock_emotion_result
        
        # Create session
        session = self.integrator.analyze_and_create_session(
            text="I'm feeling really happy today!",
            therapy_mode="targeted"
        )
        
        # Verify session creation
        self.assertIsInstance(session, EmotionMusicSession)
        self.assertEqual(session.emotion_analysis, self.mock_emotion_result)
        self.assertEqual(session.music_session.emotion_focus, "joy")
        self.assertEqual(session.integration_metadata["analysis_type"], "text")
        self.assertEqual(session.integration_metadata["therapy_mode"], "targeted")
        
        # Verify session is stored
        self.assertIn(session.session_id, self.integrator.active_sessions)
        
        # Verify analyzer was called correctly
        mock_analyze.assert_called_once_with("I'm feeling really happy today!", None)
    
    @patch.object(EnhancedEmotionAnalyzer, 'analyze_audio')
    def test_analyze_and_create_session_audio(self, mock_analyze):
        """Test creating session from audio analysis"""
        
        # Mock emotion analysis
        mock_analyze.return_value = self.mock_emotion_result
        
        # Create mock audio data
        audio_data = np.random.randn(16000)  # 1 second of audio at 16kHz
        
        # Create session
        session = self.integrator.analyze_and_create_session(
            audio_data=audio_data,
            sample_rate=16000,
            therapy_mode="full_session"
        )
        
        # Verify session creation
        self.assertIsInstance(session, EmotionMusicSession)
        self.assertEqual(session.integration_metadata["analysis_type"], "audio")
        self.assertEqual(session.integration_metadata["therapy_mode"], "full_session")
        
        # Verify analyzer was called correctly
        mock_analyze.assert_called_once()
    
    @patch.object(EnhancedEmotionAnalyzer, 'analyze_multimodal')
    def test_analyze_and_create_session_multimodal(self, mock_analyze):
        """Test creating session from multimodal analysis"""
        
        # Mock emotion analysis
        mock_analyze.return_value = self.mock_emotion_result
        
        # Create mock audio data
        audio_data = np.random.randn(16000)
        
        # Create session
        session = self.integrator.analyze_and_create_session(
            text="I'm excited about this!",
            audio_data=audio_data,
            sample_rate=16000,
            therapy_mode="custom"
        )
        
        # Verify session creation
        self.assertEqual(session.integration_metadata["analysis_type"], "multimodal")
        self.assertEqual(session.integration_metadata["therapy_mode"], "custom")
        
        # Verify analyzer was called correctly
        mock_analyze.assert_called_once()
    
    def test_analyze_and_create_session_no_input(self):
        """Test error handling when no input is provided"""
        
        with self.assertRaises(ValueError):
            self.integrator.analyze_and_create_session()
    
    def test_extract_emotions_from_result(self):
        """Test emotion extraction from analysis result"""
        
        primary, secondary = self.integrator._extract_emotions_from_result(self.mock_emotion_result)
        
        self.assertEqual(primary, "joy")
        self.assertIn("excitement", secondary)
        self.assertNotIn("optimism", secondary)  # Below threshold (0.3 == threshold)
    
    def test_calculate_optimal_song_count(self):
        """Test optimal song count calculation"""
        
        # High confidence
        count = self.integrator._calculate_optimal_song_count(self.mock_emotion_result, "targeted")
        self.assertEqual(count, 5)  # Base count for targeted mode
        
        # Medium confidence
        medium_confidence_result = EmotionResult(
            emotions=[{"label": "sadness", "score": 0.7}],
            confidence=0.7,
            analysis_type="text",
            processing_time=0.1,
            model_info={"model": "test"},
            raw_output={}
        )
        
        count = self.integrator._calculate_optimal_song_count(medium_confidence_result, "targeted")
        self.assertEqual(count, 7)  # Base + 2 for medium confidence
        
        # Low confidence
        low_confidence_result = EmotionResult(
            emotions=[{"label": "neutral", "score": 0.5}],
            confidence=0.5,
            analysis_type="text",
            processing_time=0.1,
            model_info={"model": "test"},
            raw_output={}
        )
        
        count = self.integrator._calculate_optimal_song_count(low_confidence_result, "targeted")
        self.assertEqual(count, 8)  # Base + 3 for low confidence
    
    def test_get_emotion_based_recommendations(self):
        """Test emotion-based recommendation generation"""
        
        recommendations = self.integrator.get_emotion_based_recommendations(self.mock_emotion_result)
        
        # Check recommendation structure
        self.assertIn("primary_emotion_playlists", recommendations)
        self.assertIn("secondary_emotion_songs", recommendations)
        self.assertIn("therapeutic_suggestions", recommendations)
        self.assertIn("energy_matched_songs", recommendations)
        self.assertIn("confidence_based_recommendations", recommendations)
        
        # Check confidence-based recommendations
        confidence_recs = recommendations["confidence_based_recommendations"]
        self.assertGreater(len(confidence_recs), 0)
        self.assertEqual(confidence_recs[0]["type"], "high_confidence")
        self.assertEqual(confidence_recs[0]["suggested_mode"], "targeted")
    
    @patch.object(EnhancedEmotionAnalyzer, 'analyze_text')
    def test_dynamic_session_adjustment(self, mock_analyze):
        """Test dynamic session adjustment based on new emotion analysis"""
        
        # Create initial session
        mock_analyze.return_value = self.mock_emotion_result
        session = self.integrator.analyze_and_create_session(
            text="I'm happy!",
            therapy_mode="targeted"
        )
        
        # Create new emotion result (different emotion)
        new_emotion_result = EmotionResult(
            emotions=[{"label": "sadness", "score": 0.9}],
            confidence=0.9,
            analysis_type="text",
            processing_time=0.1,
            model_info={"model": "test"},
            raw_output={}
        )
        
        # Test dynamic adjustment
        success = self.integrator.adjust_session_dynamically(session.session_id, new_emotion_result)
        self.assertTrue(success)
        
        # Check adjustment metadata
        updated_session = self.integrator.active_sessions[session.session_id]
        self.assertEqual(updated_session.integration_metadata["dynamic_adjustments"], 1)
        self.assertIn("last_adjustment", updated_session.integration_metadata)
    
    def test_dynamic_adjustment_disabled(self):
        """Test dynamic adjustment when disabled"""
        
        # Disable dynamic adjustment
        self.integrator.dynamic_adjustment_enabled = False
        
        # Try to adjust (should fail)
        success = self.integrator.adjust_session_dynamically("fake-id", self.mock_emotion_result)
        self.assertFalse(success)
    
    def test_dynamic_adjustment_nonexistent_session(self):
        """Test dynamic adjustment for non-existent session"""
        
        success = self.integrator.adjust_session_dynamically("nonexistent-id", self.mock_emotion_result)
        self.assertFalse(success)
    
    @patch.object(EnhancedEmotionAnalyzer, 'analyze_text')
    def test_user_feedback(self, mock_analyze):
        """Test user feedback collection"""
        
        # Create session
        mock_analyze.return_value = self.mock_emotion_result
        session = self.integrator.analyze_and_create_session(
            text="Test text",
            therapy_mode="targeted"
        )
        
        # Provide feedback
        feedback = {
            "rating": 5,
            "liked_songs": ["Song 1", "Song 2"],
            "comments": "Great playlist!"
        }
        
        success = self.integrator.provide_user_feedback(session.session_id, feedback)
        self.assertTrue(success)
        
        # Check feedback was stored
        updated_session = self.integrator.active_sessions[session.session_id]
        self.assertEqual(updated_session.user_feedback["rating"], 5)
        self.assertIn("timestamp", updated_session.user_feedback)
    
    def test_user_feedback_nonexistent_session(self):
        """Test user feedback for non-existent session"""
        
        success = self.integrator.provide_user_feedback("nonexistent-id", {"rating": 5})
        self.assertFalse(success)
    
    @patch.object(EnhancedEmotionAnalyzer, 'analyze_text')
    def test_session_analytics(self, mock_analyze):
        """Test session analytics generation"""
        
        # Create session
        mock_analyze.return_value = self.mock_emotion_result
        session = self.integrator.analyze_and_create_session(
            text="Test text",
            therapy_mode="full_session"
        )
        
        # Get analytics
        analytics = self.integrator.get_session_analytics(session.session_id)
        
        # Check analytics structure
        self.assertIsNotNone(analytics)
        self.assertIn("session_overview", analytics)
        self.assertIn("emotion_analysis", analytics)
        self.assertIn("music_session", analytics)
        self.assertIn("playback_performance", analytics)
        self.assertIn("integration_metadata", analytics)
        
        # Check specific values
        self.assertEqual(analytics["emotion_analysis"]["primary_emotion"], "joy")
        self.assertEqual(analytics["emotion_analysis"]["confidence"], 0.8)
        self.assertEqual(analytics["session_overview"]["therapy_mode"], "full_session")
    
    def test_session_analytics_nonexistent(self):
        """Test analytics for non-existent session"""
        
        analytics = self.integrator.get_session_analytics("nonexistent-id")
        self.assertIsNone(analytics)
    
    @patch.object(EnhancedEmotionAnalyzer, 'analyze_text')
    def test_complete_session(self, mock_analyze):
        """Test session completion"""
        
        # Create session
        mock_analyze.return_value = self.mock_emotion_result
        session = self.integrator.analyze_and_create_session(
            text="Test text",
            therapy_mode="targeted"
        )
        
        session_id = session.session_id
        
        # Complete session
        success = self.integrator.complete_session(session_id)
        self.assertTrue(success)
        
        # Check session moved to history
        self.assertNotIn(session_id, self.integrator.active_sessions)
        self.assertIn(session, self.integrator.session_history)
    
    def test_complete_nonexistent_session(self):
        """Test completing non-existent session"""
        
        success = self.integrator.complete_session("nonexistent-id")
        self.assertFalse(success)
    
    def test_map_intensity_to_energy(self):
        """Test intensity to energy level mapping"""
        
        # Test various intensity values
        self.assertEqual(self.integrator._map_intensity_to_energy(0.0), 1)
        self.assertEqual(self.integrator._map_intensity_to_energy(0.5), 5)
        self.assertEqual(self.integrator._map_intensity_to_energy(1.0), 10)
        self.assertEqual(self.integrator._map_intensity_to_energy(1.5), 10)  # Clamped to max
    
    @patch.object(EnhancedEmotionAnalyzer, 'analyze_text')
    def test_integration_statistics(self, mock_analyze):
        """Test integration statistics generation"""
        
        # Initially no sessions
        stats = self.integrator.get_integration_statistics()
        self.assertIn("message", stats)
        
        # Create some sessions
        mock_analyze.return_value = self.mock_emotion_result
        
        session1 = self.integrator.analyze_and_create_session(
            text="Happy text", therapy_mode="targeted"
        )
        session2 = self.integrator.analyze_and_create_session(
            text="Another happy text", therapy_mode="full_session"
        )
        
        # Complete one session
        self.integrator.complete_session(session1.session_id)
        
        # Get statistics
        stats = self.integrator.get_integration_statistics()
        
        # Check statistics
        self.assertEqual(stats["total_sessions"], 2)
        self.assertEqual(stats["active_sessions"], 1)
        self.assertEqual(stats["completed_sessions"], 1)
        self.assertIn("joy", stats["emotion_distribution"])
        self.assertIn("targeted", stats["therapy_mode_distribution"])
        self.assertIn("text", stats["analysis_type_distribution"])
        self.assertEqual(stats["average_confidence"], 0.8)
    
    def test_cleanup(self):
        """Test integrator cleanup"""
        
        # Create a session
        with patch.object(EnhancedEmotionAnalyzer, 'analyze_text') as mock_analyze:
            mock_analyze.return_value = self.mock_emotion_result
            session = self.integrator.analyze_and_create_session(
                text="Test", therapy_mode="targeted"
            )
        
        # Cleanup should complete active sessions
        self.integrator.cleanup()
        
        # Check session was moved to history
        self.assertEqual(len(self.integrator.active_sessions), 0)
        self.assertIn(session, self.integrator.session_history)

if __name__ == "__main__":
    unittest.main()