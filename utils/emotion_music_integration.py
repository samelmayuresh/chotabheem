# utils/emotion_music_integration.py - Integration between emotion detection and enhanced music system
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

from utils.enhanced_music_controller import EnhancedMusicController, MusicSession
from utils.enhanced_emotion_analyzer import EnhancedEmotionAnalyzer, EmotionResult

@dataclass
class EmotionMusicSession:
    """Combined emotion analysis and music therapy session"""
    session_id: str
    emotion_analysis: EmotionResult
    music_session: MusicSession
    integration_metadata: Dict[str, Any]
    created_at: datetime
    user_feedback: Optional[Dict[str, Any]] = None

class EmotionMusicIntegrator:
    """Integrates emotion detection with enhanced music playback system"""
    
    def __init__(self, 
                 music_controller: Optional[EnhancedMusicController] = None,
                 emotion_analyzer: Optional[EnhancedEmotionAnalyzer] = None):
        """
        Initialize emotion-music integrator
        
        Args:
            music_controller: Enhanced music controller instance
            emotion_analyzer: Enhanced emotion analyzer instance
        """
        
        # Initialize components
        self.music_controller = music_controller or EnhancedMusicController()
        self.emotion_analyzer = emotion_analyzer or EnhancedEmotionAnalyzer()
        
        # Session tracking
        self.active_sessions: Dict[str, EmotionMusicSession] = {}
        self.session_history: List[EmotionMusicSession] = []
        
        # Integration settings
        self.confidence_threshold = 0.6  # Minimum confidence for emotion-based recommendations
        self.secondary_emotion_threshold = 0.3  # Threshold for including secondary emotions
        self.dynamic_adjustment_enabled = True  # Enable real-time emotion adjustment
        
        logging.info("EmotionMusicIntegrator initialized")
    
    def analyze_and_create_session(self, 
                                 text: str = None,
                                 audio_data: Any = None,
                                 sample_rate: int = None,
                                 therapy_mode: str = "targeted",
                                 user_preferences: Dict[str, Any] = None,
                                 context: Dict[str, Any] = None) -> EmotionMusicSession:
        """
        Analyze emotions and create corresponding music therapy session
        
        Args:
            text: Text to analyze for emotions
            audio_data: Audio data for voice emotion analysis
            sample_rate: Sample rate for audio data
            therapy_mode: Music therapy mode ("targeted", "full_session", "custom")
            user_preferences: User preferences for music selection
            context: Additional context for analysis
            
        Returns:
            Combined emotion-music session
        """
        
        # Perform emotion analysis
        if text and audio_data is not None:
            # Multimodal analysis
            emotion_result = self.emotion_analyzer.analyze_multimodal(text, audio_data, sample_rate, context)
            analysis_type = "multimodal"
        elif text:
            # Text-only analysis
            emotion_result = self.emotion_analyzer.analyze_text(text, context)
            analysis_type = "text"
        elif audio_data is not None:
            # Audio-only analysis
            emotion_result = self.emotion_analyzer.analyze_audio(audio_data, sample_rate, context)
            analysis_type = "audio"
        else:
            raise ValueError("Either text or audio data must be provided")
        
        # Extract primary and secondary emotions
        primary_emotion, secondary_emotions = self._extract_emotions_from_result(emotion_result)
        
        # Determine song count based on therapy mode and emotion intensity
        song_count = self._calculate_optimal_song_count(emotion_result, therapy_mode)
        
        # Create music session based on detected emotions
        music_session = self.music_controller.create_continuous_session(
            emotion=primary_emotion,
            mode=therapy_mode,
            song_count=song_count,
            secondary_emotions=secondary_emotions,
            user_preferences=user_preferences
        )
        
        # Create combined session
        session_id = f"emotion_music_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        integration_metadata = {
            "analysis_type": analysis_type,
            "confidence_level": emotion_result.confidence_level,
            "emotion_count": len(emotion_result.all_emotions),
            "therapy_mode": therapy_mode,
            "song_count": song_count,
            "user_preferences_applied": user_preferences is not None,
            "context_provided": context is not None
        }
        
        combined_session = EmotionMusicSession(
            session_id=session_id,
            emotion_analysis=emotion_result,
            music_session=music_session,
            integration_metadata=integration_metadata,
            created_at=datetime.now()
        )
        
        # Store active session
        self.active_sessions[session_id] = combined_session
        
        logging.info(f"Created emotion-music session for {primary_emotion} with {len(secondary_emotions)} secondary emotions")
        
        return combined_session
    
    def get_emotion_based_recommendations(self, 
                                        emotion_result: EmotionResult,
                                        limit: int = 5) -> Dict[str, Any]:
        """
        Get music recommendations based on emotion analysis results
        
        Args:
            emotion_result: Result from emotion analysis
            limit: Maximum number of recommendations
            
        Returns:
            Dictionary with various recommendation categories
        """
        
        primary_emotion, secondary_emotions = self._extract_emotions_from_result(emotion_result)
        
        recommendations = {
            "primary_emotion_playlists": [],
            "secondary_emotion_songs": [],
            "therapeutic_suggestions": [],
            "energy_matched_songs": [],
            "confidence_based_recommendations": []
        }
        
        # Primary emotion recommendations
        primary_recs = self.music_controller.get_recommendations(primary_emotion, limit)
        recommendations["primary_emotion_playlists"] = primary_recs.get("popular_playlists", [])
        
        # Secondary emotion songs
        for emotion in secondary_emotions[:2]:  # Limit to top 2 secondary emotions
            songs = self.music_controller.song_database.get_vocal_songs_for_emotion(emotion, 3)
            recommendations["secondary_emotion_songs"].append({
                "emotion": emotion,
                "songs": [{"title": s.title, "artist": s.artist, "mood": s.mood} for s in songs]
            })
        
        # Therapeutic suggestions based on confidence
        if emotion_result.confidence_level == "low":
            # Low confidence - suggest neutral/balancing music
            neutral_songs = self.music_controller.song_database.get_songs_for_emotion("neutral", 3)
            recommendations["therapeutic_suggestions"].append({
                "reason": "low_confidence",
                "suggestion": "Neutral/balancing music recommended due to uncertain emotion detection",
                "songs": [{"title": s.title, "artist": s.artist} for s in neutral_songs]
            })
        
        # Energy level matching
        if hasattr(emotion_result, 'intensity') and emotion_result.intensity:
            energy_level = self._map_intensity_to_energy(emotion_result.intensity)
            energy_songs = self.music_controller.song_database.get_songs_by_energy_level(
                energy_level - 1, energy_level + 1, 3
            )
            recommendations["energy_matched_songs"] = [
                {"title": s.title, "artist": s.artist, "energy_level": s.energy_level} 
                for s in energy_songs
            ]
        
        # Confidence-based recommendations
        if emotion_result.confidence_level == "high":
            recommendations["confidence_based_recommendations"].append({
                "type": "high_confidence",
                "message": "High confidence detection - targeted therapy recommended",
                "suggested_mode": "targeted"
            })
        elif emotion_result.confidence_level == "medium":
            recommendations["confidence_based_recommendations"].append({
                "type": "medium_confidence", 
                "message": "Medium confidence - full session therapy recommended",
                "suggested_mode": "full_session"
            })
        else:
            recommendations["confidence_based_recommendations"].append({
                "type": "low_confidence",
                "message": "Low confidence - custom playlist with user input recommended",
                "suggested_mode": "custom"
            })
        
        return recommendations
    
    def adjust_session_dynamically(self, 
                                 session_id: str,
                                 new_emotion_result: EmotionResult) -> bool:
        """
        Dynamically adjust active music session based on new emotion analysis
        
        Args:
            session_id: ID of active session to adjust
            new_emotion_result: New emotion analysis result
            
        Returns:
            True if adjustment was successful
        """
        
        if not self.dynamic_adjustment_enabled:
            return False
        
        if session_id not in self.active_sessions:
            logging.error(f"Session {session_id} not found for dynamic adjustment")
            return False
        
        session = self.active_sessions[session_id]
        
        # Extract new emotions
        new_primary, new_secondary = self._extract_emotions_from_result(new_emotion_result)
        
        # Check if significant emotion change occurred
        old_primary = session.emotion_analysis.primary_emotion.label
        
        if new_primary != old_primary and new_emotion_result.confidence_level in ["high", "medium"]:
            logging.info(f"Significant emotion change detected: {old_primary} -> {new_primary}")
            
            # Create new playlist for the new emotion
            new_songs = self.music_controller.get_enhanced_english_playlist(
                emotion=new_primary,
                count=5,
                vocal_ratio=0.8
            )
            
            # Add new songs to current playlist
            current_playlist = session.music_session.playlist
            for song in new_songs[:3]:  # Add 3 new songs
                self.music_controller.playlist_manager.modify_playlist(
                    current_playlist.id, "add_song", song=song
                )
            
            # Update session metadata
            session.integration_metadata["dynamic_adjustments"] = session.integration_metadata.get("dynamic_adjustments", 0) + 1
            session.integration_metadata["last_adjustment"] = datetime.now().isoformat()
            session.integration_metadata["adjustment_reason"] = f"emotion_change_{old_primary}_to_{new_primary}"
            
            return True
        
        return False
    
    def provide_user_feedback(self, 
                            session_id: str,
                            feedback: Dict[str, Any]) -> bool:
        """
        Collect user feedback on emotion-music session
        
        Args:
            session_id: Session ID to provide feedback for
            feedback: User feedback dictionary
            
        Returns:
            True if feedback was recorded successfully
        """
        
        if session_id not in self.active_sessions:
            logging.error(f"Session {session_id} not found for feedback")
            return False
        
        session = self.active_sessions[session_id]
        session.user_feedback = feedback
        session.user_feedback["timestamp"] = datetime.now().isoformat()
        
        # Use feedback to improve future recommendations
        self._learn_from_feedback(session, feedback)
        
        logging.info(f"Recorded user feedback for session {session_id}")
        return True
    
    def get_session_analytics(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive analytics for an emotion-music session
        
        Args:
            session_id: Session ID to analyze
            
        Returns:
            Analytics dictionary or None if session not found
        """
        
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        
        # Get music session info
        music_info = self.music_controller.get_session_info()
        
        # Get playback statistics
        playback_stats = self.music_controller.player_engine.get_session_statistics()
        
        analytics = {
            "session_overview": {
                "session_id": session_id,
                "created_at": session.created_at.isoformat(),
                "analysis_type": session.integration_metadata["analysis_type"],
                "therapy_mode": session.integration_metadata["therapy_mode"]
            },
            "emotion_analysis": {
                "primary_emotion": session.emotion_analysis.primary_emotion.label,
                "confidence_level": session.emotion_analysis.confidence_level,
                "total_emotions_detected": len(session.emotion_analysis.all_emotions),
                "secondary_emotions": [e.label for e in session.emotion_analysis.all_emotions[1:3]]
            },
            "music_session": {
                "playlist_name": session.music_session.playlist.name,
                "total_songs": len(session.music_session.playlist.songs),
                "vocal_ratio": session.music_session.playlist.vocal_instrumental_ratio,
                "total_duration": session.music_session.playlist.get_duration_formatted()
            },
            "playback_performance": playback_stats,
            "integration_metadata": session.integration_metadata,
            "user_feedback": session.user_feedback
        }
        
        return analytics
    
    def complete_session(self, session_id: str) -> bool:
        """
        Complete and archive an emotion-music session
        
        Args:
            session_id: Session ID to complete
            
        Returns:
            True if session was completed successfully
        """
        
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        
        # Save music session
        self.music_controller.save_current_session()
        
        # Move to history
        self.session_history.append(session)
        del self.active_sessions[session_id]
        
        logging.info(f"Completed emotion-music session {session_id}")
        return True
    
    def _extract_emotions_from_result(self, emotion_result: EmotionResult) -> Tuple[str, List[str]]:
        """Extract primary and secondary emotions from analysis result"""
        
        if not emotion_result.all_emotions:
            return "neutral", []
        
        # Primary emotion (highest confidence)
        primary_emotion = emotion_result.primary_emotion.label
        
        # Secondary emotions (above threshold)
        secondary_emotions = [
            emotion.label for emotion in emotion_result.all_emotions[1:]
            if emotion.confidence > self.secondary_emotion_threshold
        ]
        
        return primary_emotion, secondary_emotions[:3]  # Limit to top 3 secondary emotions
    
    def _calculate_optimal_song_count(self, emotion_result: EmotionResult, therapy_mode: str) -> int:
        """Calculate optimal number of songs based on emotion analysis and therapy mode"""
        
        base_counts = {
            "targeted": 5,
            "full_session": 10,
            "custom": 7
        }
        
        base_count = base_counts.get(therapy_mode, 5)
        
        # Adjust based on confidence level
        if emotion_result.confidence_level == "high":
            # High confidence - can use targeted approach
            return base_count
        elif emotion_result.confidence_level == "medium":
            # Medium confidence - add variety
            return base_count + 2
        else:
            # Low confidence - more variety needed
            return base_count + 3
    
    def _map_intensity_to_energy(self, intensity: float) -> int:
        """Map emotion intensity to energy level (1-10 scale)"""
        
        # Intensity is typically 0-1, map to 1-10 energy scale
        return max(1, min(10, int(intensity * 10)))
    
    def _learn_from_feedback(self, session: EmotionMusicSession, feedback: Dict[str, Any]):
        """Learn from user feedback to improve future recommendations"""
        
        # This is a placeholder for machine learning integration
        # In a full implementation, this would update recommendation models
        
        if feedback.get("rating", 0) >= 4:  # Good rating (4-5 stars)
            # Positive feedback - reinforce this emotion-music mapping
            logging.info(f"Positive feedback for {session.emotion_analysis.emotions[0]['label']} -> {session.music_session.therapy_mode}")
        
        elif feedback.get("rating", 0) <= 2:  # Poor rating (1-2 stars)
            # Negative feedback - avoid this mapping in future
            logging.info(f"Negative feedback for {session.emotion_analysis.emotions[0]['label']} -> {session.music_session.therapy_mode}")
        
        # Store feedback patterns for future analysis
        # This could be expanded to update user preference models
    
    def get_integration_statistics(self) -> Dict[str, Any]:
        """Get overall integration statistics"""
        
        total_sessions = len(self.active_sessions) + len(self.session_history)
        
        if total_sessions == 0:
            return {"message": "No sessions recorded yet"}
        
        # Analyze session history
        all_sessions = list(self.active_sessions.values()) + self.session_history
        
        emotion_distribution = {}
        therapy_mode_distribution = {}
        analysis_type_distribution = {}
        
        for session in all_sessions:
            # Emotion distribution
            primary_emotion = session.emotion_analysis.primary_emotion.label
            emotion_distribution[primary_emotion] = emotion_distribution.get(primary_emotion, 0) + 1
            
            # Therapy mode distribution
            therapy_mode = session.integration_metadata["therapy_mode"]
            therapy_mode_distribution[therapy_mode] = therapy_mode_distribution.get(therapy_mode, 0) + 1
            
            # Analysis type distribution
            analysis_type = session.integration_metadata["analysis_type"]
            analysis_type_distribution[analysis_type] = analysis_type_distribution.get(analysis_type, 0) + 1
        
        return {
            "total_sessions": total_sessions,
            "active_sessions": len(self.active_sessions),
            "completed_sessions": len(self.session_history),
            "emotion_distribution": emotion_distribution,
            "therapy_mode_distribution": therapy_mode_distribution,
            "analysis_type_distribution": analysis_type_distribution,
            "average_confidence_level": [s.emotion_analysis.confidence_level for s in all_sessions]
        }
    
    def cleanup(self):
        """Clean up resources"""
        
        # Complete all active sessions
        for session_id in list(self.active_sessions.keys()):
            self.complete_session(session_id)
        
        # Cleanup music controller
        self.music_controller.cleanup()
        
        logging.info("EmotionMusicIntegrator cleaned up")