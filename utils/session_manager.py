# utils/session_manager.py - User session management for music therapy
import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging

@dataclass
class UserSession:
    """User session data for music therapy"""
    user_id: str
    session_start: datetime
    last_activity: datetime
    preferences: Dict[str, Any]
    playlist_history: List[str]  # Playlist IDs
    listening_stats: Dict[str, Any]
    feedback_history: List[Dict[str, Any]]
    
    def is_expired(self, timeout_hours: int = 24) -> bool:
        """Check if session is expired"""
        return datetime.now() - self.last_activity > timedelta(hours=timeout_hours)
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now()

class SessionManager:
    """Manages user sessions and playlist persistence"""
    
    def __init__(self, storage_path: str = "user_sessions.json"):
        self.storage_path = storage_path
        self.active_sessions: Dict[str, UserSession] = {}
        self.load_sessions()
    
    def create_session(self, user_id: str, preferences: Dict[str, Any] = None) -> UserSession:
        """Create new user session"""
        
        session = UserSession(
            user_id=user_id,
            session_start=datetime.now(),
            last_activity=datetime.now(),
            preferences=preferences or {},
            playlist_history=[],
            listening_stats={
                "total_sessions": 0,
                "total_listening_time": 0,
                "favorite_emotions": {},
                "preferred_therapy_mode": "targeted"
            },
            feedback_history=[]
        )
        
        self.active_sessions[user_id] = session
        self.save_sessions()
        
        logging.info(f"Created session for user {user_id}")
        return session
    
    def get_session(self, user_id: str) -> Optional[UserSession]:
        """Get user session, create if doesn't exist"""
        
        if user_id in self.active_sessions:
            session = self.active_sessions[user_id]
            if not session.is_expired():
                session.update_activity()
                return session
            else:
                # Session expired, remove it
                del self.active_sessions[user_id]
        
        # Create new session
        return self.create_session(user_id)
    
    def update_session_preferences(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        """Update user preferences"""
        
        session = self.get_session(user_id)
        if session:
            session.preferences.update(preferences)
            session.update_activity()
            self.save_sessions()
            return True
        return False
    
    def add_playlist_to_history(self, user_id: str, playlist_id: str) -> bool:
        """Add playlist to user's history"""
        
        session = self.get_session(user_id)
        if session:
            if playlist_id not in session.playlist_history:
                session.playlist_history.append(playlist_id)
                # Keep only last 50 playlists
                session.playlist_history = session.playlist_history[-50:]
            session.update_activity()
            self.save_sessions()
            return True
        return False
    
    def update_listening_stats(self, user_id: str, stats_update: Dict[str, Any]) -> bool:
        """Update user's listening statistics"""
        
        session = self.get_session(user_id)
        if session:
            # Update total sessions
            session.listening_stats["total_sessions"] += 1
            
            # Update listening time
            if "session_duration" in stats_update:
                session.listening_stats["total_listening_time"] += stats_update["session_duration"]
            
            # Update favorite emotions
            if "emotion" in stats_update:
                emotion = stats_update["emotion"]
                session.listening_stats["favorite_emotions"][emotion] = \
                    session.listening_stats["favorite_emotions"].get(emotion, 0) + 1
            
            # Update preferred therapy mode
            if "therapy_mode" in stats_update:
                session.listening_stats["preferred_therapy_mode"] = stats_update["therapy_mode"]
            
            session.update_activity()
            self.save_sessions()
            return True
        return False
    
    def add_feedback(self, user_id: str, feedback: Dict[str, Any]) -> bool:
        """Add user feedback to session"""
        
        session = self.get_session(user_id)
        if session:
            feedback["timestamp"] = datetime.now().isoformat()
            session.feedback_history.append(feedback)
            # Keep only last 100 feedback entries
            session.feedback_history = session.feedback_history[-100:]
            session.update_activity()
            self.save_sessions()
            return True
        return False
    
    def get_user_recommendations(self, user_id: str) -> Dict[str, Any]:
        """Get personalized recommendations based on user history"""
        
        session = self.get_session(user_id)
        if not session:
            return {}
        
        recommendations = {
            "preferred_therapy_mode": session.listening_stats.get("preferred_therapy_mode", "targeted"),
            "favorite_emotions": [],
            "suggested_preferences": {},
            "listening_insights": []
        }
        
        # Favorite emotions (top 3)
        favorite_emotions = session.listening_stats.get("favorite_emotions", {})
        if favorite_emotions:
            sorted_emotions = sorted(favorite_emotions.items(), key=lambda x: x[1], reverse=True)
            recommendations["favorite_emotions"] = [emotion for emotion, count in sorted_emotions[:3]]
        
        # Suggested preferences based on history
        total_sessions = session.listening_stats.get("total_sessions", 0)
        if total_sessions > 5:
            recommendations["suggested_preferences"] = {
                "vocal_ratio": 0.8,  # High vocal ratio for experienced users
                "auto_advance": True,
                "session_length": "medium"
            }
        
        # Listening insights
        total_time = session.listening_stats.get("total_listening_time", 0)
        if total_time > 3600:  # More than 1 hour
            recommendations["listening_insights"].append(
                f"You've listened to {total_time//3600} hours of therapeutic music!"
            )
        
        if len(session.playlist_history) > 10:
            recommendations["listening_insights"].append(
                f"You've explored {len(session.playlist_history)} different playlists"
            )
        
        return recommendations
    
    def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions"""
        
        expired_users = []
        for user_id, session in self.active_sessions.items():
            if session.is_expired():
                expired_users.append(user_id)
        
        for user_id in expired_users:
            del self.active_sessions[user_id]
        
        if expired_users:
            self.save_sessions()
            logging.info(f"Cleaned up {len(expired_users)} expired sessions")
        
        return len(expired_users)
    
    def get_session_analytics(self) -> Dict[str, Any]:
        """Get analytics across all sessions"""
        
        if not self.active_sessions:
            return {"message": "No active sessions"}
        
        total_users = len(self.active_sessions)
        total_sessions = sum(s.listening_stats.get("total_sessions", 0) for s in self.active_sessions.values())
        total_listening_time = sum(s.listening_stats.get("total_listening_time", 0) for s in self.active_sessions.values())
        
        # Aggregate favorite emotions
        all_emotions = {}
        for session in self.active_sessions.values():
            for emotion, count in session.listening_stats.get("favorite_emotions", {}).items():
                all_emotions[emotion] = all_emotions.get(emotion, 0) + count
        
        # Most popular therapy modes
        therapy_modes = {}
        for session in self.active_sessions.values():
            mode = session.listening_stats.get("preferred_therapy_mode", "targeted")
            therapy_modes[mode] = therapy_modes.get(mode, 0) + 1
        
        return {
            "total_users": total_users,
            "total_sessions": total_sessions,
            "total_listening_time_hours": total_listening_time / 3600,
            "average_sessions_per_user": total_sessions / total_users if total_users > 0 else 0,
            "popular_emotions": sorted(all_emotions.items(), key=lambda x: x[1], reverse=True)[:5],
            "popular_therapy_modes": therapy_modes,
            "active_users_last_24h": sum(1 for s in self.active_sessions.values() if not s.is_expired(24))
        }
    
    def export_user_data(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Export user data for backup/transfer"""
        
        session = self.get_session(user_id)
        if session:
            return {
                "user_id": user_id,
                "export_date": datetime.now().isoformat(),
                "session_data": asdict(session)
            }
        return None
    
    def import_user_data(self, user_data: Dict[str, Any]) -> bool:
        """Import user data from backup"""
        
        try:
            user_id = user_data["user_id"]
            session_data = user_data["session_data"]
            
            # Convert datetime strings back to datetime objects
            session_data["session_start"] = datetime.fromisoformat(session_data["session_start"])
            session_data["last_activity"] = datetime.fromisoformat(session_data["last_activity"])
            
            session = UserSession(**session_data)
            self.active_sessions[user_id] = session
            self.save_sessions()
            
            logging.info(f"Imported user data for {user_id}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to import user data: {e}")
            return False
    
    def save_sessions(self):
        """Save sessions to file"""
        
        try:
            # Convert sessions to serializable format
            sessions_data = {}
            for user_id, session in self.active_sessions.items():
                session_dict = asdict(session)
                session_dict["session_start"] = session.session_start.isoformat()
                session_dict["last_activity"] = session.last_activity.isoformat()
                sessions_data[user_id] = session_dict
            
            with open(self.storage_path, 'w') as f:
                json.dump(sessions_data, f, indent=2)
                
        except Exception as e:
            logging.error(f"Failed to save sessions: {e}")
    
    def load_sessions(self):
        """Load sessions from file"""
        
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    sessions_data = json.load(f)
                
                for user_id, session_dict in sessions_data.items():
                    # Convert datetime strings back to datetime objects
                    session_dict["session_start"] = datetime.fromisoformat(session_dict["session_start"])
                    session_dict["last_activity"] = datetime.fromisoformat(session_dict["last_activity"])
                    
                    session = UserSession(**session_dict)
                    self.active_sessions[user_id] = session
                
                logging.info(f"Loaded {len(self.active_sessions)} user sessions")
                
        except Exception as e:
            logging.error(f"Failed to load sessions: {e}")
            self.active_sessions = {}
    
    def cleanup(self):
        """Cleanup and save sessions"""
        self.cleanup_expired_sessions()
        self.save_sessions()
        logging.info("SessionManager cleaned up")