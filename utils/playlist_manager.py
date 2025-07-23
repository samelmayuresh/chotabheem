# utils/playlist_manager.py - Playlist management system for therapeutic music sessions
import uuid
import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from utils.enhanced_song_database import EnhancedSong
import logging

@dataclass
class TherapeuticPlaylist:
    """Therapeutic playlist with metadata and user customization support"""
    id: str
    name: str
    songs: List[EnhancedSong]
    emotion_focus: str
    secondary_emotions: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    therapeutic_notes: str = ""
    total_duration: int = 0  # in seconds
    vocal_instrumental_ratio: float = 0.7  # 70% vocal by default
    user_customized: bool = False
    play_count: int = 0
    user_rating: Optional[int] = None  # 1-5 stars
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Calculate total duration and other derived properties"""
        if self.songs:
            self.total_duration = sum(song.duration for song in self.songs)
            
            # Calculate actual vocal/instrumental ratio
            vocal_count = sum(1 for song in self.songs if song.has_vocals)
            if len(self.songs) > 0:
                self.vocal_instrumental_ratio = vocal_count / len(self.songs)
    
    def get_duration_formatted(self) -> str:
        """Get formatted duration string (MM:SS or HH:MM:SS)"""
        hours = self.total_duration // 3600
        minutes = (self.total_duration % 3600) // 60
        seconds = self.total_duration % 60
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes}:{seconds:02d}"
    
    def get_therapeutic_summary(self) -> Dict[str, Any]:
        """Get summary of therapeutic properties"""
        if not self.songs:
            return {}
        
        # Aggregate therapeutic benefits
        all_benefits = []
        for song in self.songs:
            all_benefits.extend(song.therapeutic_benefits)
        
        benefit_counts = {}
        for benefit in all_benefits:
            benefit_counts[benefit] = benefit_counts.get(benefit, 0) + 1
        
        # Calculate average energy and emotional intensity
        avg_energy = sum(song.energy_level for song in self.songs) / len(self.songs)
        avg_intensity = sum(song.emotional_intensity for song in self.songs) / len(self.songs)
        
        return {
            "primary_benefits": sorted(benefit_counts.items(), key=lambda x: x[1], reverse=True)[:3],
            "average_energy_level": round(avg_energy, 1),
            "average_emotional_intensity": round(avg_intensity, 1),
            "vocal_percentage": round(self.vocal_instrumental_ratio * 100, 1),
            "genre_distribution": self._get_genre_distribution(),
            "decade_distribution": self._get_decade_distribution()
        }
    
    def _get_genre_distribution(self) -> Dict[str, int]:
        """Get distribution of genres in playlist"""
        genres = {}
        for song in self.songs:
            genres[song.genre] = genres.get(song.genre, 0) + 1
        return genres
    
    def _get_decade_distribution(self) -> Dict[str, int]:
        """Get distribution of decades in playlist"""
        decades = {}
        for song in self.songs:
            if song.release_year:
                decade = f"{(song.release_year // 10) * 10}s"
                decades[decade] = decades.get(decade, 0) + 1
        return decades

class PlaylistManager:
    """Manages playlist creation, modification, and persistence"""
    
    def __init__(self, storage_path: str = "playlists.json"):
        self.storage_path = storage_path
        self.playlists: Dict[str, TherapeuticPlaylist] = {}
        self.user_preferences: Dict[str, Any] = {}
        self.load_playlists()
    
    def create_playlist(self, songs: List[EnhancedSong], metadata: Dict[str, Any]) -> TherapeuticPlaylist:
        """Create a new therapeutic playlist with metadata"""
        
        playlist_id = str(uuid.uuid4())
        
        # Extract metadata with defaults
        name = metadata.get("name", f"Therapy Session - {metadata.get('emotion_focus', 'Mixed')}")
        emotion_focus = metadata.get("emotion_focus", "neutral")
        secondary_emotions = metadata.get("secondary_emotions", [])
        therapeutic_notes = metadata.get("therapeutic_notes", "")
        tags = metadata.get("tags", [])
        
        # Create playlist
        playlist = TherapeuticPlaylist(
            id=playlist_id,
            name=name,
            songs=songs,
            emotion_focus=emotion_focus,
            secondary_emotions=secondary_emotions,
            therapeutic_notes=therapeutic_notes,
            tags=tags
        )
        
        # Store playlist
        self.playlists[playlist_id] = playlist
        
        logging.info(f"Created playlist '{name}' with {len(songs)} songs for emotion '{emotion_focus}'")
        
        return playlist
    
    def modify_playlist(self, playlist_id: str, action: str, **kwargs) -> Optional[TherapeuticPlaylist]:
        """Modify an existing playlist"""
        
        if playlist_id not in self.playlists:
            logging.error(f"Playlist {playlist_id} not found")
            return None
        
        playlist = self.playlists[playlist_id]
        
        if action == "add_song":
            song = kwargs.get("song")
            position = kwargs.get("position", len(playlist.songs))
            
            if song and isinstance(song, EnhancedSong):
                playlist.songs.insert(position, song)
                playlist.user_customized = True
                playlist.__post_init__()  # Recalculate duration
                logging.info(f"Added song '{song.title}' to playlist '{playlist.name}'")
        
        elif action == "remove_song":
            position = kwargs.get("position")
            
            if position is not None and 0 <= position < len(playlist.songs):
                removed_song = playlist.songs.pop(position)
                playlist.user_customized = True
                playlist.__post_init__()  # Recalculate duration
                logging.info(f"Removed song '{removed_song.title}' from playlist '{playlist.name}'")
        
        elif action == "reorder_songs":
            new_order = kwargs.get("new_order")
            
            if new_order and len(new_order) == len(playlist.songs):
                try:
                    playlist.songs = [playlist.songs[i] for i in new_order]
                    playlist.user_customized = True
                    logging.info(f"Reordered songs in playlist '{playlist.name}'")
                except IndexError:
                    logging.error("Invalid song order indices")
                    return None
        
        elif action == "update_metadata":
            # Update playlist metadata
            if "name" in kwargs:
                playlist.name = kwargs["name"]
            if "therapeutic_notes" in kwargs:
                playlist.therapeutic_notes = kwargs["therapeutic_notes"]
            if "tags" in kwargs:
                playlist.tags = kwargs["tags"]
            if "user_rating" in kwargs:
                playlist.user_rating = kwargs["user_rating"]
            
            playlist.user_customized = True
            logging.info(f"Updated metadata for playlist '{playlist.name}'")
        
        else:
            logging.error(f"Unknown playlist action: {action}")
            return None
        
        return playlist
    
    def save_playlist(self, playlist: TherapeuticPlaylist, user_id: str = "default") -> bool:
        """Save playlist to persistent storage"""
        
        try:
            # Update playlist in memory
            self.playlists[playlist.id] = playlist
            
            # Save to file
            self._save_to_file()
            
            logging.info(f"Saved playlist '{playlist.name}' for user {user_id}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to save playlist: {e}")
            return False
    
    def load_saved_playlists(self, user_id: str = "default") -> List[TherapeuticPlaylist]:
        """Load user's saved playlists"""
        
        # For now, return all playlists (in a real app, filter by user_id)
        return list(self.playlists.values())
    
    def delete_playlist(self, playlist_id: str) -> bool:
        """Delete a playlist"""
        
        if playlist_id in self.playlists:
            playlist_name = self.playlists[playlist_id].name
            del self.playlists[playlist_id]
            self._save_to_file()
            logging.info(f"Deleted playlist '{playlist_name}'")
            return True
        
        return False
    
    def get_playlist(self, playlist_id: str) -> Optional[TherapeuticPlaylist]:
        """Get a specific playlist by ID"""
        return self.playlists.get(playlist_id)
    
    def search_playlists(self, query: str, emotion: str = None, tags: List[str] = None) -> List[TherapeuticPlaylist]:
        """Search playlists by name, emotion, or tags"""
        
        results = []
        query_lower = query.lower() if query else ""
        
        for playlist in self.playlists.values():
            # Check name match
            name_match = query_lower in playlist.name.lower() if query else True
            
            # Check emotion match
            emotion_match = (emotion is None or 
                           playlist.emotion_focus.lower() == emotion.lower() or
                           emotion.lower() in [e.lower() for e in playlist.secondary_emotions])
            
            # Check tags match
            tags_match = (tags is None or 
                         any(tag.lower() in [t.lower() for t in playlist.tags] for tag in tags))
            
            if name_match and emotion_match and tags_match:
                results.append(playlist)
        
        return results
    
    def get_popular_playlists(self, limit: int = 10) -> List[TherapeuticPlaylist]:
        """Get most played playlists"""
        
        sorted_playlists = sorted(
            self.playlists.values(),
            key=lambda p: p.play_count,
            reverse=True
        )
        
        return sorted_playlists[:limit]
    
    def get_recent_playlists(self, limit: int = 10) -> List[TherapeuticPlaylist]:
        """Get recently created playlists"""
        
        sorted_playlists = sorted(
            self.playlists.values(),
            key=lambda p: p.created_at,
            reverse=True
        )
        
        return sorted_playlists[:limit]
    
    def increment_play_count(self, playlist_id: str) -> bool:
        """Increment play count for a playlist"""
        
        if playlist_id in self.playlists:
            self.playlists[playlist_id].play_count += 1
            self._save_to_file()
            return True
        
        return False
    
    def validate_playlist(self, playlist: TherapeuticPlaylist) -> Dict[str, Any]:
        """Validate playlist and return validation results"""
        
        validation = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "suggestions": []
        }
        
        # Check minimum songs
        if len(playlist.songs) < 1:
            validation["errors"].append("Playlist must contain at least one song")
            validation["is_valid"] = False
        
        # Check maximum duration (2 hours)
        if playlist.total_duration > 7200:
            validation["warnings"].append("Playlist is longer than 2 hours - consider splitting")
        
        # Check therapeutic coherence
        if playlist.emotion_focus and playlist.songs:
            # Check if songs match the emotional focus
            mismatched_songs = []
            for song in playlist.songs:
                if (song.mood.lower() not in playlist.emotion_focus.lower() and 
                    playlist.emotion_focus.lower() not in song.mood.lower()):
                    mismatched_songs.append(song.title)
            
            if len(mismatched_songs) > len(playlist.songs) * 0.5:  # More than 50% mismatch
                validation["warnings"].append(
                    f"Many songs don't match emotion focus '{playlist.emotion_focus}'"
                )
        
        # Suggest improvements
        if playlist.vocal_instrumental_ratio < 0.3:
            validation["suggestions"].append("Consider adding more vocal tracks for emotional connection")
        
        if len(set(song.genre for song in playlist.songs)) == 1:
            validation["suggestions"].append("Consider adding genre variety for richer experience")
        
        return validation
    
    def load_playlists(self):
        """Load playlists from file"""
        
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                
                # Convert back to playlist objects
                for playlist_data in data.get("playlists", []):
                    # Convert songs back to EnhancedSong objects
                    songs = []
                    for song_data in playlist_data.get("songs", []):
                        song = EnhancedSong(**song_data)
                        songs.append(song)
                    
                    # Create playlist object
                    playlist_data["songs"] = songs
                    playlist_data["created_at"] = datetime.fromisoformat(playlist_data["created_at"])
                    
                    playlist = TherapeuticPlaylist(**playlist_data)
                    self.playlists[playlist.id] = playlist
                
                self.user_preferences = data.get("user_preferences", {})
                
                logging.info(f"Loaded {len(self.playlists)} playlists from storage")
                
        except FileNotFoundError:
            logging.info("No existing playlist file found, starting fresh")
        except Exception as e:
            logging.error(f"Error loading playlists: {e}")
    
    def _save_to_file(self):
        """Save playlists to file"""
        
        try:
            # Convert playlists to serializable format
            playlists_data = []
            for playlist in self.playlists.values():
                playlist_dict = asdict(playlist)
                playlist_dict["created_at"] = playlist.created_at.isoformat()
                playlists_data.append(playlist_dict)
            
            data = {
                "playlists": playlists_data,
                "user_preferences": self.user_preferences
            }
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logging.error(f"Error saving playlists: {e}")
    
    def export_playlist(self, playlist_id: str, format: str = "json") -> Optional[str]:
        """Export playlist in various formats"""
        
        if playlist_id not in self.playlists:
            return None
        
        playlist = self.playlists[playlist_id]
        
        if format == "json":
            return json.dumps(asdict(playlist), indent=2, default=str)
        
        elif format == "text":
            lines = [
                f"# {playlist.name}",
                f"Emotion Focus: {playlist.emotion_focus}",
                f"Duration: {playlist.get_duration_formatted()}",
                f"Songs: {len(playlist.songs)}",
                "",
                "## Tracklist:"
            ]
            
            for i, song in enumerate(playlist.songs, 1):
                lines.append(f"{i}. {song.title} - {song.artist} ({song.duration//60}:{song.duration%60:02d})")
            
            if playlist.therapeutic_notes:
                lines.extend(["", "## Therapeutic Notes:", playlist.therapeutic_notes])
            
            return "\n".join(lines)
        
        elif format == "m3u":
            lines = ["#EXTM3U"]
            for song in playlist.songs:
                lines.append(f"#EXTINF:{song.duration},{song.artist} - {song.title}")
                lines.append(song.url)
            
            return "\n".join(lines)
        
        return None