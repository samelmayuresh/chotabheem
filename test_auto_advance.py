# test_auto_advance.py - Test auto-advance and continuous playback functionality
import time
import threading
from unittest.mock import Mock, patch
from utils.player_engine import PlayerEngine, PlaybackStatus
from utils.enhanced_song_database import EnhancedSong
from utils.playlist_manager import TherapeuticPlaylist

def test_auto_advance_functionality():
    """Test auto-advance and continuous playback"""
    
    print("🎵 Testing Auto-Advance and Continuous Playback")
    print("=" * 50)
    
    # Create test songs with short durations for testing
    test_songs = [
        EnhancedSong(
            title="Short Song 1", artist="Test Artist", url="https://example.com/1",
            duration=2, genre="Pop", mood="Happy"  # 2 second duration for testing
        ),
        EnhancedSong(
            title="Short Song 2", artist="Test Artist", url="https://example.com/2",
            duration=2, genre="Rock", mood="Energetic"
        ),
        EnhancedSong(
            title="Short Song 3", artist="Test Artist", url="https://example.com/3",
            duration=2, genre="Folk", mood="Calm"
        )
    ]
    
    test_playlist = TherapeuticPlaylist(
        id="auto-advance-test",
        name="Auto-Advance Test Playlist",
        songs=test_songs,
        emotion_focus="mixed"
    )
    
    # Create callbacks to track events
    song_changes = []
    playlist_completions = []
    errors = []
    
    def on_song_change(song, state):
        song_changes.append({
            'song': song.title,
            'position': state.playlist_position,
            'timestamp': time.time()
        })
        print(f"  🎵 Song changed to: {song.title} (position {state.playlist_position})")
    
    def on_playlist_complete(playlist, state):
        playlist_completions.append({
            'playlist': playlist.name,
            'timestamp': time.time()
        })
        print(f"  ✅ Playlist completed: {playlist.name}")
    
    def on_error(error, song, state):
        errors.append({
            'error': str(error),
            'song': song.title,
            'timestamp': time.time()
        })
        print(f"  ❌ Error with {song.title}: {error}")
    
    # Create player engine with callbacks
    engine = PlayerEngine(
        on_song_change=on_song_change,
        on_playlist_complete=on_playlist_complete,
        on_error=on_error
    )
    
    try:
        # Test 1: Basic auto-advance
        print("\n1. Testing basic auto-advance...")
        engine.load_playlist(test_playlist)
        
        # Start playing with auto-advance enabled
        success = engine.play_song(auto_advance=True)
        assert success, "Failed to start playback"
        assert engine.state.auto_advance == True
        print("  ✅ Auto-advance enabled and playback started")
        
        # Wait for first song to complete and auto-advance
        print("  ⏳ Waiting for auto-advance...")
        time.sleep(3)  # Wait longer than song duration
        
        # Check that we advanced to the next song
        if engine.state.playlist_position > 0:
            print("  ✅ Auto-advance to next song successful")
        else:
            print("  ⚠️  Auto-advance may not have triggered yet")
        
        # Test 2: Manual skip during auto-advance
        print("\n2. Testing manual skip during auto-advance...")
        original_position = engine.state.playlist_position
        engine.skip_to_next()
        
        if engine.state.playlist_position > original_position:
            print("  ✅ Manual skip during auto-advance works")
        
        # Test 3: Pause and resume with auto-advance
        print("\n3. Testing pause/resume with auto-advance...")
        engine.pause()
        assert engine.state.status == PlaybackStatus.PAUSED
        print("  ✅ Paused successfully")
        
        engine.resume()
        assert engine.state.status == PlaybackStatus.PLAYING
        print("  ✅ Resumed successfully")
        
        # Test 4: Repeat modes
        print("\n4. Testing repeat modes...")
        
        # Test song repeat
        engine.set_repeat_mode("song")
        current_song = engine.state.current_song.title
        print(f"  🔁 Set to repeat song: {current_song}")
        
        # Test playlist repeat
        engine.set_repeat_mode("playlist")
        print("  🔁 Set to repeat playlist")
        
        # Reset to no repeat
        engine.set_repeat_mode("none")
        print("  ⏹️  Set to no repeat")
        
        # Test 5: Playlist completion handling
        print("\n5. Testing playlist completion...")
        
        # Skip to last song
        engine.load_playlist(test_playlist, start_position=len(test_songs) - 1)
        engine.play_song(auto_advance=True)
        
        print("  ⏳ Waiting for playlist completion...")
        time.sleep(3)  # Wait for last song to complete
        
        # Check if playlist completed
        if engine.state.status == PlaybackStatus.STOPPED:
            print("  ✅ Playlist completion handled correctly")
        
        # Test 6: Error recovery
        print("\n6. Testing error recovery...")
        
        # Create a playlist with a "problematic" song
        problematic_song = EnhancedSong(
            title="Problematic Song", artist="Test", url="invalid://url",
            duration=2, genre="Test", mood="Error"
        )
        
        error_test_songs = [test_songs[0], problematic_song, test_songs[1]]
        error_playlist = TherapeuticPlaylist(
            id="error-test",
            name="Error Test Playlist", 
            songs=error_test_songs,
            emotion_focus="test"
        )
        
        engine.load_playlist(error_playlist)
        engine.play_song(auto_advance=True)
        
        # Skip to problematic song
        engine.skip_to_next()
        print("  ⚠️  Simulated error scenario")
        
        # Test 7: Session statistics
        print("\n7. Testing session statistics...")
        stats = engine.get_session_statistics()
        
        print(f"  📊 Session duration: {stats.get('session_duration', 0):.1f}s")
        print(f"  📊 Songs played: {stats.get('songs_played', 0)}")
        print(f"  📊 User interactions: {stats.get('user_interactions', 0)}")
        print(f"  📊 Error count: {stats.get('error_count', 0)}")
        
        # Test 8: Monitoring thread management
        print("\n8. Testing monitoring thread management...")
        
        # Check if monitoring is active during playback
        engine.load_playlist(test_playlist)
        engine.play_song(auto_advance=True)
        
        if engine.is_monitoring:
            print("  ✅ Monitoring thread started with auto-advance")
        
        engine.stop()
        time.sleep(0.1)  # Brief wait for thread cleanup
        
        if not engine.is_monitoring:
            print("  ✅ Monitoring thread stopped correctly")
        
        # Summary
        print("\n" + "=" * 50)
        print("📈 Auto-Advance Test Summary:")
        print(f"  🎵 Song changes tracked: {len(song_changes)}")
        print(f"  ✅ Playlist completions: {len(playlist_completions)}")
        print(f"  ❌ Errors encountered: {len(errors)}")
        
        if song_changes:
            print("  🎵 Song change sequence:")
            for change in song_changes:
                print(f"    - {change['song']} (pos {change['position']})")
        
        print("\n🎉 Auto-advance and continuous playback tests completed!")
        
    finally:
        # Cleanup
        engine.cleanup()
        print("🧹 Cleanup completed")

def test_repeat_modes():
    """Test different repeat modes in detail"""
    
    print("\n🔁 Testing Repeat Modes in Detail")
    print("=" * 40)
    
    # Create short test playlist
    test_songs = [
        EnhancedSong(title="Song A", artist="Artist", url="https://example.com/a", 
                    duration=1, genre="Pop", mood="Happy"),
        EnhancedSong(title="Song B", artist="Artist", url="https://example.com/b", 
                    duration=1, genre="Rock", mood="Energetic")
    ]
    
    playlist = TherapeuticPlaylist(
        id="repeat-test", name="Repeat Test", songs=test_songs, emotion_focus="test"
    )
    
    engine = PlayerEngine()
    
    try:
        # Test song repeat
        print("\n1. Testing song repeat mode...")
        engine.load_playlist(playlist)
        engine.set_repeat_mode("song")
        engine.play_song(auto_advance=True)
        
        original_song = engine.state.current_song.title
        print(f"  🎵 Playing: {original_song}")
        
        # Wait for song to complete and check if it repeats
        time.sleep(2)
        
        if engine.state.current_song.title == original_song:
            print("  ✅ Song repeat mode working")
        
        # Test playlist repeat
        print("\n2. Testing playlist repeat mode...")
        engine.set_repeat_mode("playlist")
        engine.load_playlist(playlist, start_position=1)  # Start at last song
        engine.play_song(auto_advance=True)
        
        print("  🎵 Started at last song, waiting for playlist loop...")
        time.sleep(2)
        
        # Should loop back to first song
        if engine.state.playlist_position == 0:
            print("  ✅ Playlist repeat mode working")
        
        print("🔁 Repeat mode tests completed!")
        
    finally:
        engine.cleanup()

def test_shuffle_mode():
    """Test shuffle functionality"""
    
    print("\n🔀 Testing Shuffle Mode")
    print("=" * 30)
    
    # Create larger playlist for shuffle testing
    test_songs = [
        EnhancedSong(title=f"Song {i}", artist="Artist", url=f"https://example.com/{i}", 
                    duration=1, genre="Pop", mood="Happy")
        for i in range(5)
    ]
    
    playlist = TherapeuticPlaylist(
        id="shuffle-test", name="Shuffle Test", songs=test_songs, emotion_focus="test"
    )
    
    engine = PlayerEngine()
    
    try:
        engine.load_playlist(playlist)
        engine.set_shuffle(True)
        
        print("  🔀 Shuffle enabled")
        
        # Track song order
        song_order = []
        
        for i in range(3):  # Test a few skips
            current_song = engine.state.current_song.title
            song_order.append(current_song)
            print(f"  🎵 Playing: {current_song}")
            
            if i < 2:  # Don't skip on last iteration
                engine.skip_to_next()
        
        print(f"  📝 Song order: {' -> '.join(song_order)}")
        print("🔀 Shuffle mode test completed!")
        
    finally:
        engine.cleanup()

if __name__ == "__main__":
    test_auto_advance_functionality()
    test_repeat_modes()
    test_shuffle_mode()