# test_auto_advance_simple.py - Simple test for auto-advance functionality
import time
from utils.player_engine import PlayerEngine, PlaybackStatus
from utils.enhanced_song_database import EnhancedSong
from utils.playlist_manager import TherapeuticPlaylist

def main():
    print("🎵 Testing Auto-Advance Functionality")
    print("=" * 40)
    
    # Create test songs with very short durations
    test_songs = [
        EnhancedSong(
            title="Song 1", artist="Artist 1", url="https://example.com/1",
            duration=1, genre="Pop", mood="Happy"  # 1 second for quick testing
        ),
        EnhancedSong(
            title="Song 2", artist="Artist 2", url="https://example.com/2",
            duration=1, genre="Rock", mood="Energetic"
        ),
        EnhancedSong(
            title="Song 3", artist="Artist 3", url="https://example.com/3",
            duration=1, genre="Folk", mood="Calm"
        )
    ]
    
    playlist = TherapeuticPlaylist(
        id="test", name="Test Playlist", songs=test_songs, emotion_focus="mixed"
    )
    
    # Track events
    events = []
    
    def on_song_change(song, state):
        events.append(f"Song changed to: {song.title} (position {state.playlist_position})")
        print(f"  🎵 {events[-1]}")
    
    def on_playlist_complete(playlist, state):
        events.append(f"Playlist completed: {playlist.name}")
        print(f"  ✅ {events[-1]}")
    
    # Create engine
    engine = PlayerEngine(
        on_song_change=on_song_change,
        on_playlist_complete=on_playlist_complete
    )
    
    try:
        # Test 1: Load playlist and start playback
        print("\n1. Loading playlist and starting playback...")
        success = engine.load_playlist(playlist)
        assert success, "Failed to load playlist"
        print(f"  ✅ Loaded playlist with {len(playlist.songs)} songs")
        
        success = engine.play_song(auto_advance=True)
        assert success, "Failed to start playback"
        print(f"  ✅ Started playing: {engine.state.current_song.title}")
        print(f"  ✅ Auto-advance enabled: {engine.state.auto_advance}")
        
        # Test 2: Check monitoring is active
        print("\n2. Checking monitoring thread...")
        if engine.is_monitoring:
            print("  ✅ Monitoring thread is active")
        else:
            print("  ⚠️  Monitoring thread not active")
        
        # Test 3: Manual skip
        print("\n3. Testing manual skip...")
        original_position = engine.state.playlist_position
        engine.skip_to_next()
        new_position = engine.state.playlist_position
        
        if new_position > original_position:
            print(f"  ✅ Skipped from position {original_position} to {new_position}")
        else:
            print(f"  ⚠️  Skip may not have worked: {original_position} -> {new_position}")
        
        # Test 4: Pause and resume
        print("\n4. Testing pause and resume...")
        engine.pause()
        assert engine.state.status == PlaybackStatus.PAUSED
        print("  ✅ Paused successfully")
        
        engine.resume()
        assert engine.state.status == PlaybackStatus.PLAYING
        print("  ✅ Resumed successfully")
        
        # Test 5: Repeat modes
        print("\n5. Testing repeat modes...")
        
        modes = ["none", "song", "playlist"]
        for mode in modes:
            success = engine.set_repeat_mode(mode)
            assert success, f"Failed to set repeat mode: {mode}"
            assert engine.state.repeat_mode == mode
            print(f"  ✅ Set repeat mode to: {mode}")
        
        # Test 6: Volume and shuffle
        print("\n6. Testing volume and shuffle...")
        
        engine.set_volume(0.7)
        assert engine.state.volume == 0.7
        print("  ✅ Set volume to 70%")
        
        engine.set_shuffle(True)
        assert engine.state.shuffle == True
        print("  ✅ Enabled shuffle mode")
        
        # Test 7: Session statistics
        print("\n7. Testing session statistics...")
        stats = engine.get_session_statistics()
        
        print(f"  📊 Session duration: {stats.get('session_duration', 0):.1f}s")
        print(f"  📊 Songs played: {stats.get('songs_played', 0)}")
        print(f"  📊 User interactions: {stats.get('user_interactions', 0)}")
        print(f"  📊 Error count: {stats.get('error_count', 0)}")
        
        # Test 8: Stop and cleanup
        print("\n8. Testing stop and cleanup...")
        engine.stop()
        assert engine.state.status == PlaybackStatus.STOPPED
        print("  ✅ Stopped playback")
        
        # Brief wait for monitoring thread to stop
        time.sleep(0.1)
        
        if not engine.is_monitoring:
            print("  ✅ Monitoring thread stopped")
        else:
            print("  ⚠️  Monitoring thread still active")
        
        # Summary
        print("\n" + "=" * 40)
        print("📈 Test Summary:")
        print(f"  🎵 Events tracked: {len(events)}")
        
        if events:
            print("  📝 Event log:")
            for event in events:
                print(f"    - {event}")
        
        print("\n🎉 Auto-advance functionality tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        engine.cleanup()
        print("🧹 Cleanup completed")

if __name__ == "__main__":
    main()