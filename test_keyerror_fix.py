#!/usr/bin/env python3
"""
Test the KeyError fix for mood_trajectory
"""

from utils.hybrid_database import HybridEmotionDatabase

def test_mood_trajectory_keys():
    """Test that all required keys are present in mood_trajectory"""
    try:
        # Initialize database
        db = HybridEmotionDatabase()
        
        # Get analytics
        analytics = db.get_emotion_analytics()
        
        # Check mood_trajectory keys
        mood_trajectory = analytics.get("mood_trajectory", {})
        
        required_keys = ["trajectory", "recent_mood_score", "historical_mood_score", "improvement"]
        
        print("🔍 Checking mood_trajectory keys...")
        for key in required_keys:
            if key in mood_trajectory:
                print(f"✅ {key}: {mood_trajectory[key]}")
            else:
                print(f"❌ Missing key: {key}")
                return False
        
        print("\n✅ All required keys are present!")
        print("✅ KeyError fix successful!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_mood_trajectory_keys()
    if success:
        print("\n🎉 Your app should now run without KeyError!")
    else:
        print("\n❌ There are still issues to fix.")