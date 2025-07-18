#!/usr/bin/env python3
"""
Test database connection and functionality
"""

from config import api_config
from utils.hybrid_database import HybridEmotionDatabase
from datetime import datetime
import json

def test_database_operations():
    """Test all database operations"""
    print("🔬 Testing Database Operations")
    print("="*50)
    
    # Initialize database
    print("1️⃣ Initializing database...")
    db = HybridEmotionDatabase(
        supabase_url=api_config.supabase_url,
        supabase_key=api_config.supabase_key
    )
    
    if db.supabase_available:
        print("✅ Supabase connection active")
    else:
        print("⚠️ Using local storage fallback")
    
    # Test emotion logging
    print("\n2️⃣ Testing emotion logging...")
    test_emotion = {
        "primary": "joy",
        "confidence": 0.85,
        "all_scores": [
            {"emotion": "joy", "score": 0.85},
            {"emotion": "excitement", "score": 0.12},
            {"emotion": "neutral", "score": 0.03}
        ],
        "source": "test"
    }
    
    test_context = {
        "session_id": "test_session_123",
        "input_length": 50,
        "processing_time": 1.2
    }
    
    success = db.log_emotion(test_emotion, test_context)
    if success:
        print("✅ Emotion logged successfully")
    else:
        print("❌ Failed to log emotion")
        return False
    
    # Test data retrieval
    print("\n3️⃣ Testing data retrieval...")
    df = db.get_mood_history(days=30)
    print(f"✅ Retrieved {len(df)} records")
    
    if not df.empty:
        print("   Recent records:")
        for _, row in df.head(3).iterrows():
            print(f"   - {row['created_at']}: {row['emotion']} (confidence: {row.get('confidence', row.get('score', 0)):.2f})")
    
    # Test analytics
    print("\n4️⃣ Testing analytics...")
    analytics = db.get_emotion_analytics(days=30)
    print(f"✅ Analytics generated:")
    print(f"   - Total entries: {analytics['total_entries']}")
    print(f"   - Most common emotion: {analytics['most_common_emotion']}")
    print(f"   - Average confidence: {analytics['average_confidence']:.2f}")
    print(f"   - Confidence trend: {analytics['confidence_trend']}")
    
    # Test insights
    print("\n5️⃣ Testing personalized insights...")
    insights = db.get_personalized_insights(days=30)
    print("✅ Generated insights:")
    for i, insight in enumerate(insights, 1):
        print(f"   {i}. {insight}")
    
    print("\n🎉 All database operations working correctly!")
    return True

if __name__ == "__main__":
    test_database_operations()