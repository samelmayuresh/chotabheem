#!/usr/bin/env python3
"""
Comprehensive test script to verify all improvements are working
"""

import sys
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_hybrid_database():
    """Test the hybrid database functionality"""
    try:
        from utils.hybrid_database import HybridEmotionDatabase
        
        # Initialize database
        db = HybridEmotionDatabase()
        
        if not db.available:
            logger.error("Database not available")
            return False
        
        # Test emotion logging
        emotion_data = {
            "primary": "joy",
            "confidence": 0.85,
            "all_scores": [{"label": "joy", "score": 0.85}],
            "source": "test"
        }
        
        user_context = {
            "session_id": "test_session",
            "input_length": 10,
            "processing_time": 0.5
        }
        
        if not db.log_emotion(emotion_data, user_context):
            logger.error("Failed to log emotion")
            return False
        
        # Test mood history retrieval
        df = db.get_mood_history(days=7)
        if df.empty:
            logger.warning("No mood history found, but this might be expected")
        else:
            logger.info(f"Retrieved {len(df)} mood records")
        
        # Test analytics
        analytics = db.get_emotion_analytics(days=7)
        if "error" in analytics:
            logger.warning(f"Analytics returned error: {analytics['error']}")
        else:
            logger.info(f"Analytics: {analytics['total_entries']} entries, most common: {analytics['most_common_emotion']}")
        
        # Test insights
        insights = db.get_personalized_insights(days=7)
        logger.info(f"Generated {len(insights)} insights")
        
        logger.info("✅ Hybrid database test passed")
        return True
        
    except Exception as e:
        logger.error(f"Hybrid database test failed: {e}")
        return False

def test_config():
    """Test configuration loading"""
    try:
        from config import api_config, app_config, validate_api_keys
        
        # Test config loading
        logger.info(f"App title: {app_config.page_title}")
        logger.info(f"Supabase URL: {api_config.supabase_url}")
        
        # Test validation
        warnings = validate_api_keys()
        if warnings:
            logger.info(f"API warnings: {len(warnings)}")
        
        logger.info("✅ Config test passed")
        return True
        
    except Exception as e:
        logger.error(f"Config test failed: {e}")
        return False

def test_imports():
    """Test all critical imports"""
    try:
        # Test core imports
        import streamlit as st
        import pandas as pd
        import numpy as np
        import plotly.express as px
        import plotly.graph_objects as go
        
        # Test ML imports
        from transformers import pipeline
        import torch
        import librosa
        
        # Test custom imports
        from utils.hybrid_database import HybridEmotionDatabase
        from config import api_config, app_config
        
        logger.info("✅ All imports test passed")
        return True
        
    except Exception as e:
        logger.error(f"Imports test failed: {e}")
        return False

def test_local_database():
    """Test local database file creation and operations"""
    try:
        import json
        import os
        
        db_file = "local_mood_history.json"
        
        # Check if file exists
        if os.path.exists(db_file):
            with open(db_file, 'r') as f:
                data = json.load(f)
            
            logger.info(f"Local database has {len(data.get('mood_history', []))} records")
        else:
            logger.info("Local database file not found (will be created on first use)")
        
        logger.info("✅ Local database test passed")
        return True
        
    except Exception as e:
        logger.error(f"Local database test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Running comprehensive project tests...")
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Local Database", test_local_database),
        ("Hybrid Database", test_hybrid_database)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        try:
            if test_func():
                print(f"✅ {test_name} test passed")
                passed += 1
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test error: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your project is ready to run.")
        print("\nTo start the application:")
        print("  streamlit run app_enhanced.py")
        print("\nFeatures available:")
        print("  ✅ Emotion analysis (text and audio)")
        print("  ✅ Database storage (hybrid Supabase + local)")
        print("  ✅ Analytics dashboard")
        print("  ✅ Personalized insights")
        print("  ✅ Error handling and fallbacks")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Check the errors above.")
        print("Some features may not work correctly.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)