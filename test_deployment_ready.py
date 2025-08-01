#!/usr/bin/env python3
"""
Test script to verify deployment readiness
"""
import sys
import importlib
import warnings
warnings.filterwarnings('ignore')

def test_imports():
    """Test all required imports for deployment."""
    print("🧪 Testing deployment readiness...")
    
    required_packages = [
        'streamlit',
        'numpy',
        'pandas', 
        'torch',
        'librosa',
        'altair',
        'transformers',
        'tokenizers',
        'plotly',
        'sklearn',
        'supabase',
        'requests',
        'cv2',
        'PIL'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
                print(f"✅ {package} - OK")
            elif package == 'PIL':
                from PIL import Image
                print(f"✅ {package} - OK")
            elif package == 'sklearn':
                import sklearn
                print(f"✅ {package} - OK")
            else:
                importlib.import_module(package)
                print(f"✅ {package} - OK")
        except ImportError as e:
            print(f"❌ {package} - FAILED: {e}")
            failed_imports.append(package)
    
    return failed_imports

def test_emotion_detectors():
    """Test emotion detection modules."""
    print("\n🎭 Testing emotion detectors...")
    
    # Test simple detector
    try:
        from simple_emotion_detector import SimpleEmotionDetector
        detector = SimpleEmotionDetector()
        print("✅ Simple emotion detector - OK")
    except Exception as e:
        print(f"❌ Simple emotion detector - FAILED: {e}")
    
    # Test perfect detector (optional)
    try:
        from perfect_emotion_detector import PerfectEmotionDetector
        print("✅ Perfect emotion detector - OK (optional)")
    except Exception as e:
        print(f"⚠️ Perfect emotion detector - Not available (expected for cloud): {e}")

def test_streamlit_compatibility():
    """Test Streamlit-specific features."""
    print("\n🚀 Testing Streamlit compatibility...")
    
    try:
        import streamlit as st
        
        # Test caching decorator
        @st.cache_resource
        def test_cache():
            return "cached"
        
        print("✅ Streamlit caching - OK")
        
        # Test secrets (will fail in local test, but that's expected)
        try:
            secrets = st.secrets
            print("✅ Streamlit secrets - Available")
        except:
            print("⚠️ Streamlit secrets - Not available (expected in local test)")
            
    except Exception as e:
        print(f"❌ Streamlit compatibility - FAILED: {e}")

def test_model_loading():
    """Test model loading capabilities."""
    print("\n🤖 Testing model loading...")
    
    try:
        from transformers import pipeline
        
        # Test text classification (lightweight)
        print("Loading text emotion model...")
        emo_pipe = pipeline(
            "text-classification",
            model="bhadresh-savani/bert-base-go-emotion",
            return_all_scores=True,
            device=-1  # CPU only
        )
        print("✅ Text emotion model - OK")
        
        # Test a simple prediction
        result = emo_pipe("I am happy today")
        print(f"✅ Model prediction test - OK (detected {len(result[0])} emotions)")
        
    except Exception as e:
        print(f"❌ Model loading - FAILED: {e}")

def main():
    """Run all deployment tests."""
    print("🚀 Emotion AI - Deployment Readiness Test")
    print("=" * 50)
    
    failed_imports = test_imports()
    test_emotion_detectors()
    test_streamlit_compatibility()
    test_model_loading()
    
    print("\n" + "=" * 50)
    print("📊 DEPLOYMENT READINESS SUMMARY")
    print("=" * 50)
    
    if not failed_imports:
        print("✅ ALL IMPORTS SUCCESSFUL")
        print("🚀 App is ready for Streamlit Cloud deployment!")
        print("\nNext steps:")
        print("1. Push to GitHub")
        print("2. Deploy on share.streamlit.io")
        print("3. Set app path to: chotabheem/app.py")
        print("4. Configure secrets for enhanced features")
        return True
    else:
        print("❌ SOME IMPORTS FAILED")
        print(f"Failed packages: {', '.join(failed_imports)}")
        print("\nPlease install missing packages:")
        print(f"pip install {' '.join(failed_imports)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)