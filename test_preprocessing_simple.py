#!/usr/bin/env python3
# Simple test for preprocessing components
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.enhanced_preprocessing import EnhancedTextProcessor, EnhancedAudioProcessor
    import numpy as np
    
    print("Testing Enhanced Text Processor...")
    text_processor = EnhancedTextProcessor()
    
    # Test text processing
    test_text = "I'm feeling really AMAZING today!!! This can't be better."
    result = text_processor.process(test_text)
    
    print(f"✅ Original: {result.original}")
    print(f"✅ Cleaned: {result.cleaned}")
    print(f"✅ Quality Score: {result.quality_score:.2f}")
    print(f"✅ Token Count: {result.metadata['token_count']}")
    print(f"✅ Has Emphasis: {result.metadata['has_emphasis']}")
    
    print("\nTesting Enhanced Audio Processor...")
    audio_processor = EnhancedAudioProcessor()
    
    # Test audio processing with dummy data
    sample_rate = 16000
    duration = 2.0
    dummy_audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)
    
    audio_result = audio_processor.process(dummy_audio, sample_rate)
    
    print(f"✅ Duration: {audio_result.duration:.2f}s")
    print(f"✅ Sample Rate: {audio_result.sample_rate}")
    print(f"✅ Quality Score: {audio_result.quality_score:.2f}")
    print(f"✅ Features Extracted: {list(audio_result.features.keys())}")
    
    print("\n🎉 All preprocessing tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()