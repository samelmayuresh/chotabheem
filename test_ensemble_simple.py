#!/usr/bin/env python3
# Simple test for ensemble with voting
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.text_emotion_ensemble import TextEmotionEnsemble, ModelConfig
    
    print("Testing Simple Ensemble with Voting...")
    
    # Create simple ensemble configuration
    print("\n1. Creating Simple Ensemble...")
    test_configs = [
        ModelConfig(
            name="test_model",
            model_path="j-hartmann/emotion-english-distilroberta-base",
            model_type="bert",
            weight=1.0,
            enabled=True
        )
    ]
    
    # Test with simple voting strategy first
    ensemble = TextEmotionEnsemble(test_configs, voting_strategy="simple")
    print(f"✅ Ensemble created with simple voting")
    print(f"✅ Models: {list(ensemble.models.keys())}")
    
    # Test prediction
    print("\n2. Testing Prediction...")
    test_text = "I am feeling really happy today!"
    
    try:
        predictions = ensemble.predict(test_text)
        print(f"✅ Predictions: {len(predictions)} emotions")
        
        for pred in predictions[:3]:
            print(f"   - {pred.label}: {pred.score:.3f} (conf: {pred.confidence:.3f})")
            print(f"     Method: {pred.metadata.get('voting_method', 'unknown')}")
    
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test performance tracking
    print("\n3. Testing Performance...")
    performance = ensemble.get_ensemble_performance()
    print(f"✅ Predictions made: {performance['prediction_count']}")
    print(f"✅ Active models: {performance['model_count']}")
    
    print("\n🎉 Simple ensemble test completed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()