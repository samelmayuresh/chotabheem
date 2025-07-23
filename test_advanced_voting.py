#!/usr/bin/env python3
# Test advanced voting with ensemble
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.text_emotion_ensemble import TextEmotionEnsemble, ModelConfig
    
    print("Testing Advanced Voting with Ensemble...")
    
    # Create ensemble with multiple models for better voting
    print("\n1. Creating Multi-Model Ensemble...")
    test_configs = [
        ModelConfig(
            name="bert_primary",
            model_path="j-hartmann/emotion-english-distilroberta-base",
            model_type="bert",
            weight=1.0,
            enabled=True
        ),
        ModelConfig(
            name="context_aware",
            model_path="j-hartmann/emotion-english-distilroberta-base",
            model_type="context_aware",
            weight=0.8,
            enabled=True
        )
    ]
    
    # Test different voting strategies
    voting_strategies = ["weighted_average", "adaptive"]
    
    for strategy in voting_strategies:
        print(f"\n2. Testing {strategy} voting...")
        
        try:
            ensemble = TextEmotionEnsemble(test_configs, voting_strategy=strategy)
            print(f"✅ Ensemble created with {strategy} voting")
            print(f"✅ Models: {list(ensemble.models.keys())}")
            
            # Test prediction
            test_text = "I am absolutely thrilled and excited about this amazing opportunity!"
            
            predictions = ensemble.predict(test_text)
            print(f"✅ Predictions: {len(predictions)} emotions")
            
            for pred in predictions[:3]:
                voting_method = pred.metadata.get('voting_method', 'unknown')
                agreement = pred.metadata.get('agreement_score', 0.0)
                confidence_level = pred.metadata.get('confidence_level', 'unknown')
                
                print(f"   - {pred.label}: {pred.score:.3f} (conf: {pred.confidence:.3f})")
                print(f"     Method: {voting_method}, Agreement: {agreement:.3f}, Level: {confidence_level}")
            
            # Test performance
            performance = ensemble.get_ensemble_performance()
            print(f"✅ Performance: {performance['prediction_count']} predictions, calibration: {performance['confidence_calibration']:.3f}")
        
        except Exception as e:
            print(f"❌ Strategy {strategy} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n🎉 Advanced voting tests completed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()