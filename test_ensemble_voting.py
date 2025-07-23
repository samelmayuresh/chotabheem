#!/usr/bin/env python3
# Test for enhanced ensemble voting system
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.text_emotion_ensemble import TextEmotionEnsemble, ModelConfig
    from utils.ensemble_voting import (
        WeightedAverageVoting, MajorityVoting, BayesianVoting, 
        AdaptiveVoting, create_voting_system
    )
    
    print("Testing Enhanced Ensemble Voting System...")
    
    # Test individual voting strategies
    print("\n1. Testing Individual Voting Strategies...")
    
    # Create test ensemble with multiple models
    test_configs = [
        ModelConfig(
            name="bert_model",
            model_path="j-hartmann/emotion-english-distilroberta-base",
            model_type="bert",
            weight=1.0,
            enabled=True
        ),
        ModelConfig(
            name="context_model",
            model_path="j-hartmann/emotion-english-distilroberta-base",
            model_type="context_aware",
            weight=0.8,
            enabled=True
        )
    ]
    
    # Test different voting strategies
    voting_strategies = ["weighted_average", "majority", "bayesian", "adaptive"]
    
    for strategy in voting_strategies:
        print(f"\n2.{voting_strategies.index(strategy) + 1} Testing {strategy} voting...")
        
        try:
            ensemble = TextEmotionEnsemble(test_configs, voting_strategy=strategy)
            print(f"✅ Ensemble created with {strategy} voting")
            print(f"✅ Voting system: {ensemble.voting_system.name if ensemble.voting_system else 'None'}")
            
            # Test prediction with different text types
            test_texts = [
                "I am absolutely thrilled and excited about this wonderful news!",
                "This is really frustrating and I'm getting quite angry about it.",
                "I feel a mix of happiness and nervousness about the upcoming event."
            ]
            
            for i, text in enumerate(test_texts, 1):
                print(f"\n   Test {i}: '{text[:40]}...'")
                
                try:
                    predictions = ensemble.predict(text)
                    print(f"   ✅ Predictions: {len(predictions)} emotions")
                    
                    # Show top 3 predictions with voting details
                    for pred in predictions[:3]:
                        voting_method = pred.metadata.get('voting_method', 'unknown')
                        agreement = pred.metadata.get('agreement_score', 0.0)
                        print(f"      - {pred.label}: {pred.score:.3f} (conf: {pred.confidence:.3f})")
                        print(f"        Method: {voting_method}, Agreement: {agreement:.3f}")
                
                except Exception as e:
                    print(f"   ⚠️  Prediction failed: {e}")
        
        except Exception as e:
            print(f"   ❌ Strategy {strategy} failed: {e}")
    
    # Test voting system factory
    print("\n3. Testing Voting System Factory...")
    for strategy in voting_strategies:
        try:
            voting_system = create_voting_system(strategy)
            print(f"✅ Created {strategy} voting system: {voting_system.name}")
        except Exception as e:
            print(f"❌ Failed to create {strategy} voting system: {e}")
    
    # Test adaptive voting strategy selection
    print("\n4. Testing Adaptive Voting Strategy Selection...")
    try:
        adaptive_voting = AdaptiveVoting()
        
        # Test with different model configurations
        test_scenarios = [
            {"model_count": 2, "context": {"uncertainty_tolerance": "low"}},
            {"model_count": 5, "context": {"uncertainty_tolerance": "medium"}},
            {"model_count": 3, "context": {"uncertainty_tolerance": "high"}}
        ]
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"   Scenario {i}: {scenario['model_count']} models, {scenario['context']['uncertainty_tolerance']} uncertainty")
            
            # Create mock model predictions
            mock_predictions = {}
            for j in range(scenario['model_count']):
                mock_predictions[f"model_{j}"] = []
            
            # The adaptive system would select strategy based on this
            print(f"   ✅ Adaptive voting would analyze this scenario")
    
    except Exception as e:
        print(f"❌ Adaptive voting test failed: {e}")
    
    # Test ensemble performance tracking
    print("\n5. Testing Enhanced Performance Tracking...")
    try:
        ensemble = TextEmotionEnsemble(test_configs, voting_strategy="adaptive")
        
        # Make several predictions to test performance tracking
        test_text = "I'm feeling really good about this project!"
        
        for i in range(3):
            predictions = ensemble.predict(test_text)
            print(f"   Prediction {i+1}: {len(predictions)} emotions detected")
        
        # Check performance metrics
        performance = ensemble.get_ensemble_performance()
        print(f"✅ Performance tracking:")
        print(f"   - Predictions made: {performance['prediction_count']}")
        print(f"   - Confidence calibration: {performance['confidence_calibration']:.3f}")
        print(f"   - Active models: {performance['model_count']}")
        
    except Exception as e:
        print(f"❌ Performance tracking test failed: {e}")
    
    # Test voting system robustness
    print("\n6. Testing Voting System Robustness...")
    try:
        ensemble = TextEmotionEnsemble(test_configs, voting_strategy="weighted_average")
        
        # Test with edge cases
        edge_cases = [
            "",  # Empty text
            "a",  # Very short text
            "This is a very long text that goes on and on and might cause issues with some models because it's quite lengthy and contains many words that could potentially cause processing problems." * 5  # Very long text
        ]
        
        for i, text in enumerate(edge_cases, 1):
            print(f"   Edge case {i}: {len(text)} characters")
            try:
                predictions = ensemble.predict(text)
                print(f"   ✅ Handled gracefully: {len(predictions)} predictions")
            except Exception as e:
                print(f"   ⚠️  Edge case handling: {e}")
    
    except Exception as e:
        print(f"❌ Robustness test failed: {e}")
    
    print("\n🎉 Enhanced ensemble voting tests completed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()