#!/usr/bin/env python3
# Test for confidence calibration system
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.confidence_calibration import ConfidenceCalibrator, CalibrationMetrics, ReliabilityAssessment
    from utils.text_emotion_ensemble import EmotionScore
    import numpy as np
    
    print("Testing Confidence Calibration System...")
    
    # Create mock emotion scores for testing
    def create_mock_emotions(confidence_levels):
        emotions = []
        for i, conf in enumerate(confidence_levels):
            emotions.append(EmotionScore(
                label=f"emotion_{i}",
                score=conf + np.random.normal(0, 0.1),
                confidence=conf,
                source="test",
                model_name="test_model",
                processing_time=0.1,
                metadata={"model_type": "test"}
            ))
        return emotions
    
    # Test confidence calibrator
    print("\n1. Testing Confidence Calibrator...")
    calibrator = ConfidenceCalibrator()
    
    # Test with various confidence levels
    test_confidences = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    mock_emotions = create_mock_emotions(test_confidences)
    
    print(f"✅ Created {len(mock_emotions)} mock emotions")
    
    # Test calibration without ground truth
    calibrated_emotions = calibrator.calibrate_confidence(mock_emotions)
    
    print(f"✅ Calibrated {len(calibrated_emotions)} emotions")
    for i, (orig, calib) in enumerate(zip(mock_emotions, calibrated_emotions)):
        print(f"   {i+1}. Original: {orig.confidence:.3f} → Calibrated: {calib.confidence:.3f}")
    
    # Test reliability assessment
    print("\n2. Testing Reliability Assessment...")
    reliability = calibrator.assess_reliability(mock_emotions)
    
    print(f"✅ Reliability assessment completed")
    print(f"   - Overall reliability: {reliability.overall_reliability:.3f}")
    print(f"   - Confidence level: {reliability.confidence_level}")
    print(f"   - Reliability factors: {len(reliability.reliability_factors)} factors")
    print(f"   - Uncertainty metrics: {len(reliability.uncertainty_quantification)} metrics")
    print(f"   - Recommendations: {len(reliability.recommendations)} items")
    
    for rec in reliability.recommendations[:3]:
        print(f"     • {rec}")
    
    # Test calibration metrics
    print("\n3. Testing Calibration Metrics...")
    
    # Create mock prediction data
    mock_predictions = []
    for i in range(100):
        confidence = np.random.uniform(0.1, 0.9)
        # Simulate imperfect calibration
        actual_accuracy = confidence + np.random.normal(0, 0.2)
        is_correct = np.random.random() < max(0, min(1, actual_accuracy))
        
        mock_predictions.append({
            'confidence': confidence,
            'is_correct': is_correct,
            'model_name': 'test_model'
        })
    
    metrics = calibrator.calculate_calibration_metrics(mock_predictions)
    
    print(f"✅ Calibration metrics calculated")
    print(f"   - Reliability score: {metrics.reliability_score:.3f}")
    print(f"   - Calibration error: {metrics.calibration_error:.3f}")
    print(f"   - Brier score: {metrics.brier_score:.3f}")
    print(f"   - Expected calibration error: {metrics.expected_calibration_error:.3f}")
    print(f"   - Maximum calibration error: {metrics.maximum_calibration_error:.3f}")
    print(f"   - Sharpness: {metrics.sharpness:.3f}")
    
    # Test with ground truth for model training
    print("\n4. Testing Calibration with Ground Truth...")
    
    ground_truth_emotions = create_mock_emotions([0.8, 0.6, 0.4])
    ground_truth_labels = ["emotion_0", "emotion_1"]  # Some correct, some not
    
    calibrated_with_gt = calibrator.calibrate_confidence(
        ground_truth_emotions, 
        ground_truth_labels
    )
    
    print(f"✅ Calibration with ground truth completed")
    print(f"   - History size: {len(calibrator.calibration_history)}")
    
    # Test calibration summary
    print("\n5. Testing Calibration Summary...")
    summary = calibrator.get_calibration_summary()
    
    print(f"✅ Calibration summary generated")
    print(f"   - Models fitted: {summary['calibration_models']}")
    print(f"   - History size: {summary['history_size']}")
    print(f"   - Recent performance: {summary['recent_performance']:.3f}")
    print(f"   - Method: {summary['calibration_method']}")
    
    # Test edge cases
    print("\n6. Testing Edge Cases...")
    
    # Empty emotions
    empty_reliability = calibrator.assess_reliability([])
    print(f"✅ Empty emotions handled: {empty_reliability.confidence_level}")
    
    # Single emotion
    single_emotion = [create_mock_emotions([0.7])[0]]
    single_reliability = calibrator.assess_reliability(single_emotion)
    print(f"✅ Single emotion handled: {single_reliability.overall_reliability:.3f}")
    
    # Very high confidence
    high_conf_emotions = create_mock_emotions([0.99, 0.98, 0.97])
    high_conf_reliability = calibrator.assess_reliability(high_conf_emotions)
    print(f"✅ High confidence handled: {high_conf_reliability.confidence_level}")
    
    # Very low confidence
    low_conf_emotions = create_mock_emotions([0.01, 0.02, 0.03])
    low_conf_reliability = calibrator.assess_reliability(low_conf_emotions)
    print(f"✅ Low confidence handled: {low_conf_reliability.confidence_level}")
    
    print("\n🎉 Confidence calibration tests completed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()