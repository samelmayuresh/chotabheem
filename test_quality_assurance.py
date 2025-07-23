#!/usr/bin/env python3
# Test for quality assurance system
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.quality_assurance import QualityAssurance, ValidationResult, BiasDetectionResult, AnomalyDetectionResult
    from utils.text_emotion_ensemble import EmotionScore
    import numpy as np
    from datetime import datetime, timedelta
    
    print("Testing Quality Assurance System...")
    
    # Create mock emotion scores for testing
    def create_normal_emotions():
        return [
            EmotionScore("joy", 0.8, 0.7, "text", "model1", 0.1, {}),
            EmotionScore("sadness", 0.2, 0.3, "text", "model1", 0.1, {}),
            EmotionScore("neutral", 0.5, 0.6, "voice", "model2", 0.2, {})
        ]
    
    def create_biased_emotions():
        return [
            EmotionScore("joy", 0.95, 0.99, "text", "model1", 0.1, {}),
            EmotionScore("joy", 0.93, 0.98, "text", "model1", 0.1, {}),
            EmotionScore("joy", 0.91, 0.97, "text", "model1", 0.1, {})
        ]
    
    def create_inconsistent_emotions():
        return [
            EmotionScore("joy", 0.9, 0.2, "text", "model1", 0.1, {}),  # Score-confidence mismatch
            EmotionScore("sadness", 0.1, 0.8, "text", "model1", 5.0, {}),  # High processing time
            EmotionScore("anger", 1.5, 0.5, "voice", "model2", 0.1, {})  # Invalid score
        ]
    
    # Test Quality Assurance System
    print("\n1. Testing Quality Assurance System...")
    qa_system = QualityAssurance()
    
    # Test normal validation
    print("\n1.1 Testing Normal Validation...")
    normal_emotions = create_normal_emotions()
    input_data = {"type": "text", "length": 100}
    
    validation_result = qa_system.validate_prediction(normal_emotions, input_data)
    
    print(f"✅ Normal validation completed")
    print(f"   - Is valid: {validation_result.is_valid}")
    print(f"   - Validation score: {validation_result.validation_score:.3f}")
    print(f"   - Failed checks: {len(validation_result.failed_checks)}")
    print(f"   - Warnings: {len(validation_result.warnings)}")
    print(f"   - Recommendations: {len(validation_result.recommendations)}")
    
    # Test biased validation
    print("\n1.2 Testing Biased Validation...")
    biased_emotions = create_biased_emotions()
    
    biased_validation = qa_system.validate_prediction(biased_emotions, input_data)
    
    print(f"✅ Biased validation completed")
    print(f"   - Is valid: {biased_validation.is_valid}")
    print(f"   - Validation score: {biased_validation.validation_score:.3f}")
    print(f"   - Failed checks: {biased_validation.failed_checks}")
    print(f"   - Warnings: {biased_validation.warnings}")
    
    # Test inconsistent validation
    print("\n1.3 Testing Inconsistent Validation...")
    inconsistent_emotions = create_inconsistent_emotions()
    
    inconsistent_validation = qa_system.validate_prediction(inconsistent_emotions, input_data)
    
    print(f"✅ Inconsistent validation completed")
    print(f"   - Is valid: {inconsistent_validation.is_valid}")
    print(f"   - Validation score: {inconsistent_validation.validation_score:.3f}")
    print(f"   - Failed checks: {inconsistent_validation.failed_checks}")
    
    # Test anomaly detection
    print("\n2. Testing Anomaly Detection...")
    
    # Create multiple result sets for anomaly detection
    normal_results = [create_normal_emotions() for _ in range(10)]
    
    # Add some anomalous results
    anomalous_results = normal_results.copy()
    anomalous_results.extend([
        [EmotionScore("extreme_joy", 1.0, 1.0, "text", "model1", 0.1, {})],  # Unusual emotion
        [EmotionScore("joy", 0.01, 0.01, "text", "model1", 0.1, {})],  # Very low scores
        create_biased_emotions()  # Biased result
    ])
    
    anomaly_result = qa_system.detect_anomalies(anomalous_results)
    
    print(f"✅ Anomaly detection completed")
    print(f"   - Anomalies detected: {anomaly_result.anomalies_detected}")
    print(f"   - Anomaly types: {set(anomaly_result.anomaly_types)}")
    print(f"   - Severity levels: {set(anomaly_result.severity_levels)}")
    print(f"   - Investigation needed: {anomaly_result.investigation_needed}")
    print(f"   - Details: {list(anomaly_result.details.keys())}")
    
    # Test quality report
    print("\n3. Testing Quality Report...")
    
    # Generate multiple validations to build history
    for i in range(20):
        if i % 3 == 0:
            emotions = create_biased_emotions()
        elif i % 3 == 1:
            emotions = create_inconsistent_emotions()
        else:
            emotions = create_normal_emotions()
        
        qa_system.validate_prediction(emotions, input_data)
    
    quality_report = qa_system.get_quality_report(timedelta(days=1))
    
    print(f"✅ Quality report generated")
    print(f"   - Total validations: {quality_report['total_validations']}")
    print(f"   - Validation rate: {quality_report['validation_rate']:.3f}")
    print(f"   - Average score: {quality_report['average_validation_score']:.3f}")
    print(f"   - Quality grade: {quality_report['quality_grade']}")
    print(f"   - Trend: {quality_report['trend']}")
    print(f"   - Recommendations: {len(quality_report['recommendations'])}")
    
    for rec in quality_report['recommendations']:
        print(f"     • {rec}")
    
    # Test individual validators
    print("\n4. Testing Individual Validators...")
    
    from utils.quality_assurance import ConsistencyValidator, BiasDetector, ConfidenceValidator, OutlierDetector
    
    # Test Consistency Validator
    print("\n4.1 Testing Consistency Validator...")
    consistency_validator = ConsistencyValidator()
    
    consistency_result = consistency_validator.validate(normal_emotions, input_data)
    print(f"✅ Consistency validation: score={consistency_result['score']:.3f}, valid={consistency_result['is_valid']}")
    
    inconsistency_result = consistency_validator.validate(inconsistent_emotions, input_data)
    print(f"✅ Inconsistency validation: score={inconsistency_result['score']:.3f}, valid={inconsistency_result['is_valid']}")
    
    # Test Bias Detector
    print("\n4.2 Testing Bias Detector...")
    bias_detector = BiasDetector()
    
    normal_bias_result = bias_detector.validate(normal_emotions, input_data)
    print(f"✅ Normal bias check: score={normal_bias_result['score']:.3f}, valid={normal_bias_result['is_valid']}")
    
    biased_bias_result = bias_detector.validate(biased_emotions, input_data)
    print(f"✅ Biased bias check: score={biased_bias_result['score']:.3f}, valid={biased_bias_result['is_valid']}")
    
    # Test Confidence Validator
    print("\n4.3 Testing Confidence Validator...")
    confidence_validator = ConfidenceValidator()
    
    conf_result = confidence_validator.validate(normal_emotions, input_data)
    print(f"✅ Confidence validation: score={conf_result['score']:.3f}, valid={conf_result['is_valid']}")
    
    # Test Outlier Detector
    print("\n4.4 Testing Outlier Detector...")
    outlier_detector = OutlierDetector()
    
    # Create data with outliers
    outlier_emotions = normal_emotions + [
        EmotionScore("outlier", 10.0, 0.5, "text", "model1", 0.1, {}),  # Score outlier
        EmotionScore("normal", 0.5, -0.5, "text", "model1", 0.1, {})   # Confidence outlier
    ]
    
    outlier_result = outlier_detector.validate(outlier_emotions, input_data)
    print(f"✅ Outlier detection: score={outlier_result['score']:.3f}, valid={outlier_result['is_valid']}")
    print(f"   - Warnings: {outlier_result.get('warnings', [])}")
    
    # Test edge cases
    print("\n5. Testing Edge Cases...")
    
    # Empty predictions
    empty_validation = qa_system.validate_prediction([], input_data)
    print(f"✅ Empty predictions handled: valid={empty_validation.is_valid}")
    
    # Single prediction
    single_emotion = [EmotionScore("joy", 0.8, 0.7, "text", "model1", 0.1, {})]
    single_validation = qa_system.validate_prediction(single_emotion, input_data)
    print(f"✅ Single prediction handled: valid={single_validation.is_valid}")
    
    # Very large dataset
    large_emotions = [create_normal_emotions()[0] for _ in range(100)]
    large_validation = qa_system.validate_prediction(large_emotions, input_data)
    print(f"✅ Large dataset handled: valid={large_validation.is_valid}")
    
    print("\n🎉 Quality assurance tests completed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()