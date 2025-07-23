# utils/quality_assurance.py - Quality assurance and validation system
import numpy as np
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, deque
from utils.text_emotion_ensemble import EmotionScore
import scipy.stats as stats

@dataclass
class ValidationResult:
    """Result of validation checks"""
    is_valid: bool
    validation_score: float
    failed_checks: List[str]
    warnings: List[str]
    recommendations: List[str]
    metadata: Dict[str, Any]

@dataclass
class BiasDetectionResult:
    """Result of bias detection analysis"""
    bias_detected: bool
    bias_score: float
    bias_types: List[str]
    affected_groups: List[str]
    mitigation_suggestions: List[str]
    detailed_analysis: Dict[str, Any]

@dataclass
class AnomalyDetectionResult:
    """Result of anomaly detection"""
    anomalies_detected: int
    anomaly_scores: List[float]
    anomaly_types: List[str]
    severity_levels: List[str]
    investigation_needed: bool
    details: Dict[str, Any]

class QualityAssurance:
    """Comprehensive quality assurance system for emotion predictions"""
    
    def __init__(self):
        self.validators = [
            ConsistencyValidator(),
            BiasDetector(),
            ConfidenceValidator(),
            OutlierDetector(),
            PerformanceValidator()
        ]
        self.validation_history = deque(maxlen=5000)
        self.quality_thresholds = {
            'minimum_confidence': 0.3,
            'maximum_variance': 0.5,
            'consistency_threshold': 0.7,
            'bias_threshold': 0.3,
            'anomaly_threshold': 0.8
        }
    
    def validate_prediction(self, result: List[EmotionScore], 
                          input_data: Dict, context: Dict = None) -> ValidationResult:
        """Comprehensive validation of emotion predictions"""
        
        if not result:
            return ValidationResult(
                is_valid=False,
                validation_score=0.0,
                failed_checks=['empty_prediction'],
                warnings=['No emotion predictions provided'],
                recommendations=['Check input processing pipeline'],
                metadata={'reason': 'empty_prediction'}
            )
        
        failed_checks = []
        warnings = []
        recommendations = []
        validation_scores = []
        
        # Run all validators
        for validator in self.validators:
            try:
                validator_result = validator.validate(result, input_data, context)
                
                if not validator_result['is_valid']:
                    failed_checks.extend(validator_result.get('failed_checks', []))
                
                warnings.extend(validator_result.get('warnings', []))
                recommendations.extend(validator_result.get('recommendations', []))
                validation_scores.append(validator_result.get('score', 0.5))
                
            except Exception as e:
                logging.error(f"Validator {validator.__class__.__name__} failed: {e}")
                failed_checks.append(f"validator_error_{validator.__class__.__name__}")
        
        # Calculate overall validation score
        overall_score = np.mean(validation_scores) if validation_scores else 0.0
        is_valid = len(failed_checks) == 0 and overall_score >= 0.6
        
        # Create validation result
        validation_result = ValidationResult(
            is_valid=is_valid,
            validation_score=overall_score,
            failed_checks=failed_checks,
            warnings=warnings,
            recommendations=list(set(recommendations)),  # Remove duplicates
            metadata={
                'validation_timestamp': datetime.now(),
                'validator_count': len(self.validators),
                'input_type': input_data.get('type', 'unknown'),
                'context': context or {}
            }
        )
        
        # Update validation history
        self._update_validation_history(validation_result, result, input_data)
        
        return validation_result
    
    def detect_anomalies(self, results: List[List[EmotionScore]], 
                        time_window: timedelta = timedelta(hours=1)) -> AnomalyDetectionResult:
        """Detect unusual patterns in emotion predictions"""
        
        if not results:
            return AnomalyDetectionResult(
                anomalies_detected=0,
                anomaly_scores=[],
                anomaly_types=[],
                severity_levels=[],
                investigation_needed=False,
                details={'reason': 'no_data'}
            )
        
        anomalies = []
        anomaly_scores = []
        anomaly_types = []
        severity_levels = []
        
        # Statistical anomaly detection
        statistical_anomalies = self._detect_statistical_anomalies(results)
        anomalies.extend(statistical_anomalies)
        
        # Pattern-based anomaly detection
        pattern_anomalies = self._detect_pattern_anomalies(results)
        anomalies.extend(pattern_anomalies)
        
        # Temporal anomaly detection
        temporal_anomalies = self._detect_temporal_anomalies(results, time_window)
        anomalies.extend(temporal_anomalies)
        
        # Process detected anomalies
        for anomaly in anomalies:
            anomaly_scores.append(anomaly.get('score', 0.5))
            anomaly_types.append(anomaly.get('type', 'unknown'))
            severity_levels.append(anomaly.get('severity', 'medium'))
        
        investigation_needed = (
            len(anomalies) > 5 or 
            any(score > self.quality_thresholds['anomaly_threshold'] for score in anomaly_scores)
        )
        
        return AnomalyDetectionResult(
            anomalies_detected=len(anomalies),
            anomaly_scores=anomaly_scores,
            anomaly_types=anomaly_types,
            severity_levels=severity_levels,
            investigation_needed=investigation_needed,
            details={
                'statistical_anomalies': len(statistical_anomalies),
                'pattern_anomalies': len(pattern_anomalies),
                'temporal_anomalies': len(temporal_anomalies),
                'detection_timestamp': datetime.now()
            }
        )
    
    def _detect_statistical_anomalies(self, results: List[List[EmotionScore]]) -> List[Dict]:
        """Detect statistical anomalies in predictions"""
        anomalies = []
        
        # Collect all scores and confidences
        all_scores = []
        all_confidences = []
        
        for result in results:
            for emotion in result:
                all_scores.append(emotion.score)
                all_confidences.append(emotion.confidence)
        
        if len(all_scores) < 10:
            return anomalies  # Need minimum data for statistical analysis
        
        # Z-score based anomaly detection
        score_mean = np.mean(all_scores)
        score_std = np.std(all_scores)
        
        conf_mean = np.mean(all_confidences)
        conf_std = np.std(all_confidences)
        
        for i, result in enumerate(results):
            for emotion in result:
                # Check for score anomalies
                if score_std > 0:
                    score_z = abs(emotion.score - score_mean) / score_std
                    if score_z > 3:  # 3-sigma rule
                        anomalies.append({
                            'type': 'statistical_score',
                            'score': min(score_z / 3, 1.0),
                            'severity': 'high' if score_z > 4 else 'medium',
                            'details': f'Score z-score: {score_z:.2f}',
                            'result_index': i,
                            'emotion': emotion.label
                        })
                
                # Check for confidence anomalies
                if conf_std > 0:
                    conf_z = abs(emotion.confidence - conf_mean) / conf_std
                    if conf_z > 3:
                        anomalies.append({
                            'type': 'statistical_confidence',
                            'score': min(conf_z / 3, 1.0),
                            'severity': 'medium',
                            'details': f'Confidence z-score: {conf_z:.2f}',
                            'result_index': i,
                            'emotion': emotion.label
                        })
        
        return anomalies
    
    def _detect_pattern_anomalies(self, results: List[List[EmotionScore]]) -> List[Dict]:
        """Detect pattern-based anomalies"""
        anomalies = []
        
        # Check for unusual emotion distributions
        emotion_counts = defaultdict(int)
        for result in results:
            for emotion in result:
                emotion_counts[emotion.label] += 1
        
        total_emotions = sum(emotion_counts.values())
        if total_emotions == 0:
            return anomalies
        
        # Detect unusually rare or common emotions
        for emotion, count in emotion_counts.items():
            frequency = count / total_emotions
            
            # Very rare emotions (< 1%)
            if frequency < 0.01 and count > 0:
                anomalies.append({
                    'type': 'rare_emotion',
                    'score': 1.0 - frequency * 100,  # Higher score for rarer emotions
                    'severity': 'low',
                    'details': f'Emotion {emotion} appears only {frequency:.1%} of the time',
                    'emotion': emotion
                })
            
            # Overly dominant emotions (> 80%)
            elif frequency > 0.8:
                anomalies.append({
                    'type': 'dominant_emotion',
                    'score': frequency,
                    'severity': 'medium',
                    'details': f'Emotion {emotion} dominates with {frequency:.1%} frequency',
                    'emotion': emotion
                })
        
        return anomalies
    
    def _detect_temporal_anomalies(self, results: List[List[EmotionScore]], 
                                  time_window: timedelta) -> List[Dict]:
        """Detect temporal anomalies in predictions"""
        anomalies = []
        
        # This would be enhanced with actual timestamps in production
        # For now, simulate temporal analysis
        
        if len(results) < 5:
            return anomalies
        
        # Check for sudden changes in prediction patterns
        window_size = min(5, len(results) // 2)
        
        for i in range(window_size, len(results)):
            recent_window = results[i-window_size:i]
            current_result = results[i]
            
            # Calculate average scores in recent window
            recent_scores = []
            for result in recent_window:
                for emotion in result:
                    recent_scores.append(emotion.score)
            
            current_scores = [emotion.score for emotion in current_result]
            
            if recent_scores and current_scores:
                recent_mean = np.mean(recent_scores)
                current_mean = np.mean(current_scores)
                
                # Detect sudden jumps
                if abs(current_mean - recent_mean) > 0.5:
                    anomalies.append({
                        'type': 'temporal_jump',
                        'score': abs(current_mean - recent_mean),
                        'severity': 'high' if abs(current_mean - recent_mean) > 0.7 else 'medium',
                        'details': f'Score jump from {recent_mean:.2f} to {current_mean:.2f}',
                        'result_index': i
                    })
        
        return anomalies
    
    def _update_validation_history(self, validation_result: ValidationResult, 
                                  emotion_result: List[EmotionScore], 
                                  input_data: Dict):
        """Update validation history for trend analysis"""
        
        history_entry = {
            'timestamp': datetime.now(),
            'is_valid': validation_result.is_valid,
            'validation_score': validation_result.validation_score,
            'failed_checks_count': len(validation_result.failed_checks),
            'warnings_count': len(validation_result.warnings),
            'emotion_count': len(emotion_result),
            'top_emotion': emotion_result[0].label if emotion_result else 'none',
            'input_type': input_data.get('type', 'unknown')
        }
        
        self.validation_history.append(history_entry)
    
    def get_quality_report(self, time_period: timedelta = timedelta(days=7)) -> Dict[str, Any]:
        """Generate comprehensive quality report"""
        
        cutoff_time = datetime.now() - time_period
        recent_history = [
            entry for entry in self.validation_history 
            if entry['timestamp'] > cutoff_time
        ]
        
        if not recent_history:
            return {'error': 'No recent validation data available'}
        
        # Calculate quality metrics
        total_validations = len(recent_history)
        valid_predictions = sum(1 for entry in recent_history if entry['is_valid'])
        validation_rate = valid_predictions / total_validations
        
        avg_validation_score = np.mean([entry['validation_score'] for entry in recent_history])
        
        # Failed checks analysis
        all_failed_checks = []
        for entry in recent_history:
            all_failed_checks.extend([f"check_{i}" for i in range(entry['failed_checks_count'])])
        
        failed_checks_frequency = defaultdict(int)
        for check in all_failed_checks:
            failed_checks_frequency[check] += 1
        
        # Trend analysis
        scores_over_time = [entry['validation_score'] for entry in recent_history[-50:]]
        trend = 'stable'
        if len(scores_over_time) > 10:
            recent_avg = np.mean(scores_over_time[-10:])
            earlier_avg = np.mean(scores_over_time[:10])
            
            if recent_avg > earlier_avg + 0.1:
                trend = 'improving'
            elif recent_avg < earlier_avg - 0.1:
                trend = 'declining'
        
        return {
            'period': str(time_period),
            'total_validations': total_validations,
            'validation_rate': validation_rate,
            'average_validation_score': avg_validation_score,
            'trend': trend,
            'most_common_failures': dict(list(failed_checks_frequency.items())[:5]),
            'quality_grade': self._calculate_quality_grade(validation_rate, avg_validation_score),
            'recommendations': self._generate_quality_recommendations(
                validation_rate, avg_validation_score, failed_checks_frequency
            )
        }
    
    def _calculate_quality_grade(self, validation_rate: float, avg_score: float) -> str:
        """Calculate overall quality grade"""
        combined_score = (validation_rate + avg_score) / 2
        
        if combined_score >= 0.9:
            return 'A'
        elif combined_score >= 0.8:
            return 'B'
        elif combined_score >= 0.7:
            return 'C'
        elif combined_score >= 0.6:
            return 'D'
        else:
            return 'F'
    
    def _generate_quality_recommendations(self, validation_rate: float, 
                                        avg_score: float, 
                                        failed_checks: Dict) -> List[str]:
        """Generate recommendations for quality improvement"""
        recommendations = []
        
        if validation_rate < 0.8:
            recommendations.append("Validation rate is low - review input preprocessing")
        
        if avg_score < 0.7:
            recommendations.append("Average validation score is low - consider model retraining")
        
        if 'consistency' in str(failed_checks):
            recommendations.append("Consistency issues detected - check ensemble weighting")
        
        if 'bias' in str(failed_checks):
            recommendations.append("Bias detected - implement bias mitigation strategies")
        
        if 'confidence' in str(failed_checks):
            recommendations.append("Confidence calibration may be needed")
        
        return recommendations

class ConsistencyValidator:
    """Validator for prediction consistency"""
    
    def validate(self, result: List[EmotionScore], input_data: Dict, 
                context: Dict = None) -> Dict[str, Any]:
        """Validate consistency of predictions"""
        
        if len(result) < 2:
            return {
                'is_valid': True,
                'score': 1.0,
                'warnings': ['Single prediction - consistency check skipped']
            }
        
        # Check score-confidence consistency
        score_conf_consistency = self._check_score_confidence_consistency(result)
        
        # Check temporal consistency (if applicable)
        temporal_consistency = self._check_temporal_consistency(result, context)
        
        # Check ensemble consistency (if applicable)
        ensemble_consistency = self._check_ensemble_consistency(result)
        
        overall_consistency = np.mean([
            score_conf_consistency, temporal_consistency, ensemble_consistency
        ])
        
        is_valid = overall_consistency >= 0.7
        failed_checks = []
        warnings = []
        recommendations = []
        
        if score_conf_consistency < 0.7:
            failed_checks.append('score_confidence_inconsistency')
            recommendations.append('Review confidence calibration')
        
        if temporal_consistency < 0.7:
            warnings.append('Temporal inconsistency detected')
        
        if ensemble_consistency < 0.7:
            warnings.append('Ensemble predictions show inconsistency')
        
        return {
            'is_valid': is_valid,
            'score': overall_consistency,
            'failed_checks': failed_checks,
            'warnings': warnings,
            'recommendations': recommendations
        }
    
    def _check_score_confidence_consistency(self, result: List[EmotionScore]) -> float:
        """Check consistency between scores and confidences"""
        differences = [abs(emotion.score - emotion.confidence) for emotion in result]
        avg_difference = np.mean(differences)
        return max(0.0, 1.0 - avg_difference * 2)
    
    def _check_temporal_consistency(self, result: List[EmotionScore], 
                                   context: Dict = None) -> float:
        """Check temporal consistency of predictions"""
        # Simplified temporal consistency check
        processing_times = [emotion.processing_time for emotion in result]
        if len(set(processing_times)) == 1:
            return 1.0  # All same processing time
        
        time_variance = np.var(processing_times)
        return max(0.0, 1.0 - time_variance)
    
    def _check_ensemble_consistency(self, result: List[EmotionScore]) -> float:
        """Check consistency across ensemble predictions"""
        # Check if predictions come from ensemble
        model_names = [emotion.model_name for emotion in result]
        if 'ensemble' not in str(model_names).lower():
            return 1.0  # Not ensemble, skip check
        
        # Check score distribution consistency
        scores = [emotion.score for emotion in result]
        score_variance = np.var(scores)
        return max(0.0, 1.0 - score_variance)

class BiasDetector:
    """Detector for various types of bias in predictions"""
    
    def validate(self, result: List[EmotionScore], input_data: Dict, 
                context: Dict = None) -> Dict[str, Any]:
        """Detect bias in emotion predictions"""
        
        bias_score = 0.0
        failed_checks = []
        warnings = []
        recommendations = []
        
        # Check for emotion bias
        emotion_bias = self._detect_emotion_bias(result)
        if emotion_bias > 0.3:
            failed_checks.append('emotion_bias')
            recommendations.append('Review training data for emotion balance')
        
        # Check for confidence bias
        confidence_bias = self._detect_confidence_bias(result)
        if confidence_bias > 0.3:
            warnings.append('Confidence bias detected')
        
        # Check for model bias
        model_bias = self._detect_model_bias(result)
        if model_bias > 0.3:
            warnings.append('Model bias detected')
        
        overall_bias = max(emotion_bias, confidence_bias, model_bias)
        is_valid = overall_bias < 0.3
        
        return {
            'is_valid': is_valid,
            'score': 1.0 - overall_bias,
            'failed_checks': failed_checks,
            'warnings': warnings,
            'recommendations': recommendations
        }
    
    def _detect_emotion_bias(self, result: List[EmotionScore]) -> float:
        """Detect bias towards certain emotions"""
        if not result:
            return 0.0
        
        # Check if one emotion dominates
        emotion_scores = {}
        for emotion in result:
            emotion_scores[emotion.label] = emotion_scores.get(emotion.label, 0) + emotion.score
        
        if not emotion_scores:
            return 0.0
        
        total_score = sum(emotion_scores.values())
        max_emotion_score = max(emotion_scores.values())
        
        # Bias if one emotion has > 80% of total score
        bias_ratio = max_emotion_score / total_score if total_score > 0 else 0
        return max(0.0, bias_ratio - 0.8) * 5  # Scale to [0,1]
    
    def _detect_confidence_bias(self, result: List[EmotionScore]) -> float:
        """Detect bias in confidence scores"""
        confidences = [emotion.confidence for emotion in result]
        if not confidences:
            return 0.0
        
        # Check for overconfidence (all confidences > 0.9)
        if all(c > 0.9 for c in confidences):
            return 0.8
        
        # Check for underconfidence (all confidences < 0.3)
        if all(c < 0.3 for c in confidences):
            return 0.6
        
        return 0.0
    
    def _detect_model_bias(self, result: List[EmotionScore]) -> float:
        """Detect bias from specific models"""
        model_names = [emotion.model_name for emotion in result]
        if len(set(model_names)) == 1:
            return 0.2  # Slight bias if all from same model
        
        return 0.0

class ConfidenceValidator:
    """Validator for confidence scores"""
    
    def validate(self, result: List[EmotionScore], input_data: Dict, 
                context: Dict = None) -> Dict[str, Any]:
        """Validate confidence scores"""
        
        confidences = [emotion.confidence for emotion in result]
        if not confidences:
            return {'is_valid': False, 'score': 0.0, 'failed_checks': ['no_confidences']}
        
        failed_checks = []
        warnings = []
        recommendations = []
        
        # Check confidence range
        if any(c < 0 or c > 1 for c in confidences):
            failed_checks.append('confidence_out_of_range')
        
        # Check for reasonable confidence distribution
        avg_confidence = np.mean(confidences)
        if avg_confidence < 0.1:
            warnings.append('Very low average confidence')
            recommendations.append('Check model calibration')
        elif avg_confidence > 0.95:
            warnings.append('Very high average confidence - possible overconfidence')
            recommendations.append('Apply confidence calibration')
        
        # Check confidence variance
        conf_variance = np.var(confidences)
        if conf_variance < 0.01:
            warnings.append('Low confidence variance - predictions may be too uniform')
        
        is_valid = len(failed_checks) == 0
        score = max(0.0, 1.0 - len(failed_checks) * 0.5 - len(warnings) * 0.1)
        
        return {
            'is_valid': is_valid,
            'score': score,
            'failed_checks': failed_checks,
            'warnings': warnings,
            'recommendations': recommendations
        }

class OutlierDetector:
    """Detector for outlier predictions"""
    
    def validate(self, result: List[EmotionScore], input_data: Dict, 
                context: Dict = None) -> Dict[str, Any]:
        """Detect outlier predictions"""
        
        if len(result) < 3:
            return {'is_valid': True, 'score': 1.0}
        
        scores = [emotion.score for emotion in result]
        confidences = [emotion.confidence for emotion in result]
        
        # Detect score outliers using IQR method
        score_outliers = self._detect_iqr_outliers(scores)
        conf_outliers = self._detect_iqr_outliers(confidences)
        
        total_outliers = len(score_outliers) + len(conf_outliers)
        outlier_ratio = total_outliers / (len(scores) + len(confidences))
        
        warnings = []
        if score_outliers:
            warnings.append(f'{len(score_outliers)} score outliers detected')
        if conf_outliers:
            warnings.append(f'{len(conf_outliers)} confidence outliers detected')
        
        is_valid = outlier_ratio < 0.2  # Less than 20% outliers
        score = max(0.0, 1.0 - outlier_ratio * 2)
        
        return {
            'is_valid': is_valid,
            'score': score,
            'warnings': warnings,
            'recommendations': ['Investigate outlier predictions'] if not is_valid else []
        }
    
    def _detect_iqr_outliers(self, values: List[float]) -> List[int]:
        """Detect outliers using Interquartile Range method"""
        if len(values) < 4:
            return []
        
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = []
        for i, value in enumerate(values):
            if value < lower_bound or value > upper_bound:
                outliers.append(i)
        
        return outliers

class PerformanceValidator:
    """Validator for performance metrics"""
    
    def validate(self, result: List[EmotionScore], input_data: Dict, 
                context: Dict = None) -> Dict[str, Any]:
        """Validate performance characteristics"""
        
        processing_times = [emotion.processing_time for emotion in result]
        
        warnings = []
        recommendations = []
        
        # Check processing time
        avg_processing_time = np.mean(processing_times)
        if avg_processing_time > 5.0:
            warnings.append('High processing time detected')
            recommendations.append('Optimize model inference speed')
        
        # Check processing time consistency
        time_variance = np.var(processing_times)
        if time_variance > 1.0:
            warnings.append('Inconsistent processing times')
        
        # Performance score based on speed and consistency
        speed_score = max(0.0, 1.0 - avg_processing_time / 10.0)
        consistency_score = max(0.0, 1.0 - time_variance)
        performance_score = (speed_score + consistency_score) / 2
        
        return {
            'is_valid': performance_score > 0.5,
            'score': performance_score,
            'warnings': warnings,
            'recommendations': recommendations
        }