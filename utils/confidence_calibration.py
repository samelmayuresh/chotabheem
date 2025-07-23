# utils/confidence_calibration.py - Confidence calibration and quality assurance
import numpy as np
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from collections import defaultdict, deque
from utils.text_emotion_ensemble import EmotionScore
import scipy.stats as stats
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve

@dataclass
class CalibrationMetrics:
    """Metrics for confidence calibration assessment"""
    reliability_score: float
    calibration_error: float
    sharpness: float
    brier_score: float
    expected_calibration_error: float
    maximum_calibration_error: float
    confidence_histogram: Dict[str, int]
    accuracy_by_confidence: Dict[str, float]

@dataclass
class ReliabilityAssessment:
    """Assessment of prediction reliability"""
    overall_reliability: float
    confidence_level: str
    reliability_factors: Dict[str, float]
    uncertainty_quantification: Dict[str, float]
    recommendations: List[str]
    metadata: Dict[str, Any]

class ConfidenceCalibrator:
    """Advanced confidence calibration with uncertainty quantification"""
    
    def __init__(self, calibration_method: str = "isotonic"):
        self.calibration_method = calibration_method
        self.calibration_models = {}
        self.calibration_history = deque(maxlen=10000)
        self.performance_history = deque(maxlen=1000)
        self.reliability_thresholds = {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        }
        
        # Initialize calibration models
        self._initialize_calibration_models()
    
    def _initialize_calibration_models(self):
        """Initialize calibration models for different scenarios"""
        self.calibration_models = {
            'text': IsotonicRegression(out_of_bounds='clip'),
            'voice': IsotonicRegression(out_of_bounds='clip'),
            'multimodal': IsotonicRegression(out_of_bounds='clip'),
            'ensemble': IsotonicRegression(out_of_bounds='clip')
        }
        
        # Track if models are fitted
        self.models_fitted = {key: False for key in self.calibration_models.keys()}
    
    def calibrate_confidence(self, emotion_scores: List[EmotionScore], 
                           ground_truth: List[str] = None,
                           context: Dict = None) -> List[EmotionScore]:
        """Calibrate confidence scores for emotion predictions"""
        
        if not emotion_scores:
            return emotion_scores
        
        calibrated_scores = []
        
        for emotion_score in emotion_scores:
            # Determine calibration model to use
            model_key = self._select_calibration_model(emotion_score, context)
            
            # Get calibrated confidence
            calibrated_confidence = self._apply_calibration(
                emotion_score.confidence, 
                model_key,
                emotion_score
            )
            
            # Create calibrated emotion score
            calibrated_score = EmotionScore(
                label=emotion_score.label,
                score=emotion_score.score,
                confidence=calibrated_confidence,
                source=emotion_score.source,
                model_name=emotion_score.model_name,
                processing_time=emotion_score.processing_time,
                metadata={
                    **emotion_score.metadata,
                    'original_confidence': emotion_score.confidence,
                    'calibration_applied': True,
                    'calibration_model': model_key,
                    'calibration_method': self.calibration_method
                }
            )
            
            calibrated_scores.append(calibrated_score)
        
        # Update calibration history if ground truth available
        if ground_truth:
            self._update_calibration_history(emotion_scores, ground_truth, context)
        
        return calibrated_scores
    
    def _select_calibration_model(self, emotion_score: EmotionScore, 
                                 context: Dict = None) -> str:
        """Select appropriate calibration model based on emotion score characteristics"""
        
        # Primary selection based on source
        if emotion_score.source == 'multimodal':
            return 'multimodal'
        elif emotion_score.source == 'text':
            return 'text'
        elif emotion_score.source == 'voice':
            return 'voice'
        
        # Secondary selection based on model name
        if 'ensemble' in emotion_score.model_name.lower():
            return 'ensemble'
        
        # Default fallback
        return 'ensemble'
    
    def _apply_calibration(self, confidence: float, model_key: str, 
                          emotion_score: EmotionScore) -> float:
        """Apply calibration to confidence score"""
        
        # If model not fitted, apply heuristic calibration
        if not self.models_fitted.get(model_key, False):
            return self._heuristic_calibration(confidence, emotion_score)
        
        try:
            # Apply learned calibration
            calibrated = self.calibration_models[model_key].predict([confidence])[0]
            
            # Ensure valid range
            calibrated = max(0.01, min(0.99, calibrated))
            
            return calibrated
            
        except Exception as e:
            logging.warning(f"Calibration failed for {model_key}: {e}")
            return self._heuristic_calibration(confidence, emotion_score)
    
    def _heuristic_calibration(self, confidence: float, emotion_score: EmotionScore) -> float:
        """Apply heuristic calibration when learned models not available"""
        
        # Base calibration adjustments
        calibrated = confidence
        
        # Adjust based on model type
        model_type = emotion_score.metadata.get('model_type', 'unknown')
        
        if model_type == 'ensemble':
            # Ensemble models tend to be overconfident
            calibrated *= 0.9
        elif model_type == 'single':
            # Single models may be underconfident
            calibrated *= 1.1
        
        # Adjust based on score-confidence consistency
        score_confidence_diff = abs(emotion_score.score - confidence)
        if score_confidence_diff > 0.3:
            # Large difference suggests miscalibration
            calibrated *= 0.8
        
        # Adjust based on processing complexity
        processing_time = emotion_score.processing_time
        if processing_time > 1.0:
            # Longer processing might indicate more reliable prediction
            calibrated *= 1.05
        elif processing_time < 0.1:
            # Very fast processing might be less reliable
            calibrated *= 0.95
        
        # Ensure valid range
        return max(0.01, min(0.99, calibrated))
    
    def _update_calibration_history(self, emotion_scores: List[EmotionScore], 
                                   ground_truth: List[str], context: Dict = None):
        """Update calibration history with new data"""
        
        for emotion_score in emotion_scores:
            # Check if prediction was correct
            is_correct = emotion_score.label in ground_truth
            
            history_entry = {
                'timestamp': datetime.now(),
                'confidence': emotion_score.confidence,
                'score': emotion_score.score,
                'predicted_label': emotion_score.label,
                'is_correct': is_correct,
                'source': emotion_score.source,
                'model_name': emotion_score.model_name,
                'context': context or {}
            }
            
            self.calibration_history.append(history_entry)
        
        # Retrain calibration models periodically
        if len(self.calibration_history) % 100 == 0:
            self._retrain_calibration_models()
    
    def _retrain_calibration_models(self):
        """Retrain calibration models with accumulated data"""
        
        if len(self.calibration_history) < 50:
            return  # Need minimum data for training
        
        # Group data by source/model type
        data_by_model = defaultdict(list)
        
        for entry in self.calibration_history:
            model_key = entry['source']
            if 'ensemble' in entry['model_name'].lower():
                model_key = 'ensemble'
            
            data_by_model[model_key].append(entry)
        
        # Train each calibration model
        for model_key, data in data_by_model.items():
            if len(data) < 20:
                continue  # Need minimum data per model
            
            try:
                # Prepare training data
                confidences = [entry['confidence'] for entry in data]
                accuracies = [float(entry['is_correct']) for entry in data]
                
                # Fit calibration model
                self.calibration_models[model_key].fit(confidences, accuracies)
                self.models_fitted[model_key] = True
                
                logging.info(f"Retrained calibration model for {model_key} with {len(data)} samples")
                
            except Exception as e:
                logging.error(f"Failed to retrain calibration model for {model_key}: {e}")
    
    def assess_reliability(self, emotion_scores: List[EmotionScore], 
                          context: Dict = None) -> ReliabilityAssessment:
        """Assess overall reliability of emotion predictions"""
        
        if not emotion_scores:
            return ReliabilityAssessment(
                overall_reliability=0.0,
                confidence_level='low',
                reliability_factors={},
                uncertainty_quantification={},
                recommendations=['No predictions to assess'],
                metadata={'reason': 'no_predictions'}
            )
        
        # Calculate reliability factors
        reliability_factors = self._calculate_reliability_factors(emotion_scores, context)
        
        # Calculate uncertainty quantification
        uncertainty_metrics = self._quantify_uncertainty(emotion_scores)
        
        # Calculate overall reliability
        overall_reliability = self._calculate_overall_reliability(reliability_factors)
        
        # Determine confidence level
        confidence_level = self._determine_confidence_level(overall_reliability)
        
        # Generate recommendations
        recommendations = self._generate_reliability_recommendations(
            reliability_factors, uncertainty_metrics, overall_reliability
        )
        
        return ReliabilityAssessment(
            overall_reliability=overall_reliability,
            confidence_level=confidence_level,
            reliability_factors=reliability_factors,
            uncertainty_quantification=uncertainty_metrics,
            recommendations=recommendations,
            metadata={
                'assessment_timestamp': datetime.now(),
                'emotion_count': len(emotion_scores),
                'context': context or {}
            }
        )
    
    def _calculate_reliability_factors(self, emotion_scores: List[EmotionScore], 
                                     context: Dict = None) -> Dict[str, float]:
        """Calculate various factors that contribute to reliability"""
        
        factors = {}
        
        # Confidence consistency
        confidences = [e.confidence for e in emotion_scores]
        factors['confidence_mean'] = np.mean(confidences)
        factors['confidence_std'] = np.std(confidences)
        factors['confidence_consistency'] = max(0.0, 1.0 - factors['confidence_std'])
        
        # Score-confidence alignment
        score_conf_diffs = [abs(e.score - e.confidence) for e in emotion_scores]
        factors['score_confidence_alignment'] = max(0.0, 1.0 - np.mean(score_conf_diffs))
        
        # Model diversity (if ensemble)
        model_names = [e.model_name for e in emotion_scores]
        unique_models = len(set(model_names))
        factors['model_diversity'] = min(1.0, unique_models / max(len(model_names), 1))
        
        # Processing time consistency
        processing_times = [e.processing_time for e in emotion_scores]
        factors['processing_time_mean'] = np.mean(processing_times)
        factors['processing_stability'] = max(0.0, 1.0 - np.std(processing_times))
        
        # Source reliability
        sources = [e.source for e in emotion_scores]
        source_reliability = {
            'multimodal': 0.9,
            'ensemble': 0.8,
            'text': 0.7,
            'voice': 0.7,
            'single': 0.6
        }
        factors['source_reliability'] = np.mean([
            source_reliability.get(source, 0.5) for source in sources
        ])
        
        # Historical performance (if available)
        factors['historical_performance'] = self._get_historical_performance()
        
        return factors
    
    def _quantify_uncertainty(self, emotion_scores: List[EmotionScore]) -> Dict[str, float]:
        """Quantify various types of uncertainty"""
        
        uncertainty = {}
        
        # Aleatoric uncertainty (inherent randomness)
        scores = [e.score for e in emotion_scores]
        uncertainty['aleatoric'] = np.var(scores) if len(scores) > 1 else 0.0
        
        # Epistemic uncertainty (model uncertainty)
        confidences = [e.confidence for e in emotion_scores]
        uncertainty['epistemic'] = 1.0 - np.mean(confidences)
        
        # Prediction entropy
        if len(scores) > 1:
            # Normalize scores to probabilities
            scores_norm = np.array(scores)
            scores_norm = scores_norm / np.sum(scores_norm) if np.sum(scores_norm) > 0 else scores_norm
            uncertainty['entropy'] = -np.sum(scores_norm * np.log(scores_norm + 1e-10))
        else:
            uncertainty['entropy'] = 0.0
        
        # Confidence interval width
        if len(confidences) > 1:
            uncertainty['confidence_interval_width'] = np.max(confidences) - np.min(confidences)
        else:
            uncertainty['confidence_interval_width'] = 0.0
        
        # Overall uncertainty
        uncertainty['overall'] = (
            uncertainty['aleatoric'] * 0.3 +
            uncertainty['epistemic'] * 0.4 +
            uncertainty['entropy'] * 0.2 +
            uncertainty['confidence_interval_width'] * 0.1
        )
        
        return uncertainty
    
    def _calculate_overall_reliability(self, reliability_factors: Dict[str, float]) -> float:
        """Calculate overall reliability score from individual factors"""
        
        # Weighted combination of reliability factors
        weights = {
            'confidence_consistency': 0.2,
            'score_confidence_alignment': 0.2,
            'model_diversity': 0.15,
            'processing_stability': 0.1,
            'source_reliability': 0.25,
            'historical_performance': 0.1
        }
        
        overall = 0.0
        total_weight = 0.0
        
        for factor, weight in weights.items():
            if factor in reliability_factors:
                overall += reliability_factors[factor] * weight
                total_weight += weight
        
        return overall / total_weight if total_weight > 0 else 0.5
    
    def _determine_confidence_level(self, overall_reliability: float) -> str:
        """Determine confidence level based on overall reliability"""
        
        if overall_reliability >= self.reliability_thresholds['high']:
            return 'high'
        elif overall_reliability >= self.reliability_thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    def _generate_reliability_recommendations(self, reliability_factors: Dict[str, float],
                                           uncertainty_metrics: Dict[str, float],
                                           overall_reliability: float) -> List[str]:
        """Generate recommendations for improving reliability"""
        
        recommendations = []
        
        # Confidence-based recommendations
        if reliability_factors.get('confidence_consistency', 0) < 0.6:
            recommendations.append("Consider ensemble methods to improve confidence consistency")
        
        if reliability_factors.get('score_confidence_alignment', 0) < 0.7:
            recommendations.append("Confidence calibration may be needed")
        
        # Uncertainty-based recommendations
        if uncertainty_metrics.get('overall', 0) > 0.7:
            recommendations.append("High uncertainty detected - consider additional validation")
        
        if uncertainty_metrics.get('epistemic', 0) > 0.5:
            recommendations.append("Model uncertainty is high - consider model improvement")
        
        # Processing-based recommendations
        if reliability_factors.get('processing_stability', 0) < 0.5:
            recommendations.append("Processing time varies significantly - check system stability")
        
        # Overall recommendations
        if overall_reliability < 0.6:
            recommendations.append("Overall reliability is low - consider multiple validation approaches")
        elif overall_reliability > 0.8:
            recommendations.append("High reliability achieved - predictions can be trusted")
        
        return recommendations
    
    def _get_historical_performance(self) -> float:
        """Get historical performance metric"""
        
        if not self.performance_history:
            return 0.5  # Default neutral performance
        
        # Calculate recent accuracy
        recent_performance = list(self.performance_history)[-100:]  # Last 100 predictions
        accuracies = [entry.get('accuracy', 0.5) for entry in recent_performance]
        
        return np.mean(accuracies)
    
    def calculate_calibration_metrics(self, predictions: List[Dict]) -> CalibrationMetrics:
        """Calculate comprehensive calibration metrics"""
        
        if not predictions:
            return CalibrationMetrics(
                reliability_score=0.0,
                calibration_error=1.0,
                sharpness=0.0,
                brier_score=1.0,
                expected_calibration_error=1.0,
                maximum_calibration_error=1.0,
                confidence_histogram={},
                accuracy_by_confidence={}
            )
        
        # Extract data
        confidences = [p['confidence'] for p in predictions]
        accuracies = [float(p['is_correct']) for p in predictions]
        
        # Calculate reliability (calibration curve)
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                accuracies, confidences, n_bins=10
            )
            
            # Reliability score (how close predicted probabilities are to actual frequencies)
            reliability_score = 1.0 - np.mean(np.abs(fraction_of_positives - mean_predicted_value))
            
        except Exception:
            reliability_score = 0.5
        
        # Calculate Expected Calibration Error (ECE)
        ece = self._calculate_ece(confidences, accuracies)
        
        # Calculate Maximum Calibration Error (MCE)
        mce = self._calculate_mce(confidences, accuracies)
        
        # Calculate Brier Score
        brier_score = np.mean([(c - a)**2 for c, a in zip(confidences, accuracies)])
        
        # Calculate sharpness (how confident the model is)
        sharpness = np.mean([abs(c - 0.5) for c in confidences])
        
        # Create confidence histogram
        confidence_histogram = self._create_confidence_histogram(confidences)
        
        # Calculate accuracy by confidence bins
        accuracy_by_confidence = self._calculate_accuracy_by_confidence(confidences, accuracies)
        
        return CalibrationMetrics(
            reliability_score=reliability_score,
            calibration_error=1.0 - reliability_score,
            sharpness=sharpness,
            brier_score=brier_score,
            expected_calibration_error=ece,
            maximum_calibration_error=mce,
            confidence_histogram=confidence_histogram,
            accuracy_by_confidence=accuracy_by_confidence
        )
    
    def _calculate_ece(self, confidences: List[float], accuracies: List[float], 
                      n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error"""
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find predictions in this bin
            in_bin = [(c >= bin_lower) and (c < bin_upper) for c in confidences]
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                # Calculate accuracy and confidence for this bin
                bin_accuracies = [a for a, in_b in zip(accuracies, in_bin) if in_b]
                bin_confidences = [c for c, in_b in zip(confidences, in_bin) if in_b]
                
                accuracy_in_bin = np.mean(bin_accuracies)
                avg_confidence_in_bin = np.mean(bin_confidences)
                
                # Add to ECE
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _calculate_mce(self, confidences: List[float], accuracies: List[float], 
                      n_bins: int = 10) -> float:
        """Calculate Maximum Calibration Error"""
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        max_error = 0.0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find predictions in this bin
            in_bin = [(c >= bin_lower) and (c < bin_upper) for c in confidences]
            
            if any(in_bin):
                # Calculate accuracy and confidence for this bin
                bin_accuracies = [a for a, in_b in zip(accuracies, in_bin) if in_b]
                bin_confidences = [c for c, in_b in zip(confidences, in_bin) if in_b]
                
                accuracy_in_bin = np.mean(bin_accuracies)
                avg_confidence_in_bin = np.mean(bin_confidences)
                
                # Update maximum error
                error = abs(avg_confidence_in_bin - accuracy_in_bin)
                max_error = max(max_error, error)
        
        return max_error
    
    def _create_confidence_histogram(self, confidences: List[float]) -> Dict[str, int]:
        """Create histogram of confidence values"""
        
        bins = ['0.0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5',
                '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
        
        histogram = {bin_name: 0 for bin_name in bins}
        
        for confidence in confidences:
            bin_idx = min(int(confidence * 10), 9)
            histogram[bins[bin_idx]] += 1
        
        return histogram
    
    def _calculate_accuracy_by_confidence(self, confidences: List[float], 
                                        accuracies: List[float]) -> Dict[str, float]:
        """Calculate accuracy for each confidence bin"""
        
        bins = ['0.0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5',
                '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
        
        accuracy_by_bin = {}
        
        for i, bin_name in enumerate(bins):
            bin_lower = i * 0.1
            bin_upper = (i + 1) * 0.1
            
            # Find predictions in this bin
            in_bin_indices = [j for j, c in enumerate(confidences) 
                             if bin_lower <= c < bin_upper]
            
            if in_bin_indices:
                bin_accuracies = [accuracies[j] for j in in_bin_indices]
                accuracy_by_bin[bin_name] = np.mean(bin_accuracies)
            else:
                accuracy_by_bin[bin_name] = 0.0
        
        return accuracy_by_bin
    
    def update_performance_history(self, predictions: List[Dict]):
        """Update performance history with new predictions"""
        
        for prediction in predictions:
            self.performance_history.append({
                'timestamp': datetime.now(),
                'accuracy': float(prediction.get('is_correct', False)),
                'confidence': prediction.get('confidence', 0.5),
                'model': prediction.get('model_name', 'unknown')
            })
    
    def get_calibration_summary(self) -> Dict[str, Any]:
        """Get summary of calibration system status"""
        
        return {
            'calibration_models': {
                model: fitted for model, fitted in self.models_fitted.items()
            },
            'history_size': len(self.calibration_history),
            'performance_history_size': len(self.performance_history),
            'recent_performance': self._get_historical_performance(),
            'calibration_method': self.calibration_method,
            'reliability_thresholds': self.reliability_thresholds
        }