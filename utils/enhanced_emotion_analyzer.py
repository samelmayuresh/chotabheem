# utils/enhanced_emotion_analyzer.py - Main enhanced emotion analyzer
import numpy as np
import logging
from typing import List, Dict, Optional, Any, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import json
import os

# Import all the components we've built
from utils.enhanced_preprocessing import EnhancedTextProcessor, EnhancedAudioProcessor, ContextAnalyzer
from utils.text_emotion_ensemble import TextEmotionEnsemble, create_default_text_ensemble, ModelConfig
from utils.voice_emotion_ensemble import VoiceEmotionEnsemble, create_default_voice_ensemble
from utils.multimodal_fusion import MultimodalFusionEngine, FusionResult
from utils.confidence_calibration import ConfidenceCalibrator, ReliabilityAssessment
from utils.quality_assurance import QualityAssurance, ValidationResult
from utils.text_emotion_ensemble import EmotionScore

@dataclass
class EmotionConfig:
    """Configuration for enhanced emotion analysis"""
    precision_level: str = 'balanced'  # 'fast', 'balanced', 'high_precision'
    enable_ensemble: bool = True
    enable_multimodal: bool = True
    text_models: List[str] = None
    voice_models: List[str] = None
    fusion_strategy: str = 'adaptive'
    confidence_threshold: float = 0.5
    enable_quality_assurance: bool = True
    enable_confidence_calibration: bool = True
    custom_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.text_models is None:
            self.text_models = ['bert', 'roberta', 'context_aware']
        if self.voice_models is None:
            self.voice_models = ['wav2vec2', 'spectral', 'prosodic']
        if self.custom_config is None:
            self.custom_config = {}

@dataclass
class EmotionResult:
    """Comprehensive emotion analysis result"""
    primary_emotion: EmotionScore
    all_emotions: List[EmotionScore]
    confidence_level: str  # 'high', 'medium', 'low'
    uncertainty_score: float
    processing_metadata: Dict[str, Any]
    insights: List[str]
    recommendations: List[str]
    validation_result: Optional[ValidationResult] = None
    reliability_assessment: Optional[ReliabilityAssessment] = None
    fusion_result: Optional[FusionResult] = None

class EnhancedEmotionAnalyzer:
    """Advanced emotion analyzer with ensemble methods and multimodal fusion"""
    
    def __init__(self, config: EmotionConfig = None):
        self.config = config or EmotionConfig()
        
        # Initialize components
        self.text_processor = EnhancedTextProcessor()
        self.audio_processor = EnhancedAudioProcessor()
        self.context_analyzer = ContextAnalyzer()
        
        # Initialize ensembles
        self.text_ensemble = None
        self.voice_ensemble = None
        
        # Initialize fusion and quality systems
        self.fusion_engine = None
        self.confidence_calibrator = None
        self.quality_assurance = None
        
        # Performance tracking
        self.analysis_history = []
        self.performance_metrics = {
            'total_analyses': 0,
            'text_analyses': 0,
            'voice_analyses': 0,
            'multimodal_analyses': 0,
            'average_processing_time': 0.0,
            'average_confidence': 0.0
        }
        
        # Initialize all components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all analysis components based on configuration"""
        
        try:
            # Initialize text ensemble if enabled
            if self.config.enable_ensemble:
                self.text_ensemble = self._create_text_ensemble()
                self.voice_ensemble = self._create_voice_ensemble()
                logging.info("Initialized emotion ensembles")
            
            # Initialize multimodal fusion if enabled
            if self.config.enable_multimodal:
                fusion_config = {
                    'default_strategy': self.config.fusion_strategy,
                    'confidence_threshold': self.config.confidence_threshold
                }
                self.fusion_engine = MultimodalFusionEngine(fusion_config)
                logging.info("Initialized multimodal fusion engine")
            
            # Initialize confidence calibration if enabled
            if self.config.enable_confidence_calibration:
                self.confidence_calibrator = ConfidenceCalibrator()
                logging.info("Initialized confidence calibrator")
            
            # Initialize quality assurance if enabled
            if self.config.enable_quality_assurance:
                self.quality_assurance = QualityAssurance()
                logging.info("Initialized quality assurance system")
            
        except Exception as e:
            logging.error(f"Failed to initialize components: {e}")
            # Continue with basic functionality
    
    def _create_text_ensemble(self) -> TextEmotionEnsemble:
        """Create text emotion ensemble based on configuration"""
        
        if self.config.precision_level == 'fast':
            # Fast configuration - fewer models
            configs = [
                ModelConfig(
                    name="fast_bert",
                    model_path="j-hartmann/emotion-english-distilroberta-base",
                    model_type="bert",
                    weight=1.0,
                    enabled=True
                )
            ]
        elif self.config.precision_level == 'high_precision':
            # High precision - more models
            configs = [
                ModelConfig(
                    name="primary_bert",
                    model_path="j-hartmann/emotion-english-distilroberta-base",
                    model_type="bert",
                    weight=1.0,
                    enabled=True
                ),
                ModelConfig(
                    name="roberta_social",
                    model_path="cardiffnlp/twitter-roberta-base-emotion",
                    model_type="roberta",
                    weight=0.9,
                    enabled=True
                ),
                ModelConfig(
                    name="context_aware",
                    model_path="j-hartmann/emotion-english-distilroberta-base",
                    model_type="context_aware",
                    weight=0.8,
                    enabled=True
                ),
                ModelConfig(
                    name="multilingual",
                    model_path="cardiffnlp/twitter-xlm-roberta-base-sentiment",
                    model_type="multilingual",
                    weight=0.7,
                    enabled=True
                )
            ]
        else:
            # Balanced configuration
            configs = [
                ModelConfig(
                    name="primary_bert",
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
        
        return TextEmotionEnsemble(configs, voting_strategy=self.config.fusion_strategy)
    
    def _create_voice_ensemble(self) -> VoiceEmotionEnsemble:
        """Create voice emotion ensemble based on configuration"""
        
        if self.config.precision_level == 'fast':
            # Fast configuration - rule-based models only
            configs = [
                ModelConfig(
                    name="spectral_fast",
                    model_path="",
                    model_type="spectral",
                    weight=1.0,
                    enabled=True
                )
            ]
        elif self.config.precision_level == 'high_precision':
            # High precision - all models
            configs = [
                ModelConfig(
                    name="wav2vec2_primary",
                    model_path="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                    model_type="wav2vec2",
                    weight=1.0,
                    enabled=True
                ),
                ModelConfig(
                    name="spectral_analyzer",
                    model_path="",
                    model_type="spectral",
                    weight=0.7,
                    enabled=True
                ),
                ModelConfig(
                    name="prosodic_analyzer",
                    model_path="",
                    model_type="prosodic",
                    weight=0.8,
                    enabled=True
                ),
                ModelConfig(
                    name="dimensional_analyzer",
                    model_path="",
                    model_type="dimensional",
                    weight=0.6,
                    enabled=True
                )
            ]
        else:
            # Balanced configuration
            configs = [
                ModelConfig(
                    name="spectral_primary",
                    model_path="",
                    model_type="spectral",
                    weight=1.0,
                    enabled=True
                ),
                ModelConfig(
                    name="prosodic_secondary",
                    model_path="",
                    model_type="prosodic",
                    weight=0.8,
                    enabled=True
                )
            ]
        
        return VoiceEmotionEnsemble(configs, voting_strategy=self.config.fusion_strategy)
    
    def analyze_text(self, text: str, context: Dict = None) -> EmotionResult:
        """Enhanced text emotion analysis with ensemble methods"""
        
        start_time = datetime.now()
        
        try:
            # Preprocess text
            processed_text = self.text_processor.process(text, context)
            
            # Analyze with ensemble if available
            if self.text_ensemble:
                emotion_scores = self.text_ensemble.predict(processed_text.normalized, context)
            else:
                # Fallback to basic analysis
                emotion_scores = self._basic_text_analysis(processed_text.normalized)
            
            # Apply confidence calibration if enabled
            if self.confidence_calibrator:
                emotion_scores = self.confidence_calibrator.calibrate_confidence(emotion_scores, context=context)
            
            # Create analysis result
            result = self._create_emotion_result(
                emotion_scores, 'text', start_time, processed_text, context
            )
            
            # Apply quality assurance if enabled
            if self.quality_assurance:
                input_data = {
                    'type': 'text',
                    'length': len(text),
                    'quality_score': processed_text.quality_score
                }
                result.validation_result = self.quality_assurance.validate_prediction(
                    emotion_scores, input_data, context
                )
            
            # Apply reliability assessment if enabled
            if self.confidence_calibrator:
                result.reliability_assessment = self.confidence_calibrator.assess_reliability(
                    emotion_scores, context
                )
            
            # Update performance metrics
            self._update_performance_metrics('text', result)
            
            return result
            
        except Exception as e:
            logging.error(f"Text analysis failed: {e}")
            return self._create_error_result('text', str(e))
    
    def analyze_audio(self, audio: np.ndarray, sample_rate: int, context: Dict = None) -> EmotionResult:
        """Enhanced voice emotion analysis with ensemble methods"""
        
        start_time = datetime.now()
        
        try:
            # Preprocess audio
            processed_audio = self.audio_processor.process(audio, sample_rate, context)
            
            # Analyze with ensemble if available
            if self.voice_ensemble:
                emotion_scores = self.voice_ensemble.predict(
                    processed_audio.audio_data, processed_audio.sample_rate, context
                )
            else:
                # Fallback to basic analysis
                emotion_scores = self._basic_audio_analysis(processed_audio.audio_data, processed_audio.sample_rate)
            
            # Apply confidence calibration if enabled
            if self.confidence_calibrator:
                emotion_scores = self.confidence_calibrator.calibrate_confidence(emotion_scores, context=context)
            
            # Create analysis result
            result = self._create_emotion_result(
                emotion_scores, 'voice', start_time, processed_audio, context
            )
            
            # Apply quality assurance if enabled
            if self.quality_assurance:
                input_data = {
                    'type': 'voice',
                    'duration': processed_audio.duration,
                    'quality_score': processed_audio.quality_score
                }
                result.validation_result = self.quality_assurance.validate_prediction(
                    emotion_scores, input_data, context
                )
            
            # Apply reliability assessment if enabled
            if self.confidence_calibrator:
                result.reliability_assessment = self.confidence_calibrator.assess_reliability(
                    emotion_scores, context
                )
            
            # Update performance metrics
            self._update_performance_metrics('voice', result)
            
            return result
            
        except Exception as e:
            logging.error(f"Audio analysis failed: {e}")
            return self._create_error_result('voice', str(e))
    
    def analyze_multimodal(self, text: str, audio: np.ndarray, sample_rate: int, 
                          context: Dict = None) -> EmotionResult:
        """Multimodal emotion analysis with fusion"""
        
        start_time = datetime.now()
        
        try:
            # Analyze text and audio separately
            text_result = self.analyze_text(text, context)
            audio_result = self.analyze_audio(audio, sample_rate, context)
            
            # Fuse results if fusion engine available
            if self.fusion_engine:
                fusion_result = self.fusion_engine.fuse_predictions(
                    text_result.all_emotions,
                    audio_result.all_emotions,
                    context
                )
                
                # Create multimodal result
                result = self._create_emotion_result(
                    fusion_result.emotion_scores, 'multimodal', start_time, None, context
                )
                result.fusion_result = fusion_result
                
                # Combine insights and recommendations
                result.insights.extend(text_result.insights)
                result.insights.extend(audio_result.insights)
                result.recommendations.extend(text_result.recommendations)
                result.recommendations.extend(audio_result.recommendations)
                
                # Add fusion-specific insights
                result.insights.append(f"Multimodal fusion using {fusion_result.fusion_method}")
                if fusion_result.conflict_resolution:
                    result.insights.append(f"Resolved {len(fusion_result.conflict_resolution)} conflicts")
                
            else:
                # Simple combination without fusion
                combined_emotions = text_result.all_emotions + audio_result.all_emotions
                combined_emotions.sort(key=lambda x: x.score, reverse=True)
                
                result = self._create_emotion_result(
                    combined_emotions, 'multimodal', start_time, None, context
                )
            
            # Apply quality assurance for multimodal result
            if self.quality_assurance:
                input_data = {
                    'type': 'multimodal',
                    'text_length': len(text),
                    'audio_duration': len(audio) / sample_rate
                }
                result.validation_result = self.quality_assurance.validate_prediction(
                    result.all_emotions, input_data, context
                )
            
            # Update performance metrics
            self._update_performance_metrics('multimodal', result)
            
            return result
            
        except Exception as e:
            logging.error(f"Multimodal analysis failed: {e}")
            return self._create_error_result('multimodal', str(e))
    
    def _basic_text_analysis(self, text: str) -> List[EmotionScore]:
        """Fallback basic text analysis"""
        # Simple rule-based analysis as fallback
        emotion_keywords = {
            'joy': ['happy', 'joy', 'excited', 'great', 'wonderful', 'amazing'],
            'sadness': ['sad', 'depressed', 'unhappy', 'terrible', 'awful'],
            'anger': ['angry', 'mad', 'furious', 'annoyed', 'frustrated'],
            'fear': ['scared', 'afraid', 'worried', 'anxious', 'nervous'],
            'surprise': ['surprised', 'shocked', 'amazed', 'unexpected'],
            'neutral': ['okay', 'fine', 'normal', 'regular']
        }
        
        text_lower = text.lower()
        emotion_scores = []
        
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                normalized_score = min(score / 3.0, 1.0)  # Normalize to [0,1]
                emotion_scores.append(EmotionScore(
                    label=emotion,
                    score=normalized_score,
                    confidence=normalized_score * 0.7,  # Lower confidence for basic analysis
                    source='text',
                    model_name='basic_text_analyzer',
                    processing_time=0.01,
                    metadata={'method': 'keyword_matching'}
                ))
        
        if not emotion_scores:
            emotion_scores.append(EmotionScore(
                label='neutral',
                score=0.5,
                confidence=0.5,
                source='text',
                model_name='basic_text_analyzer',
                processing_time=0.01,
                metadata={'method': 'default_neutral'}
            ))
        
        emotion_scores.sort(key=lambda x: x.score, reverse=True)
        return emotion_scores
    
    def _basic_audio_analysis(self, audio: np.ndarray, sample_rate: int) -> List[EmotionScore]:
        """Fallback basic audio analysis"""
        # Simple energy-based analysis as fallback
        try:
            import librosa
            
            # Calculate basic features
            rms_energy = np.mean(librosa.feature.rms(y=audio)[0])
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio)[0])
            
            # Simple rule-based classification
            if rms_energy > 0.1 and zcr > 0.1:
                primary_emotion = 'joy'
                score = min(rms_energy * 2, 1.0)
            elif rms_energy < 0.03:
                primary_emotion = 'sadness'
                score = max(0.3, 1.0 - rms_energy * 10)
            elif zcr > 0.15:
                primary_emotion = 'anger'
                score = min(zcr * 3, 1.0)
            else:
                primary_emotion = 'neutral'
                score = 0.5
            
            return [EmotionScore(
                label=primary_emotion,
                score=score,
                confidence=score * 0.6,  # Lower confidence for basic analysis
                source='voice',
                model_name='basic_audio_analyzer',
                processing_time=0.05,
                metadata={'method': 'energy_based', 'rms_energy': rms_energy, 'zcr': zcr}
            )]
            
        except Exception:
            # Ultimate fallback
            return [EmotionScore(
                label='neutral',
                score=0.5,
                confidence=0.3,
                source='voice',
                model_name='basic_audio_analyzer',
                processing_time=0.01,
                metadata={'method': 'fallback_neutral'}
            )]
    
    def _create_emotion_result(self, emotion_scores: List[EmotionScore], 
                             analysis_type: str, start_time: datetime,
                             processed_input: Any = None, context: Dict = None) -> EmotionResult:
        """Create comprehensive emotion result"""
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        if not emotion_scores:
            emotion_scores = [EmotionScore(
                label='neutral',
                score=0.5,
                confidence=0.3,
                source=analysis_type,
                model_name='fallback',
                processing_time=processing_time,
                metadata={'reason': 'no_emotions_detected'}
            )]
        
        primary_emotion = emotion_scores[0]
        
        # Determine confidence level
        avg_confidence = np.mean([e.confidence for e in emotion_scores[:3]])
        if avg_confidence >= 0.8:
            confidence_level = 'high'
        elif avg_confidence >= 0.6:
            confidence_level = 'medium'
        else:
            confidence_level = 'low'
        
        # Calculate uncertainty score
        if len(emotion_scores) > 1:
            scores = [e.score for e in emotion_scores]
            uncertainty_score = np.var(scores)
        else:
            uncertainty_score = 0.1
        
        # Generate insights
        insights = self._generate_insights(emotion_scores, analysis_type, processed_input)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(emotion_scores, confidence_level, uncertainty_score)
        
        # Create processing metadata
        processing_metadata = {
            'analysis_type': analysis_type,
            'processing_time': processing_time,
            'timestamp': datetime.now(),
            'config': {
                'precision_level': self.config.precision_level,
                'enable_ensemble': self.config.enable_ensemble,
                'enable_multimodal': self.config.enable_multimodal
            },
            'context': context or {}
        }
        
        if processed_input:
            if hasattr(processed_input, 'quality_score'):
                processing_metadata['input_quality'] = processed_input.quality_score
            if hasattr(processed_input, 'metadata'):
                processing_metadata['input_metadata'] = processed_input.metadata
        
        return EmotionResult(
            primary_emotion=primary_emotion,
            all_emotions=emotion_scores,
            confidence_level=confidence_level,
            uncertainty_score=uncertainty_score,
            processing_metadata=processing_metadata,
            insights=insights,
            recommendations=recommendations
        )
    
    def _generate_insights(self, emotion_scores: List[EmotionScore], 
                          analysis_type: str, processed_input: Any = None) -> List[str]:
        """Generate insights from emotion analysis"""
        
        insights = []
        
        if not emotion_scores:
            return ['No emotions detected in the input']
        
        primary = emotion_scores[0]
        
        # Primary emotion insight
        insights.append(f"Primary emotion detected: {primary.label} (confidence: {primary.confidence:.1%})")
        
        # Confidence insights
        if primary.confidence > 0.9:
            insights.append("Very high confidence in emotion detection")
        elif primary.confidence < 0.5:
            insights.append("Low confidence - consider additional validation")
        
        # Multiple emotions insight
        strong_emotions = [e for e in emotion_scores if e.score > 0.3]
        if len(strong_emotions) > 1:
            emotion_list = ', '.join([e.label for e in strong_emotions[:3]])
            insights.append(f"Multiple emotions detected: {emotion_list}")
        
        # Analysis type specific insights
        if analysis_type == 'multimodal':
            insights.append("Multimodal analysis provides enhanced accuracy")
        elif analysis_type == 'text':
            if processed_input and hasattr(processed_input, 'metadata'):
                if processed_input.metadata.get('has_emphasis', False):
                    insights.append("Text contains emotional emphasis")
        elif analysis_type == 'voice':
            if processed_input and hasattr(processed_input, 'duration'):
                if processed_input.duration < 1.0:
                    insights.append("Short audio clip - confidence may be limited")
                elif processed_input.duration > 10.0:
                    insights.append("Long audio provides rich emotional context")
        
        return insights
    
    def _generate_recommendations(self, emotion_scores: List[EmotionScore], 
                                confidence_level: str, uncertainty_score: float) -> List[str]:
        """Generate recommendations based on analysis results"""
        
        recommendations = []
        
        # Confidence-based recommendations
        if confidence_level == 'low':
            recommendations.append("Consider collecting additional input for better accuracy")
        elif confidence_level == 'high':
            recommendations.append("High confidence results - suitable for automated decisions")
        
        # Uncertainty-based recommendations
        if uncertainty_score > 0.5:
            recommendations.append("High uncertainty detected - manual review recommended")
        
        # Emotion-specific recommendations
        if emotion_scores:
            primary = emotion_scores[0]
            
            if primary.label in ['sadness', 'fear', 'anger']:
                recommendations.append("Negative emotion detected - consider supportive response")
            elif primary.label == 'joy':
                recommendations.append("Positive emotion detected - good opportunity for engagement")
        
        return recommendations
    
    def _create_error_result(self, analysis_type: str, error_message: str) -> EmotionResult:
        """Create error result when analysis fails"""
        
        error_emotion = EmotionScore(
            label='neutral',
            score=0.5,
            confidence=0.1,
            source=analysis_type,
            model_name='error_handler',
            processing_time=0.0,
            metadata={'error': error_message}
        )
        
        return EmotionResult(
            primary_emotion=error_emotion,
            all_emotions=[error_emotion],
            confidence_level='low',
            uncertainty_score=1.0,
            processing_metadata={
                'analysis_type': analysis_type,
                'error': error_message,
                'timestamp': datetime.now()
            },
            insights=[f"Analysis failed: {error_message}"],
            recommendations=['Check input data and try again', 'Contact support if error persists']
        )
    
    def _update_performance_metrics(self, analysis_type: str, result: EmotionResult):
        """Update performance tracking metrics"""
        
        self.performance_metrics['total_analyses'] += 1
        
        if analysis_type == 'text':
            self.performance_metrics['text_analyses'] += 1
        elif analysis_type == 'voice':
            self.performance_metrics['voice_analyses'] += 1
        elif analysis_type == 'multimodal':
            self.performance_metrics['multimodal_analyses'] += 1
        
        # Update averages
        current_time = result.processing_metadata['processing_time']
        current_conf = result.primary_emotion.confidence
        
        total = self.performance_metrics['total_analyses']
        
        self.performance_metrics['average_processing_time'] = (
            (self.performance_metrics['average_processing_time'] * (total - 1) + current_time) / total
        )
        
        self.performance_metrics['average_confidence'] = (
            (self.performance_metrics['average_confidence'] * (total - 1) + current_conf) / total
        )
        
        # Store in history (keep last 1000)
        self.analysis_history.append({
            'timestamp': datetime.now(),
            'type': analysis_type,
            'primary_emotion': result.primary_emotion.label,
            'confidence': result.primary_emotion.confidence,
            'processing_time': current_time
        })
        
        if len(self.analysis_history) > 1000:
            self.analysis_history = self.analysis_history[-1000:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        
        return {
            'metrics': self.performance_metrics.copy(),
            'recent_analyses': len(self.analysis_history),
            'component_status': {
                'text_ensemble': self.text_ensemble is not None,
                'voice_ensemble': self.voice_ensemble is not None,
                'fusion_engine': self.fusion_engine is not None,
                'confidence_calibrator': self.confidence_calibrator is not None,
                'quality_assurance': self.quality_assurance is not None
            },
            'configuration': {
                'precision_level': self.config.precision_level,
                'enable_ensemble': self.config.enable_ensemble,
                'enable_multimodal': self.config.enable_multimodal,
                'fusion_strategy': self.config.fusion_strategy
            }
        }
    
    def save_configuration(self, filepath: str):
        """Save current configuration to file"""
        
        config_data = {
            'precision_level': self.config.precision_level,
            'enable_ensemble': self.config.enable_ensemble,
            'enable_multimodal': self.config.enable_multimodal,
            'text_models': self.config.text_models,
            'voice_models': self.config.voice_models,
            'fusion_strategy': self.config.fusion_strategy,
            'confidence_threshold': self.config.confidence_threshold,
            'enable_quality_assurance': self.config.enable_quality_assurance,
            'enable_confidence_calibration': self.config.enable_confidence_calibration,
            'custom_config': self.config.custom_config
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(config_data, f, indent=2)
            logging.info(f"Configuration saved to {filepath}")
        except Exception as e:
            logging.error(f"Failed to save configuration: {e}")
    
    @classmethod
    def load_configuration(cls, filepath: str) -> 'EnhancedEmotionAnalyzer':
        """Load configuration from file and create analyzer"""
        
        try:
            with open(filepath, 'r') as f:
                config_data = json.load(f)
            
            config = EmotionConfig(**config_data)
            analyzer = cls(config)
            
            logging.info(f"Configuration loaded from {filepath}")
            return analyzer
            
        except Exception as e:
            logging.error(f"Failed to load configuration: {e}")
            return cls()  # Return default configuration