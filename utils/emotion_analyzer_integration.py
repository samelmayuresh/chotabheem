# utils/emotion_analyzer_integration.py - Integration with existing emotion analyzer
import numpy as np
import logging
from typing import List, Dict, Optional, Any, Tuple
from utils.enhanced_emotion_analyzer import EnhancedEmotionAnalyzer, EmotionConfig, EmotionResult
from utils.text_emotion_ensemble import EmotionScore

class IntegratedEmotionAnalyzer:
    """Integrated emotion analyzer that maintains compatibility with existing interface"""
    
    def __init__(self, model_config: dict = None):
        """Initialize with backward compatibility"""
        self.model_config = model_config or {}
        
        # Create enhanced analyzer configuration
        enhanced_config = EmotionConfig(
            precision_level=self.model_config.get('precision_level', 'balanced'),
            enable_ensemble=self.model_config.get('enable_ensemble', True),
            enable_multimodal=self.model_config.get('enable_multimodal', True),
            fusion_strategy=self.model_config.get('fusion_strategy', 'adaptive'),
            confidence_threshold=self.model_config.get('confidence_threshold', 0.5)
        )
        
        # Initialize enhanced analyzer
        self.enhanced_analyzer = EnhancedEmotionAnalyzer(enhanced_config)
        
        # Maintain compatibility flags
        self.legacy_mode = self.model_config.get('legacy_mode', False)
        
        logging.info("Integrated emotion analyzer initialized with enhanced capabilities")
    
    def analyze_text(self, text: str) -> List[Dict]:
        """Analyze text emotions with backward compatibility"""
        try:
            # Use enhanced analyzer
            result = self.enhanced_analyzer.analyze_text(text)
            
            # Convert to legacy format if needed
            if self.legacy_mode:
                return self._convert_to_legacy_format(result.all_emotions)
            else:
                return self._convert_to_enhanced_format(result)
                
        except Exception as e:
            logging.error(f"Enhanced text analysis failed, using fallback: {e}")
            return self._fallback_text_analysis(text)
    
    def analyze_audio(self, audio_data, sample_rate: int = 16000) -> Tuple[str, List[Dict]]:
        """Analyze audio emotions with backward compatibility"""
        try:
            # Use enhanced analyzer
            result = self.enhanced_analyzer.analyze_audio(audio_data, sample_rate)
            
            # Extract transcript (if available from metadata)
            transcript = result.processing_metadata.get('transcript', '')
            
            # Convert emotions to expected format
            if self.legacy_mode:
                emotions = self._convert_to_legacy_format(result.all_emotions)
            else:
                emotions = self._convert_to_enhanced_format(result)
            
            return transcript, emotions
            
        except Exception as e:
            logging.error(f"Enhanced audio analysis failed, using fallback: {e}")
            return self._fallback_audio_analysis(audio_data, sample_rate)
    
    def analyze_multimodal(self, text: str, audio_data, sample_rate: int = 16000) -> Dict[str, Any]:
        """Analyze multimodal emotions (new enhanced feature)"""
        try:
            result = self.enhanced_analyzer.analyze_multimodal(text, audio_data, sample_rate)
            
            return {
                'primary_emotion': result.primary_emotion.label,
                'confidence': result.primary_emotion.confidence,
                'all_emotions': self._convert_to_legacy_format(result.all_emotions),
                'confidence_level': result.confidence_level,
                'insights': result.insights,
                'recommendations': result.recommendations,
                'fusion_method': result.fusion_result.fusion_method if result.fusion_result else 'simple',
                'processing_time': result.processing_metadata['processing_time']
            }
            
        except Exception as e:
            logging.error(f"Multimodal analysis failed: {e}")
            # Fallback to separate analysis
            text_result = self.analyze_text(text)
            audio_transcript, audio_result = self.analyze_audio(audio_data, sample_rate)
            
            return {
                'primary_emotion': text_result[0]['label'] if text_result else 'neutral',
                'confidence': text_result[0]['score'] if text_result else 0.5,
                'all_emotions': text_result + audio_result,
                'confidence_level': 'medium',
                'insights': ['Fallback analysis used'],
                'recommendations': ['Consider checking system configuration'],
                'fusion_method': 'fallback',
                'processing_time': 0.1
            }
    
    def get_emotion_insights(self, emotions: List[Dict]) -> Dict:
        """Get emotion insights with enhanced capabilities"""
        if not emotions:
            return {"primary": "neutral", "confidence": 0.0, "insights": []}
        
        # Convert to EmotionScore format for enhanced processing
        emotion_scores = []
        for emotion in emotions:
            emotion_scores.append(EmotionScore(
                label=emotion.get('label', 'neutral'),
                score=emotion.get('score', 0.5),
                confidence=emotion.get('confidence', emotion.get('score', 0.5)),
                source='legacy',
                model_name='integrated',
                processing_time=0.0,
                metadata=emotion
            ))
        
        # Use enhanced analyzer's insight generation
        try:
            # Create a mock result for insight generation
            mock_result = self.enhanced_analyzer._create_emotion_result(
                emotion_scores, 'integrated', None, None, None
            )
            
            return {
                "primary": mock_result.primary_emotion.label,
                "confidence": mock_result.primary_emotion.confidence,
                "insights": mock_result.insights,
                "recommendations": mock_result.recommendations,
                "complexity": mock_result.uncertainty_score
            }
            
        except Exception as e:
            logging.error(f"Enhanced insights failed: {e}")
            return self._fallback_insights(emotions)
    
    def _convert_to_legacy_format(self, emotion_scores: List[EmotionScore]) -> List[Dict]:
        """Convert enhanced format to legacy format"""
        return [
            {
                "label": score.label,
                "score": score.score,
                "confidence": score.confidence
            }
            for score in emotion_scores
        ]
    
    def _convert_to_enhanced_format(self, result: EmotionResult) -> List[Dict]:
        """Convert to enhanced format with additional metadata"""
        enhanced_emotions = []
        
        for emotion in result.all_emotions:
            enhanced_emotions.append({
                "label": emotion.label,
                "score": emotion.score,
                "confidence": emotion.confidence,
                "source": emotion.source,
                "model_name": emotion.model_name,
                "processing_time": emotion.processing_time,
                "metadata": emotion.metadata
            })
        
        return enhanced_emotions
    
    def _fallback_text_analysis(self, text: str) -> List[Dict]:
        """Fallback text analysis using simple methods"""
        # Simple keyword-based analysis
        emotion_keywords = {
            'joy': ['happy', 'joy', 'excited', 'great', 'wonderful', 'amazing', 'fantastic'],
            'sadness': ['sad', 'depressed', 'unhappy', 'terrible', 'awful', 'disappointed'],
            'anger': ['angry', 'mad', 'furious', 'annoyed', 'frustrated', 'irritated'],
            'fear': ['scared', 'afraid', 'worried', 'anxious', 'nervous', 'frightened'],
            'surprise': ['surprised', 'shocked', 'amazed', 'unexpected', 'astonished'],
            'neutral': ['okay', 'fine', 'normal', 'regular', 'alright']
        }
        
        text_lower = text.lower()
        emotion_scores = []
        
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                normalized_score = min(score / 3.0, 1.0)
                emotion_scores.append({
                    "label": emotion,
                    "score": normalized_score,
                    "confidence": normalized_score * 0.7
                })
        
        if not emotion_scores:
            emotion_scores.append({
                "label": "neutral",
                "score": 0.5,
                "confidence": 0.5
            })
        
        emotion_scores.sort(key=lambda x: x["score"], reverse=True)
        return emotion_scores
    
    def _fallback_audio_analysis(self, audio_data, sample_rate: int) -> Tuple[str, List[Dict]]:
        """Fallback audio analysis using simple methods"""
        try:
            import librosa
            
            # Simple energy-based analysis
            rms_energy = np.mean(librosa.feature.rms(y=audio_data)[0])
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio_data)[0])
            
            # Simple classification
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
            
            emotions = [{
                "label": primary_emotion,
                "score": score,
                "confidence": score * 0.6
            }]
            
            return "", emotions
            
        except Exception:
            return "", [{"label": "neutral", "score": 0.5, "confidence": 0.3}]
    
    def _fallback_insights(self, emotions: List[Dict]) -> Dict:
        """Fallback insight generation"""
        if not emotions:
            return {"primary": "neutral", "confidence": 0.0, "insights": []}
        
        primary = emotions[0]
        insights = []
        
        if primary["confidence"] > 0.8:
            insights.append("High confidence in emotion detection")
        elif primary["confidence"] < 0.5:
            insights.append("Low confidence - results may be uncertain")
        
        if len(emotions) > 1:
            insights.append(f"Multiple emotions detected: {', '.join([e['label'] for e in emotions[:3]])}")
        
        return {
            "primary": primary["label"],
            "confidence": primary["confidence"],
            "insights": insights,
            "complexity": np.var([e["score"] for e in emotions]) if len(emotions) > 1 else 0.1
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from enhanced analyzer"""
        try:
            return self.enhanced_analyzer.get_performance_summary()
        except Exception as e:
            logging.error(f"Failed to get performance summary: {e}")
            return {
                'error': str(e),
                'fallback_mode': True,
                'enhanced_features_available': False
            }
    
    def save_configuration(self, filepath: str):
        """Save configuration"""
        try:
            self.enhanced_analyzer.save_configuration(filepath)
        except Exception as e:
            logging.error(f"Failed to save configuration: {e}")
    
    @classmethod
    def load_configuration(cls, filepath: str) -> 'IntegratedEmotionAnalyzer':
        """Load configuration"""
        try:
            enhanced_analyzer = EnhancedEmotionAnalyzer.load_configuration(filepath)
            
            # Create integrated analyzer with loaded enhanced analyzer
            integrated = cls()
            integrated.enhanced_analyzer = enhanced_analyzer
            
            return integrated
            
        except Exception as e:
            logging.error(f"Failed to load configuration: {e}")
            return cls()  # Return default

# Backward compatibility: replace the original EmotionAnalyzer
class EmotionAnalyzer(IntegratedEmotionAnalyzer):
    """Drop-in replacement for the original EmotionAnalyzer with enhanced capabilities"""
    
    def __init__(self, model_config: dict = None):
        # Enable legacy mode by default for backward compatibility
        if model_config is None:
            model_config = {}
        model_config['legacy_mode'] = model_config.get('legacy_mode', True)
        
        super().__init__(model_config)
        
        logging.info("EmotionAnalyzer initialized with enhanced backend (legacy compatibility mode)")

def create_enhanced_analyzer(precision_level: str = 'balanced') -> IntegratedEmotionAnalyzer:
    """Factory function to create enhanced analyzer with specific precision level"""
    
    config = {
        'precision_level': precision_level,
        'enable_ensemble': True,
        'enable_multimodal': True,
        'legacy_mode': False  # Use enhanced features
    }
    
    return IntegratedEmotionAnalyzer(config)