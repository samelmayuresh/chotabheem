# tests/test_enhanced_preprocessing.py - Unit tests for enhanced preprocessing
import unittest
import numpy as np
from utils.enhanced_preprocessing import (
    EnhancedTextProcessor, EnhancedAudioProcessor, ContextAnalyzer,
    ProcessedText, ProcessedAudio
)

class TestEnhancedTextProcessor(unittest.TestCase):
    """Test cases for EnhancedTextProcessor"""
    
    def setUp(self):
        self.processor = EnhancedTextProcessor()
    
    def test_basic_text_processing(self):
        """Test basic text processing functionality"""
        text = "I'm feeling really happy today! This is AMAZING!!!"
        result = self.processor.process(text)
        
        self.assertIsInstance(result, ProcessedText)
        self.assertEqual(result.original, text)
        self.assertGreater(len(result.cleaned), 0)
        self.assertGreater(len(result.tokens), 0)
        self.assertGreater(result.quality_score, 0)
    
    def test_empty_text_handling(self):
        """Test handling of empty or invalid text"""
        result = self.processor.process("")
        self.assertEqual(result.quality_score, 0.0)
        self.assertIn("error", result.metadata)
        
        result = self.processor.process("   ")
        self.assertEqual(result.quality_score, 0.0)
    
    def test_contraction_expansion(self):
        """Test contraction expansion"""
        text = "I can't believe it won't work"
        result = self.processor.process(text)
        
        # Should expand contractions
        self.assertIn("cannot", result.cleaned.lower())
        self.assertIn("will not", result.cleaned.lower())
    
    def test_emphasis_detection(self):
        """Test detection of emphasized text"""
        text = "This is REALLY IMPORTANT stuff"
        result = self.processor.process(text)
        
        # Should detect emphasis
        self.assertTrue(result.metadata['has_emphasis'])
        self.assertGreater(result.metadata['intensifier_count'], 0)
    
    def test_quality_assessment(self):
        """Test text quality assessment"""
        good_text = "I am feeling quite happy today because of the wonderful weather."
        poor_text = "ok"
        
        good_result = self.processor.process(good_text)
        poor_result = self.processor.process(poor_text)
        
        self.assertGreater(good_result.quality_score, poor_result.quality_score)

class TestEnhancedAudioProcessor(unittest.TestCase):
    """Test cases for EnhancedAudioProcessor"""
    
    def setUp(self):
        self.processor = EnhancedAudioProcessor()
    
    def test_basic_audio_processing(self):
        """Test basic audio processing functionality"""
        # Create dummy audio data
        sample_rate = 16000
        duration = 2.0
        audio_data = np.random.randn(int(sample_rate * duration)).astype(np.float32)
        
        result = self.processor.process(audio_data, sample_rate)
        
        self.assertIsInstance(result, ProcessedAudio)
        self.assertEqual(result.sample_rate, sample_rate)
        self.assertAlmostEqual(result.duration, duration, places=1)
        self.assertGreater(result.quality_score, 0)
        self.assertIn('mfcc', result.features)
    
    def test_resampling(self):
        """Test audio resampling functionality"""
        original_sr = 44100
        target_sr = 16000
        duration = 1.0
        audio_data = np.random.randn(int(original_sr * duration)).astype(np.float32)
        
        result = self.processor.process(audio_data, original_sr)
        
        self.assertEqual(result.sample_rate, target_sr)
        expected_samples = int(target_sr * duration)
        self.assertAlmostEqual(len(result.audio_data), expected_samples, delta=100)
    
    def test_stereo_to_mono_conversion(self):
        """Test stereo to mono conversion"""
        sample_rate = 16000
        duration = 1.0
        # Create stereo audio (2 channels)
        stereo_audio = np.random.randn(int(sample_rate * duration), 2).astype(np.float32)
        
        result = self.processor.process(stereo_audio, sample_rate)
        
        # Should be converted to mono
        self.assertEqual(len(result.audio_data.shape), 1)
    
    def test_audio_enhancement(self):
        """Test audio enhancement functionality"""
        sample_rate = 16000
        duration = 1.0
        # Create noisy audio
        clean_signal = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration)))
        noise = np.random.randn(int(sample_rate * duration)) * 0.1
        noisy_audio = (clean_signal + noise).astype(np.float32)
        
        result = self.processor.process(noisy_audio, sample_rate)
        
        # Enhancement should improve quality score
        self.assertGreater(result.quality_score, 0.3)
    
    def test_feature_extraction(self):
        """Test audio feature extraction"""
        sample_rate = 16000
        duration = 2.0
        audio_data = np.random.randn(int(sample_rate * duration)).astype(np.float32)
        
        result = self.processor.process(audio_data, sample_rate)
        
        # Check that key features are extracted
        expected_features = ['mfcc', 'spectral_centroid', 'zero_crossing_rate', 'rms_energy']
        for feature in expected_features:
            self.assertIn(feature, result.features)
            self.assertIsInstance(result.features[feature], np.ndarray)
    
    def test_quality_assessment(self):
        """Test audio quality assessment"""
        sample_rate = 16000
        
        # Good quality audio (appropriate duration, clear signal)
        good_duration = 3.0
        good_audio = np.sin(2 * np.pi * 440 * np.linspace(0, good_duration, int(sample_rate * good_duration)))
        good_audio = good_audio.astype(np.float32)
        
        # Poor quality audio (too short, noisy)
        poor_duration = 0.1
        poor_audio = np.random.randn(int(sample_rate * poor_duration)).astype(np.float32) * 0.01
        
        good_result = self.processor.process(good_audio, sample_rate)
        poor_result = self.processor.process(poor_audio, sample_rate)
        
        self.assertGreater(good_result.quality_score, poor_result.quality_score)

class TestContextAnalyzer(unittest.TestCase):
    """Test cases for ContextAnalyzer"""
    
    def setUp(self):
        self.analyzer = ContextAnalyzer()
    
    def test_situational_context_detection(self):
        """Test situational context detection from text"""
        work_text = "I had a difficult meeting with my boss today at the office"
        health_text = "I went to the doctor and got some medicine for my pain"
        
        work_context = self.analyzer.analyze_context(text=work_text)
        health_context = self.analyzer.analyze_context(text=health_text)
        
        self.assertIn('work', work_context['situational']['detected_contexts'])
        self.assertIn('health', health_context['situational']['detected_contexts'])
    
    def test_linguistic_context_analysis(self):
        """Test linguistic context analysis"""
        excited_text = "This is AMAZING!!! Are you serious?!"
        calm_text = "I think this is quite nice."
        
        excited_context = self.analyzer.analyze_context(text=excited_text)
        calm_context = self.analyzer.analyze_context(text=calm_text)
        
        # Excited text should have higher linguistic intensity
        excited_intensity = excited_context['linguistic']['linguistic_intensity']
        calm_intensity = calm_context['linguistic']['linguistic_intensity']
        
        self.assertGreater(excited_intensity, calm_intensity)
    
    def test_acoustic_context_analysis(self):
        """Test acoustic context analysis"""
        # Mock audio features
        audio_features = {
            'zero_crossing_rate': np.array([[0.1, 0.2, 0.15]]),
            'f0': np.array([220.0, 230.0, 225.0]),
            'rms_energy': np.array([[0.5, 0.6, 0.55]])
        }
        
        context = self.analyzer.analyze_context(audio_features=audio_features)
        
        self.assertIn('speaking_rate', context['acoustic'])
        self.assertIn('pitch_mean', context['acoustic'])
        self.assertIn('energy_mean', context['acoustic'])
    
    def test_empty_input_handling(self):
        """Test handling of empty inputs"""
        context = self.analyzer.analyze_context()
        
        # Should return empty contexts without errors
        self.assertIn('temporal', context)
        self.assertIn('situational', context)
        self.assertIn('linguistic', context)
        self.assertIn('acoustic', context)

if __name__ == '__main__':
    # Create tests directory if it doesn't exist
    import os
    os.makedirs('tests', exist_ok=True)
    
    unittest.main()