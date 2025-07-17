# utils/emotion_analyzer.py - Enhanced emotion analysis with caching and optimization
import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from transformers import pipeline
import torch
import librosa
from datetime import datetime, timedelta
import logging

class EmotionAnalyzer:
    """Enhanced emotion analyzer with caching and batch processing"""
    
    def __init__(self, model_config: dict):
        self.model_config = model_config
        self._asr_pipe = None
        self._emotion_pipe = None
        self._voice_emotion_pipe = None
        
    @st.cache_resource
    def _load_asr_model(_self):
        """Load ASR model with caching"""
        try:
            return pipeline(
                "automatic-speech-recognition",
                model=_self.model_config["asr_model"],
                device=_self.model_config["device"],
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
        except Exception as e:
            logging.error(f"Failed to load ASR model: {e}")
            return None
    
    @st.cache_resource
    def _load_emotion_model(_self):
        """Load emotion classification model with caching"""
        try:
            return pipeline(
                "text-classification",
                model=_self.model_config["emotion_model"],
                top_k=None,  # Updated from deprecated return_all_scores=True
                device=_self.model_config["device"]
            )
        except Exception as e:
            logging.error(f"Failed to load emotion model: {e}")
            return None
    
    @st.cache_resource
    def _load_voice_emotion_model(_self):
        """Load voice-specific emotion model"""
        try:
            # Use a model specifically trained on audio features
            return pipeline(
                "audio-classification",
                model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                device=_self.model_config["device"]
            )
        except Exception as e:
            logging.warning(f"Voice emotion model not available: {e}")
            return None
    
    @property
    def asr_pipe(self):
        if self._asr_pipe is None:
            self._asr_pipe = self._load_asr_model()
        return self._asr_pipe
    
    @property
    def emotion_pipe(self):
        if self._emotion_pipe is None:
            self._emotion_pipe = self._load_emotion_model()
        return self._emotion_pipe
    
    @property
    def voice_emotion_pipe(self):
        if self._voice_emotion_pipe is None:
            self._voice_emotion_pipe = self._load_voice_emotion_model()
        return self._voice_emotion_pipe
    
    def analyze_text(self, text: str) -> List[Dict]:
        """Analyze emotion in text with enhanced preprocessing"""
        if not text or not text.strip():
            return [{"label": "neutral", "score": 1.0}]
        
        # Preprocess text
        text = self._preprocess_text(text)
        
        if not self.emotion_pipe:
            return [{"label": "neutral", "score": 1.0}]
        
        try:
            # Split long text into chunks for better analysis
            chunks = self._split_text(text, max_length=512)
            all_scores = []
            
            for chunk in chunks:
                scores = self.emotion_pipe(chunk)
                all_scores.extend(scores)
            
            # Aggregate scores across chunks
            return self._aggregate_emotion_scores(all_scores)
            
        except Exception as e:
            logging.error(f"Text emotion analysis failed: {e}")
            return [{"label": "neutral", "score": 1.0}]
    
    def analyze_audio(self, audio_data, sample_rate: int = 16000) -> Tuple[str, List[Dict]]:
        """Enhanced audio analysis with voice emotion detection"""
        try:
            # Ensure audio is in correct format
            if isinstance(audio_data, bytes):
                audio_data = np.frombuffer(audio_data, dtype=np.float32)
            
            # Resample if needed
            if sample_rate != 16000:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            
            # Extract transcript
            transcript = ""
            if self.asr_pipe:
                transcript = self.asr_pipe(audio_data)["text"].strip()
            
            # Analyze transcript emotions
            text_emotions = self.analyze_text(transcript) if transcript else []
            
            # Analyze voice emotions (if model available)
            voice_emotions = []
            if self.voice_emotion_pipe:
                try:
                    voice_emotions = self.voice_emotion_pipe(audio_data)
                    # Convert to consistent format
                    voice_emotions = [{"label": item["label"], "score": item["score"]} 
                                    for item in voice_emotions]
                except Exception as e:
                    logging.warning(f"Voice emotion analysis failed: {e}")
            
            # Combine text and voice emotions with weighting
            combined_emotions = self._combine_emotion_sources(text_emotions, voice_emotions)
            
            return transcript, combined_emotions
            
        except Exception as e:
            logging.error(f"Audio analysis failed: {e}")
            return "", [{"label": "neutral", "score": 1.0}]
    
    def _preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing"""
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Handle common contractions
        contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "'re": " are", "'ve": " have", "'ll": " will", "'d": " would"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        return text
    
    def _split_text(self, text: str, max_length: int = 512) -> List[str]:
        """Split text into chunks for processing"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 > max_length and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _aggregate_emotion_scores(self, all_scores: List[List[Dict]]) -> List[Dict]:
        """Aggregate emotion scores from multiple chunks"""
        if not all_scores:
            return [{"label": "neutral", "score": 1.0}]
        
        # Flatten all scores
        flat_scores = [score for chunk_scores in all_scores for score in chunk_scores]
        
        # Group by emotion label
        emotion_groups = {}
        for score in flat_scores:
            label = score["label"]
            if label not in emotion_groups:
                emotion_groups[label] = []
            emotion_groups[label].append(score["score"])
        
        # Calculate weighted averages
        aggregated = []
        for label, scores in emotion_groups.items():
            avg_score = np.mean(scores)
            aggregated.append({"label": label, "score": avg_score})
        
        # Sort by score
        return sorted(aggregated, key=lambda x: x["score"], reverse=True)
    
    def _combine_emotion_sources(self, text_emotions: List[Dict], 
                                voice_emotions: List[Dict], 
                                text_weight: float = 0.7) -> List[Dict]:
        """Combine text and voice emotion analysis with weighting"""
        if not voice_emotions:
            return text_emotions
        
        if not text_emotions:
            return voice_emotions
        
        # Create emotion dictionaries for easier manipulation
        text_dict = {item["label"]: item["score"] for item in text_emotions}
        voice_dict = {item["label"]: item["score"] for item in voice_emotions}
        
        # Get all unique emotions
        all_emotions = set(text_dict.keys()) | set(voice_dict.keys())
        
        # Combine with weighting
        combined = []
        for emotion in all_emotions:
            text_score = text_dict.get(emotion, 0.0)
            voice_score = voice_dict.get(emotion, 0.0)
            
            # Weighted combination
            combined_score = (text_score * text_weight + 
                            voice_score * (1 - text_weight))
            
            combined.append({"label": emotion, "score": combined_score})
        
        # Sort by score
        return sorted(combined, key=lambda x: x["score"], reverse=True)
    
    def get_emotion_insights(self, emotions: List[Dict]) -> Dict:
        """Generate insights from emotion analysis"""
        if not emotions:
            return {"primary": "neutral", "confidence": 0.0, "insights": []}
        
        primary = emotions[0]
        insights = []
        
        # Confidence level insights
        if primary["score"] > 0.8:
            insights.append(f"High confidence in {primary['label']} emotion")
        elif primary["score"] > 0.6:
            insights.append(f"Moderate confidence in {primary['label']} emotion")
        else:
            insights.append("Mixed emotional signals detected")
        
        # Multiple emotions insights
        strong_emotions = [e for e in emotions if e["score"] > 0.3]
        if len(strong_emotions) > 1:
            insights.append(f"Multiple emotions present: {', '.join([e['label'] for e in strong_emotions[:3]])}")
        
        # Emotional complexity
        emotion_variance = np.var([e["score"] for e in emotions[:5]])
        if emotion_variance > 0.1:
            insights.append("Complex emotional state with varied intensities")
        
        return {
            "primary": primary["label"],
            "confidence": primary["score"],
            "insights": insights,
            "complexity": emotion_variance
        }