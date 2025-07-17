# config.py - Centralized configuration management
import os
from dataclasses import dataclass
from typing import Optional
import streamlit as st

@dataclass
class APIConfig:
    """API configuration with validation"""
    openrouter_key: Optional[str] = None
    tenor_key: Optional[str] = None
    weather_key: Optional[str] = None
    supabase_url: Optional[str] = None
    supabase_key: Optional[str] = None
    
    def __post_init__(self):
        """Load API keys from various sources"""
        # Try to load from key.txt first
        try:
            with open("key.txt", "r", encoding="utf-8") as f:
                content = f.read().strip()
                # Assume first line is OpenRouter key
                lines = content.split('\n')
                self.openrouter_key = lines[0] if lines else None
        except FileNotFoundError:
            pass
        
        # Load from environment variables (override file values)
        self.openrouter_key = os.getenv("OPENROUTER_KEY", self.openrouter_key)
        self.tenor_key = os.getenv("TENOR_API_KEY", "AIzaSyAARkiZD-Qo59Pji8ihow4hJCZDOsdchzE")
        self.weather_key = os.getenv("WEATHER_API_KEY", "24b359c76d994182864153220251507")
        self.supabase_url = os.getenv("SUPABASE_URL", "https://thvugbyogpuvcljlcplf.supabase.co")
        self.supabase_key = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRodnVnYnlvZ3B1dmNsamxjcGxmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDk1MjQxODgsImV4cCI6MjA2NTEwMDE4OH0.zM9jr8RXEqTDSjFcFhOVBzpi-iSg8-BkxvdULwqAxww")

@dataclass
class AppConfig:
    """Application configuration"""
    page_title: str = "🧠 Emotion AI - Advanced Analysis"
    page_icon: str = "🧠"
    layout: str = "wide"
    theme: str = "dark"
    max_audio_duration: int = 300  # 5 minutes
    max_text_length: int = 5000
    cache_ttl: int = 3600  # 1 hour
    
# Global config instances
api_config = APIConfig()
app_config = AppConfig()

# Validation functions
def validate_api_keys():
    """Validate required API keys and show warnings"""
    warnings = []
    
    if not api_config.openrouter_key:
        warnings.append("⚠️ OpenRouter API key missing - AI chat features disabled")
    
    if not api_config.tenor_key:
        warnings.append("⚠️ Tenor API key missing - GIF features disabled")
    
    return warnings

def get_model_config():
    """Get optimized model configuration based on available resources"""
    import torch
    
    device = 0 if torch.cuda.is_available() else -1
    
    return {
        "asr_model": "openai/whisper-base" if device == -1 else "openai/whisper-small",
        "emotion_model": "bhadresh-savani/bert-base-go-emotion",
        "device": device,
        "batch_size": 1 if device == -1 else 4
    }